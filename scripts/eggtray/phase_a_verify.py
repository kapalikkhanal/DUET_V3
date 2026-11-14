"""Phase A: verify the egg-tray environment before any arm learning.

Checks, on the frozen DUET checkpoint with the arms held at the carry pose:

  1. the DUET checkpoint's SHA-256 before and after (unchanged);
  2. the in-env DUET network reproduces the exported ONNX on live observations;
  3. every lower-body joint target equals DUET's own output (scale/offset/bias),
     including while the arm action is driven with large random values;
  4. arm targets stay within carry reference +- max residual, the soft joint
     limits and the rate limit;
  5. stand / walk forward / stop / backward / turn / sideways with six eggs:
     no NaNs, no contact or constraint overflow, eggs seated, grasp strain.

``--calibrate`` measures the gravity feed-forward for the arm PD targets
(fixed-point iteration while standing) and prints the constant for
``CARRY_FEEDFORWARD`` in ``src/tasks/eggtray/config/g1_23dof/env_cfgs.py``.

  PYTHONPATH=. python scripts/eggtray/phase_a_verify.py [--terrain flat|poultry] [--n 32] [--calibrate]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.tasks.eggtray.frozen_duet import (  # noqa: E402
  DUET_CHECKPOINT, DUET_CHECKPOINT_SHA256, DUET_EXPORT_DIR, resolve, sha256_file,
)


def onnx_check(u, n=8) -> float:
  import onnxruntime as ort

  sess = ort.InferenceSession(str(resolve(DUET_EXPORT_DIR) / "policy.onnx"), providers=["CPUExecutionProvider"])
  duet = u.action_manager.get_term("duet_lower_body")
  obs = u.obs_buf["duet"][:n]
  with torch.no_grad():
    ours = duet.policy(obs).cpu().numpy()
  x = obs.cpu().numpy().astype(np.float32)
  ref = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(len(x))])
  return float(np.abs(ours - ref).max())


def leg_target_error(u) -> float:
  duet = u.action_manager.get_term("duet_lower_body")
  robot = u.scene["robot"]
  ids = duet.joint_ids
  expected = duet._joint._processed_actions - robot.data.encoder_bias[:, ids]
  # An env reset inside this step has its targets zeroed by the entity reset
  # until the next apply (standard mjlab behaviour); compare the others.
  live = ~u.reset_buf
  if not live.any():
    return 0.0
  return float((robot.data.joint_pos_target[live][:, ids] - expected[live]).abs().max())


def _settle(wrapped, settle_s):
  """Stand still; return the mean arm angles and torso-frame tray tilt of the last second."""
  u = wrapped.unwrapped
  arm = u.action_manager.get_term("arm_residual")
  torso = u.scene["robot"].find_bodies(("torso_link",))[0][0]
  wrapped.reset()
  qs, tilts = [], []
  for k in range(int(settle_s / u.step_dt)):
    C.set_command(u, 0, 0, 0)
    wrapped.step(torch.zeros(u.num_envs, 10, device=u.device))
    if k * u.step_dt > settle_s - 1.0:
      s = C.state(u).sample()
      up = s["R"][:, :, 2]
      e = torch.cross(up, torch.tensor([0.0, 0.0, 1.0], device=u.device).expand_as(up), dim=-1)
      tq = u.scene["robot"].data.body_link_quat_w[:, torso]
      tilts.append(C.quat_apply_inverse(tq, e).mean(0))
      qs.append(u.scene["robot"].data.joint_pos[:, arm.joint_ids].mean(0))
  return torch.stack(qs).mean(0), torch.stack(tilts).mean(0), torch.rad2deg(C.state(u).sample()["tilt"]).mean()


def calibrate(wrapped, joint_iters=4, level_iters=3, settle_s=4.0):
  """Stage 1: gravity feed-forward so the loaded arms settle at the carry pose.
  Stage 2: rotate the tray (through the tray-twist map) until it is level in
  the world while DUET stands, which absorbs DUET's torso lean under load.
  The fixed-arm baseline therefore starts from the best fixed pose."""
  u = wrapped.unwrapped
  arm = u.action_manager.get_term("arm_residual")
  ff = torch.zeros(10, device=u.device)
  for it in range(joint_iters):
    arm.base_reference[:] = arm.q_nominal + ff
    q_ss, _, tilt = _settle(wrapped, settle_s)
    err = arm.q_nominal[0] - q_ss
    print(f"  joint iter {it}: max |q_nominal - q_settled| = {err.abs().max():.4f} rad, tray tilt {tilt:.2f} deg")
    ff = ff + err
  for it in range(level_iters):
    arm.base_reference[:] = arm.q_nominal + ff
    _, e_torso, tilt = _settle(wrapped, settle_s)
    print(f"  level iter {it}: tray tilt {tilt:.2f} deg")
    ff = ff + arm.twist_to_joints[:, 3:6] @ e_torso
  arm.base_reference[:] = arm.q_nominal + ff
  _, _, tilt = _settle(wrapped, settle_s)
  print(f"  final: tray tilt standing {tilt:.2f} deg")
  return {n: round(float(v), 4) for n, v in zip(arm.joint_names, ff)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--terrain", default="flat", choices=("flat", "poultry"))
  ap.add_argument("--eggs", type=int, default=6, choices=(6, 12))
  ap.add_argument("--grasp", default="two_hand", choices=("two_hand", "right_hand"))
  ap.add_argument("--n", type=int, default=32)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--calibrate", action="store_true")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()

  report: dict = {}
  sha_before = sha256_file(resolve(DUET_CHECKPOINT))
  report["duet_sha256_before_ok"] = sha_before == DUET_CHECKPOINT_SHA256
  cfg = C.make_cfg(a.terrain, a.eggs, num_envs=a.n, grasp=a.grasp)
  wrapped = C.make_env(cfg, seed=a.seed)
  u = wrapped.unwrapped
  am = u.action_manager
  report["action_terms"] = dict(zip(am.active_terms, am.action_term_dim))
  report["agent_action_dim"] = am.total_action_dim
  report["duet_obs_terms"] = list(zip(u.observation_manager.active_terms["duet"],
                                      [int(d[0]) for d in u.observation_manager.group_obs_term_dim["duet"]]))
  report["actor_obs_dim"] = int(sum(d[0] for d in u.observation_manager.group_obs_term_dim["actor"]))
  report["critic_obs_dim"] = int(sum(d[0] for d in u.observation_manager.group_obs_term_dim["critic"]))
  print(json.dumps({k: report[k] for k in ("action_terms", "agent_action_dim", "actor_obs_dim", "critic_obs_dim")}))

  if a.calibrate:
    ff = calibrate(wrapped)
    print("CARRY_FEEDFORWARD = (")
    for n, v in ff.items():
      print(f'  ("{n}", {v}),')
    print(")")
    report["feedforward"] = ff

  arm = am.get_term("arm_residual")
  onnx_err, leg_err, nan_steps = [], [], 0
  max_nacon, max_nefc = 0, 0
  naconmax = int(u.sim._wp_data.naconmax)
  acc = C.Accum(u.num_envs, u.device)
  rate_ok, bound_ok = True, True

  def frame(k, t, name, uu):
    nonlocal nan_steps, max_nacon, max_nefc, rate_ok, bound_ok
    if k % 25 == 0:
      onnx_err.append(onnx_check(uu))
    leg_err.append(leg_target_error(uu))
    if not torch.isfinite(uu.sim.data.qpos).all():
      nan_steps += 1
    max_nacon = max(max_nacon, int(uu.sim.data.nacon.max()))
    max_nefc = max(max_nefc, int(uu.sim.data.nefc.max()))

  C.run_routine(wrapped, C.FixedArm(u), acc, frame_cb=frame)
  per = acc.per_env()
  report["routine_fixed_arm"] = {k: float(v.mean()) for k, v in per.items()}

  # Random, large arm actions: the legs must still follow DUET exactly and the
  # arm targets must respect every bound.
  obs, _ = wrapped.reset()
  prev = arm.target.clone()
  for k in range(150):
    C.set_command(u, 0.3, 0.0, 0.0)
    act = (torch.rand(u.num_envs, 10, device=u.device) * 2 - 1) * 10.0
    obs, *_ = wrapped.step(act)
    leg_err.append(leg_target_error(u))
    dev = (arm.target - arm.reference).abs().max().item()
    step = (arm.target - prev).abs().max().item()
    prev = arm.target.clone()
    in_limits = bool(((arm.target - arm._lo) > -1e-6).all() and ((arm._hi - arm.target) > -1e-6).all())
    bound_ok &= dev <= arm.cfg.max_residual + 1e-5 and in_limits
    rate_ok &= step <= arm.cfg.max_rate * u.step_dt + 1e-5
    if not torch.isfinite(u.sim.data.qpos).all():
      nan_steps += 1
  s = C.state(u).sample()
  report["random_arm_actions"] = {
    "eggs_kept_frac": float((~s["lost"]).float().mean()),
    "residual_bound_ok": bool(bound_ok),
    "rate_limit_ok": bool(rate_ok),
  }
  report["onnx_max_abs_diff"] = max(onnx_err)
  report["leg_target_max_abs_err"] = max(leg_err)
  report["nan_steps"] = nan_steps
  report["max_contacts_total"] = max_nacon
  report["naconmax_total"] = naconmax
  report["max_constraints_per_world"] = max_nefc
  report["njmax"] = cfg.sim.njmax
  report["duet_sha256_after_ok"] = sha256_file(resolve(DUET_CHECKPOINT)) == DUET_CHECKPOINT_SHA256
  print(json.dumps(report, indent=1))
  if a.json:
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
  main()
