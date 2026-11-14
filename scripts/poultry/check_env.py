"""Pre-flight check for the PoultrySurface task. Run this before any long training.

  PYTHONPATH=. python scripts/poultry/check_env.py [--num-envs 256] [--device cuda:0]

It builds the training environment, steps it with small random actions and
checks the things that fail silently if they are wrong: observation layout and
history, the left-right mirror map, that the motor randomisation is really
applied (delay, gains, payloads), that the stop -> settle -> stand logic
switches when it should, and that the ramped schedules really move. Exits non-zero if any check fails.
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.actuator.delayed_actuator import DelayedActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from src.tasks.common.mdp.symmetry import check_symmetry
from src.assets.robots.unitree_g1.g1_23dof_deploy_constants import G1_23DOF_POULTRY_PD_GAINS
from src.tasks.duet.config.g1_23dof import poultry_surface as L
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import settle

TASK_ID = "Unitree-G1-23Dof-Duet-PoultrySurface"

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--device", default="cuda:0")
p.add_argument("--steps", type=int, default=150)
a = p.parse_args()

failures: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
  print(f"  [{'ok' if ok else 'FAIL'}] {what}{(' -- ' + detail) if detail else ''}")
  if not ok:
    failures.append(what)


cfg = load_env_cfg(TASK_ID)
cfg.scene.num_envs = a.num_envs
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=a.device), clip_actions=30.0)
u = env.unwrapped
robot = u.scene["robot"]
N = a.num_envs

print("\n== Interface ==")
om = u.observation_manager
frame = sum(d[0] for d in om.group_obs_term_dim["actor"]) // L.HISTORY
obs = env.get_observations()
check(obs["actor"].shape[1] == 71 * L.HISTORY, f"actor observation is {L.HISTORY} x 71 = {71 * L.HISTORY}",
      f"got {tuple(obs['actor'].shape)}, {frame} per frame")
check(env.num_actions == 13, "13 actions", f"got {env.num_actions}")
print(f"       critic observation: {obs['critic'].shape[1]} values")
print("       actor terms:", ", ".join(
  f"{n}:{d[0] // L.HISTORY}" for n, d in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"])))

print("\n== Mirror symmetry ==")
with torch.inference_mode():
  errs = check_symmetry(env, obs.to(u.device), torch.randn(N, 13, device=u.device))
check(max(errs.values()) < 1e-5, "mirror(mirror(x)) == x", str({k: f"{v:.1e}" for k, v in errs.items()}))

print("\n== Rollout with small random actions ==")
reward_sum = torch.zeros(N, device=u.device)
falls = 0
lag_first = None
with torch.inference_mode():
  for k in range(a.steps):
    obs, rew, dones, _ = env.step(0.3 * torch.randn(N, 13, device=u.device))
    reward_sum += rew
    falls += int(u.termination_manager.get_term("fell_over").sum())
    if k == 5:
      lag_first = torch.stack([
        act._delay_buffers["position"].current_lags.clone()
        for act in robot.actuators if isinstance(act, DelayedActuator)
      ])
check(bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()),
      "observations are finite")
check(bool(torch.isfinite(reward_sum).all()), "rewards are finite",
      f"mean per step {float(reward_sum.mean()) / a.steps:+.4f}, {falls} falls in {a.steps} steps x {N} robots")

print("\n== Motor randomisation ==")
delayed = [act for act in robot.actuators if isinstance(act, DelayedActuator)]
check(len(delayed) == len(robot.actuators) and len(delayed) == 6, "all 6 actuator groups are delayed")
assert lag_first is not None
lags = lag_first[0]
check(bool((lag_first == lags).all()), "one delay per robot, shared by all its joints")
check(int(lags.min()) == 0 and int(lags.max()) == L.DELAY_MAX_LAG,
      f"delay spans 0-{L.DELAY_MAX_LAG} physics steps (0-{5 * L.DELAY_MAX_LAG} ms)",
      "counts per lag: " + str(torch.bincount(lags, minlength=L.DELAY_MAX_LAG + 1).tolist()))

model = u.sim.model
mj_model = u.sim.mj_model
actuator_of_joint = {}
for i in range(mj_model.nu):
  joint = mj_model.joint(mj_model.actuator_trnid[i, 0]).name.split("/")[-1]
  actuator_of_joint[joint] = i


def gains(joint: str) -> tuple[torch.Tensor, torch.Tensor]:
  cid = actuator_of_joint[joint]
  return model.actuator_gainprm[:, cid, 0], -model.actuator_biasprm[:, cid, 2]


G = G1_23DOF_POULTRY_PD_GAINS
for joint, (kp0, kd0), spread in (
  ("left_hip_pitch_joint", G[".*_hip_pitch_joint"], 0.1),
  ("left_knee_joint", G[".*_knee_joint"], 0.1),
  ("left_ankle_pitch_joint", G[".*_ankle_pitch_joint"], 0.1),
  ("waist_yaw_joint", G["waist_yaw_joint"], 0.1),
  ("left_shoulder_pitch_joint", G[".*_shoulder_pitch_joint"], 0.5),
):
  kp, kd = gains(joint)
  lo, hi = float(kp.min()) / kp0, float(kp.max()) / kp0
  check(1 - spread - 0.02 < lo < 1 - 0.5 * spread and 1 + 0.5 * spread < hi < 1 + spread + 0.02,
        f"{joint} kp {kp0:g} randomised +-{int(spread * 100)}%",
        f"kp {float(kp.min()):.1f}-{float(kp.max()):.1f}, kd {float(kd.min()):.2f}-{float(kd.max()):.2f} (nominal {kd0:g})")
scale = u.action_manager.get_term("joint_pos").scale
check(bool(torch.allclose(torch.as_tensor(scale, dtype=torch.float32), torch.tensor(0.25))),
      "action scale is 0.25 rad on every policy joint")

mass = model.body_mass
default_mass = torch.as_tensor(mj_model.body_mass, device=mass.device, dtype=mass.dtype)


def mass_of(body: str) -> tuple[torch.Tensor, float]:
  bid = next(i for i in range(mj_model.nbody) if mj_model.body(i).name.split("/")[-1] == body)
  return mass[:, bid], float(default_mass[bid])


m, m0 = mass_of("left_wrist_roll_rubber_hand")
check(m0 + 0.24 < float(m.min()) and float(m.max()) < m0 + 1.76 and float(m.max() - m.min()) > 1.0,
      "hand payload +0.25 to +1.75 kg", f"{float(m.min()):.2f}-{float(m.max()):.2f} kg (bare hand {m0:.3f})")
m, m0 = mass_of("torso_link")
check(m0 - 0.51 < float(m.min()) and float(m.max()) < m0 + 2.01 and float(m.max() - m.min()) > 1.5,
      "torso payload -0.5 to +2.0 kg", f"{float(m.min()):.2f}-{float(m.max()):.2f} kg (bare {m0:.2f})")

print("\n== Stop -> settle -> stand ==")
twist = u.command_manager.get_term("twist")
gait = u.command_manager.get_term(settle.HELD)
height = u.command_manager.get_term("base_height")
phase_slice = None
off = 0
for name, d in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
  if name == "phase":
    phase_slice = slice(off + d[0] - 2, off + d[0])  # newest frame of the phase term
  off += d[0]


def hold_command(vx: float, steps: int):
  out = None
  for _ in range(steps):
    twist.time_left[:] = 100.0  # no resampling during the test
    twist.is_standing_env[:] = vx == 0.0
    twist.is_heading_env[:] = False
    twist.vel_command_b[:] = 0.0
    twist.vel_command_b[:, 0] = vx
    out, _, _, _ = env.step(torch.zeros(N, 13, device=u.device))
  return out


with torch.inference_mode():
  alive = torch.ones(N, dtype=torch.bool, device=u.device)
  o = hold_command(0.4, 10)
  alive &= u.episode_length_buf > 9  # ignore robots that fell and were reset
  check(bool(gait.moving[alive].all()), "walking command -> MOVING")
  check(bool((o["actor"][alive][:, phase_slice].norm(dim=1) > 0.99).all()), "gait clock visible while moving")
  half = int(0.5 * L.SETTLE_TIME / u.step_dt)
  o = hold_command(0.0, half)
  alive &= u.episode_length_buf > 9 + half - 1
  check(bool(gait.settling[alive].all()), f"{0.5 * L.SETTLE_TIME:.1f} s after the stop -> SETTLING")
  check(bool((o["actor"][alive][:, phase_slice].norm(dim=1) > 0.99).all()), "gait clock still visible while settling")
  check(bool(torch.equal(height.command[alive], height._walk_target[alive])),
        "height command stays at the walking height while settling")
  anchor = u.reward_manager.get_term_cfg("idle_position_anchor").func
  check(not bool(anchor.was_lin_idle[alive].any()), "position anchor not yet latched while settling")
  o = hold_command(0.0, half + 3)
  alive &= u.episode_length_buf > 9 + 2 * half
  check(bool(gait.standing[alive].all()), f"{L.SETTLE_TIME:.1f} s after the stop -> STANDING",
        f"{int(alive.sum())} of {N} robots stayed up through the test")
  check(bool((o["actor"][alive][:, phase_slice].abs().max() < 1e-6)), "gait clock reads zero while standing")
  check(bool(torch.equal(height.command[alive], height._squat_target[alive])),
        "height command switches to the squat target once standing")
  # A push re-opens the latch for one step, so a few robots may read False.
  latched = float(anchor.was_lin_idle[alive].float().mean())
  check(latched > 0.9, "position anchor latched at the end of settling", f"{100 * latched:.0f}% of robots")

print("\n== Schedules ==")
rm = u.reward_manager
surface = u.action_manager.get_term("poultry_surface")
ids = torch.arange(N, device=u.device)
FINAL = {n: s[-1][1] for n in L.LIFT_TERMS for s in [u.cfg.curriculum[f"weight_{n}"].params["stages"]]}
for it, frac, depth in ((0, 0.0, 0.02), (1_900, 0.5, 0.0264), (3_000, 1.0, 0.044), (12_500, 1.0, 0.06)):
  u.common_step_counter = it * ITER
  u.curriculum_manager.compute(ids)
  w = {n: rm.get_term_cfg(n).weight for n in L.LIFT_TERMS}
  ok = all(abs(w[n] - frac * FINAL[n]) < 1e-6 for n in w) and abs(surface.cfg.litter_depth_range[1] - depth) < 1e-4
  check(ok, f"iteration {it}: lift penalties at {int(100 * frac)}%, litter up to {100 * depth:.1f} cm",
        f"clearance weight {w['poultry_clearance']:+.2f}, depth {100 * surface.cfg.litter_depth_range[1]:.2f} cm, "
        f"squat floor {height.current_floor:.2f} m")
check(abs(height.current_floor - L.SQUAT_FLOOR) < 1e-6, f"squat floor ends at {L.SQUAT_FLOOR} m")
with torch.inference_mode():
  for _ in range(3):
    o, rew, _, _ = env.step(torch.zeros(N, 13, device=u.device))
check(bool(torch.isfinite(rew).all()), "rewards finite with the lift penalties at full weight")

print("\n== History layout ==")
off = 0
for name, d in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
  if name == "height_command":
    newest = o["actor"][:, off + d[0] - 1]
    cmd = u.command_manager.get_command("base_height")[:, 0]
    check(bool(torch.allclose(newest, cmd)), "newest frame is last in each term's history")
  off += d[0]

print()
if failures:
  print(f"{len(failures)} CHECK(S) FAILED:")
  for f in failures:
    print("  -", f)
  sys.exit(1)
print("All checks passed.")
env.close()
