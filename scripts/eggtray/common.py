"""Shared pieces of the egg-tray scripts: env construction, the scripted
walking routine, the three arm controllers, and the metric accumulator."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse  # noqa: E402

from src.tasks.eggtray.config.g1_23dof.env_cfgs import (  # noqa: E402
  CARRY, CARRY_12, TRAY_6, TRAY_12, unitree_g1_23dof_eggtray_env_cfg,
)
from src.tasks.eggtray.mdp.state import state  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# (name, t0, t1, vx, vy, wz): forward, stop, backward, turn, sideways, forward.
ROUTINE = [
  ("stand", 0.0, 3.0, 0.0, 0.0, 0.0),
  ("forward 0.4", 3.0, 9.0, 0.4, 0.0, 0.0),
  ("stop", 9.0, 11.0, 0.0, 0.0, 0.0),
  ("backward 0.3", 11.0, 16.0, -0.3, 0.0, 0.0),
  ("turn 0.5", 16.0, 20.0, 0.0, 0.0, 0.5),
  ("sideways 0.25", 20.0, 24.0, 0.0, 0.25, 0.0),
  ("forward 0.5", 24.0, 29.0, 0.5, 0.0, 0.0),
  ("stop", 29.0, 31.0, 0.0, 0.0, 0.0),
]
ROUTINE_S = ROUTINE[-1][2]

# Moderate shoves: (time s, dvx, dvy) added to the pelvis velocity in the
# robot's heading frame (m/s). DUET's own training pushes reach +-0.7 m/s.
PUSHES_MODERATE = [
  (1.5, 0.0, 0.40),    # standing, sideways
  (6.0, 0.0, -0.40),   # walking forward, sideways
  (10.0, 0.40, 0.0),   # stopped, from behind
  (13.5, 0.0, 0.35),   # walking backward, sideways
  (18.0, -0.30, 0.0),  # turning, from the front
  (22.0, 0.35, 0.0),   # stepping sideways, from behind
  (26.5, 0.0, 0.50),   # walking forward 0.5, sideways
  (30.0, -0.40, 0.0),  # stopped, from the front
]
PUSH_SETS = {"none": [], "moderate": PUSHES_MODERATE}


def apply_push(u, dvx, dvy):
  robot = u.scene["robot"]
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  vel = robot.data.root_link_vel_w.clone()
  vel[:, 0] += dvx * torch.cos(yaw) - dvy * torch.sin(yaw)
  vel[:, 1] += dvx * torch.sin(yaw) + dvy * torch.cos(yaw)
  robot.write_root_link_velocity_to_sim(vel)


def split_mode(spec: str) -> tuple[str, str]:
  """'rl' -> ('rl', '91500'); 'fixed@squatreach50k' -> ('fixed', 'squatreach50k')."""
  mode, _, duet = spec.partition("@")
  return mode, duet or "91500"


def use_duet(u, name: str) -> None:
  term = u.action_manager.get_term("duet_lower_body")
  if getattr(term, "policy_name", "91500") != name:
    term.use_policy(name)


def segment_at(t: float):
  for seg in ROUTINE:
    if seg[1] <= t < seg[2]:
      return seg
  return ROUTINE[-1]


def make_cfg(terrain="flat", eggs=6, contact="built_up", num_envs=16, litter=None,
             grasp="two_hand", scripted=True, duet_obs_noise=False):
  tray, carry = (TRAY_6, CARRY) if eggs == 6 else (TRAY_12, CARRY_12)
  if grasp != "two_hand":
    carry = replace(carry, grasp=grasp, feedforward=())
  cfg = unitree_g1_23dof_eggtray_env_cfg(
    play=True, terrain=terrain, tray=tray, carry=carry, floor_contact=contact,
    duet_obs_noise=duet_obs_noise,
  )
  cfg.scene.num_envs = num_envs
  if litter is not None:
    cfg.actions["poultry_surface"].litter_depth_range = (litter, litter)
  if scripted:
    tw = cfg.commands["twist"]
    tw.heading_command = False
    tw.ranges.heading = None
    tw.resampling_time_range = (1e9, 1e9)
    tw.rel_standing_envs = 0.0
    cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)
    cfg.events.pop("stop_backward_walkers", None)
  return cfg


def make_env(cfg, render=False, seed=0):
  cfg.seed = seed
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array" if render else None)
  return RslRlVecEnvWrapper(env, clip_actions=None)


def set_command(u, vx, vy, wz, height=0.78):
  tw = u.command_manager.get_term("twist")
  tw.vel_command_b[:, 0] = vx
  tw.vel_command_b[:, 1] = vy
  tw.vel_command_b[:, 2] = wz
  tw.is_standing_env[:] = (abs(vx) + abs(vy) + abs(wz)) == 0
  bh = u.command_manager.get_term("base_height")
  bh._squat_target[:, 0] = height
  bh._walk_target[:, 0] = height
  bh._height[:, 0] = height


# -- Arm controllers: obs (TensorDict) -> raw arm action [B, 10] ---------------


class FixedArm:
  """Baseline A: hold the carry reference (zero residual)."""
  name = "fixed"

  def __init__(self, u):
    self.n = u.num_envs
    self.device = u.device

  def __call__(self, obs):
    return torch.zeros(self.n, 10, device=self.device)

  def reset(self, ids=None):
    pass


class TrayLeveler:
  """Baseline B: conventional tray stabilisation from the tray IMU alone.

  Orientation: a leaky integral of the tilt-from-level (axis-angle ``e`` from
  the IMU attitude) plus rate damping on the roll/pitch gyro, commanded as a
  tray rotation. Vertical: a "sky-hook" loop -- the accelerometer's vertical
  acceleration is twice leaky-integrated to a displacement estimate and the
  tray is moved against it. The commanded tray twist is mapped to arm joint
  offsets with the carry geometry's least-squares map and passes through the
  same residual bound and rate limit as the RL policy. Gains hand-tuned on
  flat ground; see docs/eggtray_rl.md.
  """
  name = "feedback"

  def __init__(self, u, ki=4.0, kd=0.1, leak=0.5, kz=0.6, leak_v=1.5, leak_z=1.5):
    self.u = u
    self.arm = u.action_manager.get_term("arm_residual")
    self.M = self.arm.twist_to_joints  # [10, 6]
    self.ki, self.kd, self.leak, self.kz, self.leak_v, self.leak_z = ki, kd, leak, kz, leak_v, leak_z
    self.dt = u.step_dt
    self.theta = torch.zeros(u.num_envs, 3, device=u.device)
    self.v = torch.zeros(u.num_envs, device=u.device)
    self.z = torch.zeros(u.num_envs, device=u.device)
    self.torso = u.scene["robot"].find_bodies(("torso_link",))[0][0]

  def reset(self, ids=None):
    ids = slice(None) if ids is None else ids
    self.theta[ids] = 0
    self.v[ids] = 0
    self.z[ids] = 0

  def __call__(self, obs):
    u, dt = self.u, self.dt
    R = matrix_from_quat(u.scene["tray"].data.root_link_quat_w)  # IMU attitude estimate
    up = R[:, :, 2]
    e = torch.cross(up, torch.tensor([0.0, 0.0, 1.0], device=u.device).expand_as(up), dim=-1)
    w = torch.einsum("bij,bj->bi", R, u.scene["tray/tray_gyro"].data)
    w = w - up * (w * up).sum(-1, keepdim=True)  # keep roll/pitch rate, drop spin
    self.theta += (self.ki * e - self.leak * self.theta) * dt
    rot = self.theta - self.kd * w
    a_z = torch.einsum("bij,bj->bi", R, u.scene["tray/tray_acc"].data)[:, 2] - 9.81
    self.v += (a_z - self.leak_v * self.v) * dt
    self.z += (self.v - self.leak_z * self.z) * dt
    lin = torch.zeros_like(rot)
    lin[:, 2] = -self.kz * self.z
    tq = u.scene["robot"].data.body_link_quat_w[:, self.torso]
    twist = torch.cat([quat_apply_inverse(tq, lin), quat_apply_inverse(tq, rot)], -1)
    return (twist @ self.M.T) / self.arm.cfg.scale


class ArmPolicy:
  """Proposed: the trained residual policy (deterministic mean)."""
  name = "rl"

  def __init__(self, u, checkpoint):
    from src.tasks.eggtray.rl.runner import load_arm_policy

    self.policy = load_arm_policy(checkpoint, u.device)

  def reset(self, ids=None):
    pass

  def __call__(self, obs):
    with torch.no_grad():
      return self.policy(obs["actor"])


def make_controller(spec, u, checkpoint=None):
  """``spec`` is an arm mode, optionally with the DUET legs: 'rl', 'fixed@squatreach50k'.
  Also loads those legs into the env's frozen DUET term (91500 by default)."""
  mode, duet = split_mode(spec)
  use_duet(u, duet)
  if mode == "fixed":
    return FixedArm(u)
  if mode == "feedback":
    return TrayLeveler(u)
  if mode == "rl":
    if checkpoint is None:
      raise ValueError("--checkpoint is required for mode rl")
    return ArmPolicy(u, checkpoint)
  raise ValueError(mode)


# -- Metrics -------------------------------------------------------------------


@dataclass
class Accum:
  """Per-environment running sums over one routine (moving segments flagged)."""
  n: int
  device: str

  def __post_init__(self):
    z = lambda: torch.zeros(self.n, device=self.device)  # noqa: E731
    self.steps = z(); self.move_steps = z()
    self.acc2 = z(); self.acc_peak = z(); self.acc2_move = z()
    self.imu2 = z(); self.imu_peak = z()
    self.tilt2 = z(); self.tilt_max = z(); self.angvel2 = z(); self.jerk2 = z()
    self.force_peak = z(); self.excess_impulse = z(); self.egg_speed = z()
    self.lin_err = z(); self.yaw_err = z(); self.track_steps = z()
    self.dist = z(); self.ok = torch.ones(self.n, dtype=torch.bool, device=self.device)
    self.fell = torch.zeros(self.n, dtype=torch.bool, device=self.device)
    self.eggs_kept = torch.ones(self.n, device=self.device)
    self.residual = z(); self.strain_max = z(); self.extreme = z()
    self.eggs_over_10n = None; self.eggs_over_25n = None

  def update(self, u, moving: bool, settle: bool, fell, prev_xy):
    s = state(u).sample()
    alive = (~self.fell & ~fell).float()
    acc = s["acc"].norm(dim=-1)
    imu = s["acc_imu"].norm(dim=-1)
    tilt = torch.rad2deg(s["tilt"])
    kept = (~s["lost"]).float()
    f = s["egg_force"] * kept
    if self.eggs_over_10n is None:
      self.eggs_over_10n = torch.zeros_like(f, dtype=torch.bool)
      self.eggs_over_25n = torch.zeros_like(f, dtype=torch.bool)
    self.eggs_over_10n |= (f > 2.0) & (alive[:, None] > 0)
    self.eggs_over_25n |= (f > 5.0) & (alive[:, None] > 0)
    self.steps += alive
    self.acc2 += acc**2 * alive
    self.acc_peak = torch.maximum(self.acc_peak, acc * alive)
    self.imu2 += imu**2 * alive
    self.imu_peak = torch.maximum(self.imu_peak, imu * alive)
    self.tilt2 += tilt**2 * alive
    self.tilt_max = torch.maximum(self.tilt_max, tilt * alive)
    self.angvel2 += s["ang_vel_b"][:, :2].square().sum(-1) * alive
    self.jerk2 += s["jerk"].square().sum(-1) * alive
    self.force_peak = torch.maximum(self.force_peak, f.amax(-1) * alive)
    rest = 9.81 * 0.06
    self.excess_impulse += ((f - 2 * rest).clamp_min(0).sum(-1)) * u.step_dt * alive
    self.egg_speed += (s["egg_rel_vel"].norm(dim=-1) * kept).mean(-1) * alive
    self.residual += u.action_manager.get_term("arm_residual").residual.abs().mean(-1) * alive
    self.strain_max = torch.maximum(self.strain_max, s["grasp_strain"].amax(-1) * alive)
    if moving:
      self.move_steps += alive
      self.acc2_move += acc**2 * alive
    if not settle:
      robot = u.scene["robot"].data
      cmd = u.command_manager.get_command("twist")
      v = robot.root_link_lin_vel_b
      w = robot.root_link_ang_vel_b
      self.lin_err += (v[:, :2] - cmd[:, :2]).norm(dim=-1) * alive
      self.yaw_err += (w[:, 2] - cmd[:, 2]).abs() * alive
      self.track_steps += alive
    self.fell |= fell
    self.eggs_kept = torch.minimum(self.eggs_kept, kept.mean(-1) + self.fell.float() * 0)
    xy = u.scene["robot"].data.root_link_pos_w[:, :2]
    self.ok &= ~s["lost"].any(-1) & ~self.fell
    self.dist += (xy - prev_xy).norm(dim=-1) * self.ok.float()
    self.extreme = u.action_manager.get_term("duet_lower_body").extreme_count.clone()

  def per_env(self) -> dict[str, np.ndarray]:
    st = self.steps.clamp_min(1)
    out = {
      "acc_rms": (self.acc2 / st).sqrt(),
      "acc_rms_walking": (self.acc2_move / self.move_steps.clamp_min(1)).sqrt(),
      "acc_peak": self.acc_peak,
      "acc_imu_rms": (self.imu2 / st).sqrt(),
      "acc_imu_peak": self.imu_peak,
      "tilt_rms_deg": (self.tilt2 / st).sqrt(),
      "tilt_max_deg": self.tilt_max,
      "ang_vel_rms": (self.angvel2 / st).sqrt(),
      "jerk_rms": (self.jerk2 / st).sqrt(),
      "egg_force_peak_n": self.force_peak,
      "egg_excess_impulse_ns": self.excess_impulse,
      "egg_speed_mean": self.egg_speed / st,
      "eggs_over_2n": self.eggs_over_10n.float().sum(-1),
      "eggs_over_5n": self.eggs_over_25n.float().sum(-1),
      "all_eggs_kept": (self.eggs_kept >= 1.0).float(),
      "eggs_kept_frac": self.eggs_kept,
      "fell": self.fell.float(),
      "lin_vel_err": self.lin_err / self.track_steps.clamp_min(1),
      "yaw_rate_err": self.yaw_err / self.track_steps.clamp_min(1),
      "transport_m": self.dist,
      "residual_mean_rad": self.residual / st,
      "grasp_strain_max_mm": 1000 * self.strain_max,
      "duet_extreme_steps": self.extreme,
    }
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def run_routine(wrapped, controller, accum: Accum | None = None, frame_cb=None, routine=ROUTINE,
                height=0.78, pushes=()):
  """One scripted routine for every env. Falls are recorded, not reset early.
  ``pushes``: (time, dvx, dvy) shoves applied to every env at that time."""
  u = wrapped.unwrapped
  obs, _ = wrapped.reset()
  controller.reset()
  dt = u.step_dt
  total = routine[-1][2]
  push_at = {int(round(t / dt)): (dvx, dvy) for t, dvx, dvy in pushes}
  u._eggtray_last_push = -10**9
  for k in range(int(round(total / dt))):
    t = k * dt
    name, t0, _, vx, vy, wz = next((s for s in routine if s[1] <= t < s[2]), routine[-1])
    set_command(u, vx, vy, wz, height)
    if k in push_at:
      apply_push(u, *push_at[k])
      u._eggtray_last_push = k
    prev_xy = u.scene["robot"].data.root_link_pos_w[:, :2].clone()
    action = controller(obs)
    obs, _, dones, _ = wrapped.step(action)
    fell = u.termination_manager.get_term("fell_over")
    if accum is not None:
      accum.update(u, moving=(vx or vy or wz) != 0, settle=(t - t0) < 1.0, fell=fell, prev_xy=prev_xy)
    if dones.any():
      controller.reset(dones.nonzero().squeeze(-1))
    if frame_cb is not None:
      frame_cb(k, t, name, u)
  return accum


def summarize(per_env_runs: list[dict[str, np.ndarray]]) -> dict:
  """Mean over envs per seed, then mean and std across seeds; also pooled percentiles."""
  keys = per_env_runs[0].keys()
  out = {}
  for k in keys:
    seed_means = np.array([r[k].mean() for r in per_env_runs])
    pooled = np.concatenate([r[k] for r in per_env_runs])
    out[k] = {
      "mean": float(seed_means.mean()),
      "std_across_seeds": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
      "p10": float(np.percentile(pooled, 10)),
      "p90": float(np.percentile(pooled, 90)),
      "n_episodes": int(pooled.size),
    }
  return out


def fmt(x, nd=3):
  return f"{x:.{nd}f}" if isinstance(x, float) and math.isfinite(x) else str(x)
