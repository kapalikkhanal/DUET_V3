"""Split-level test: one foot on a raised strip, one on the lower ground.

  PYTHONPATH=. python scripts/eval_split_level.py CKPT [--task TASK] [--clearance 0.09]

Default task is the 47k lean task; pass --task Adaptive_Lift for its checkpoints.
Rigid strips 0.6 m wide alternate low and raised (heights below). For every
height, half the robots start straddling a seam with the LEFT foot up, half
with the RIGHT foot up; 0.6 kg per hand, arms still, no pushes. Scenarios run
one after another without a reset:

  placed    placed across the seam with the lower foot h above its ground,
            2 s settle then 4 s standing (does it keep the split stance?)
  along     walk 0.3 m/s along the seam (one foot up the whole time), 8 s
  stop      stand 4 s wherever that walk ended (still across the seam: the
            clean standing-on-a-step measurement)
  turn      turn 0.5 rad/s in place, 6 s (feet cross the seam)
  then from a reset on a low strip, facing across the strips:
  across    walk 0.3 m/s across the strips (step up / step down), 8 s
  side      sidestep 0.25 m/s across the strips, 8 s

Columns (means over robots that have not fallen, last part of each scenario):
  falls     robots that fell during the scenario (of N per height)
  track     achieved / commanded speed (%)
  tilt      torso tilt from upright (deg; the lean command is 0)
  pel roll  pelvis roll toward the lower foot (deg). With equal legs a 5 cm
            step over the ~0.24 m stance would roll the pelvis ~12 deg.
  knee dif  uphill minus downhill knee flexion while straddling (deg; > 0 means
            the uphill leg bends more, i.e. the legs absorb the step)
  slip      stance-foot horizontal speed (cm/s)
  straddle  robots with the feet on different levels at the end (%)
  settle    after 'stop': time until tilt stays within 2 deg of its final value (s)
"""

import argparse
import math
import os
from dataclasses import asdict, replace

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.duet.mdp.lift_terrain import SplitTile

HEIGHTS = (0., .02, .035, .05, .07)
PER_SIDE = 8
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-PoultrySurface-Lean")
ap.add_argument("--clearance", type=float, default=.09, help="Adaptive_Lift clearance command, m")
args = ap.parse_args()

cols = [(h, side) for h in HEIGHTS for side in (1, -1)]
N = len(cols) * PER_SIDE
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = 120.
gen = cfg.scene.terrain.terrain_generator
cfg.scene.terrain.terrain_generator = replace(
  gen, num_rows=1, num_cols=len(cols), curriculum=True, border_width=8.,
  sub_terrains={f"split_{i}": SplitTile(proportion=1., fixed_height=max(h, 1e-4), seam_side=s)
                for i, (h, s) in enumerate(cols)})
cfg.scene.terrain.max_init_terrain_level = 0
cfg.events["reset_base"].func = envs_mdp.reset_root_state_uniform
cfg.events["reset_base"].params = {"pose_range": {"yaw": (0., 0.)}, "velocity_range": {}}
for name in ("push_robot", "random_stops", "stop_backward_walkers", "randomize_terrain"):
  cfg.events.pop(name, None)
cfg.curriculum.pop("terrain_levels", None)
cfg.events["hand_payload"].params["ranges"] = (0.6, 0.6)
cfg.events["torso_payload"].params["ranges"] = (0.0, 0.0)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=30.0)
u = env.unwrapped
runner = load_runner_cls(args.task)(env, asdict(load_rl_cfg(args.task)), device=DEV)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

terrain = u.scene.terrain
col_of_env = torch.arange(N, device=DEV) // PER_SIDE
terrain.terrain_types[:] = col_of_env
terrain.terrain_levels[:] = 0
seam_origin = terrain.terrain_origins[0, col_of_env].clone()
height = torch.tensor([cols[c][0] for c in col_of_env.tolist()], device=DEV)
side = torch.tensor([cols[c][1] for c in col_of_env.tolist()], device=DEV)
width = cfg.scene.terrain.terrain_generator.size[1] / SplitTile(proportion=1., size=gen.size).strips()[0]
# A low strip next to the seam, for the runs that start facing across the strips.
low_origin = seam_origin.clone()
low_origin[:, 1] -= side * width / 2
low_origin[:, 2] = 0.

tw = u.command_manager.get_term("twist")
bh = u.command_manager.get_term("base_height")
ln = u.command_manager.get_term("lean")
cl = u.command_manager.get_term("clearance") if "clearance" in u.command_manager.active_terms else None
arm = u.action_manager.get_term("upper_body_pose")
robot = u.scene["robot"]
torso = robot.body_names.index("torso_link")
sites = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
knees = robot.find_joints(("left_knee_joint", "right_knee_joint"), preserve_order=True)[0]
contact = u.scene["feet_ground_contact"]
home = arm.default_pose.clone()
obs = None


def step(vx=0., vy=0., wz=0.):
  global obs
  for term in (tw, bh, ln):
    term.time_left[:] = 100.
  tw.is_standing_env[:] = vx == 0 and vy == 0 and wz == 0
  tw.is_heading_env[:] = False
  tw.vel_command_b[:] = torch.tensor([vx, vy, wz], device=DEV)
  bh._squat_target[:] = .78
  bh._walk_target[:] = .78
  ln.target[:] = 0.
  if cl is not None:
    cl.time_left[:] = 100.
    cl.goal[:] = args.clearance
    cl.value[:] = args.clearance
  arm.set_external_target(home)
  obs, _, done, _ = env.step(policy(obs))
  return done.bool()


def reset(origins, yaw):
  global obs
  terrain.env_origins[:] = origins
  u.event_manager.get_term_cfg("reset_base").params["pose_range"]["yaw"] = (yaw, yaw)
  env.reset()
  obs = env.get_observations()


def tilt_deg():
  g = quat_apply_inverse(robot.data.body_link_quat_w[:, torso], robot.data.gravity_vec_w)
  return torch.rad2deg(torch.asin(g[:, :2].norm(dim=1).clamp(max=1.)))


def pelvis_roll_deg():
  g = quat_apply_inverse(robot.data.root_link_quat_w, robot.data.gravity_vec_w)
  return torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2]))  # + = right side down


def stance_slip():
  v = robot.data.site_lin_vel_w[:, sites, :2].norm(dim=-1)
  planted = contact.data.found > 0
  return (v * planted).sum(-1) / planted.sum(-1).clamp_min(1)


def run(seconds, cmd=(0., 0., 0.), axis=None, measure_from=1.):
  """Step the scenario; average the columns over its last part."""
  steps, start = int(seconds * 50), int(measure_from * 50)
  fell = torch.zeros(N, dtype=torch.bool, device=DEV)
  acc = {k: torch.zeros(N, device=DEV) for k in ("track", "tilt", "roll", "knee", "slip")}
  tilts = []
  for k in range(steps):
    fell |= step(*cmd)
    tilts.append(tilt_deg())
    if k < start:
      continue
    if axis is not None:
      vel = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:]), -1)
      acc["track"] += vel[:, axis] / cmd[axis]
    acc["tilt"] += tilts[-1]
    acc["roll"] += pelvis_roll_deg() * side  # + = toward the lower foot
    q = robot.data.joint_pos[:, knees]
    acc["knee"] += torch.rad2deg(torch.where(side > 0, q[:, 0] - q[:, 1], q[:, 1] - q[:, 0]))
    acc["slip"] += 100 * stance_slip()
  out = {k: v / (steps - start) for k, v in acc.items()}
  out["fell"] = fell
  z = robot.data.site_pos_w[:, sites, 2]
  out["straddle"] = ((z[:, 0] - z[:, 1]).abs() > height / 2).float() * (height > 0)
  out["tilt_trace"] = torch.stack(tilts, 1)
  return out


def settle_time(trace):
  final = trace[:, -25:].mean(1, keepdim=True)
  outside = (trace - final).abs() > 2.
  idx = torch.arange(trace.shape[1], device=DEV).expand_as(trace)
  last = torch.where(outside, idx, torch.full_like(idx, -1)).amax(1)
  return (last + 1).float() / 50


def report(name, r, cols_shown):
  print(f"\n{name}")
  print("  height  | falls | " + " | ".join(f"{c:>8s}" for c in cols_shown))
  for h in HEIGHTS:
    sel = height == h
    alive = sel & ~r["fell"]
    vals = []
    for c in cols_shown:
      if c == "settle":
        v = settle_time(r["tilt_trace"])[alive].mean() if alive.any() else float("nan")
      else:
        key = {"track": "track", "tilt": "tilt", "pel roll": "roll", "knee dif": "knee", "slip": "slip",
               "straddle": "straddle"}[c]
        v = r[key][alive].mean() * (100 if c in ("track", "straddle") else 1) if alive.any() else float("nan")
      vals.append(f"{float(v):8.1f}")
    print(f"  {100 * h:4.1f} cm | {int(r['fell'][sel].sum()):2d}/{int(sel.sum())} | " + " | ".join(vals))


with torch.inference_mode():
  reset(seam_origin, 0.)
  r = run(6., measure_from=2.)
  report("placed across the seam, lower foot hanging (left or right foot up)", r, ("tilt", "pel roll", "knee dif", "slip", "straddle"))
  r = run(8., (.3, 0., 0.), axis=0, measure_from=2.)
  report("walk 0.3 m/s along the seam", r, ("track", "tilt", "pel roll", "slip", "straddle"))
  r = run(4., measure_from=2.)
  report("stop and stand where the walk ended", r, ("tilt", "settle", "pel roll", "knee dif", "slip", "straddle"))
  r = run(6., (0., 0., .5), axis=2, measure_from=2.)
  report("turn 0.5 rad/s in place across the seam", r, ("track", "tilt", "slip"))
  reset(low_origin, math.pi / 2)
  for _ in range(100):
    step()
  r = run(8., (.3, 0., 0.), axis=0, measure_from=1.)
  report("walk 0.3 m/s across the strips (step up / down every 0.6 m)", r, ("track", "tilt", "pel roll", "slip"))
  reset(low_origin, 0.)
  for _ in range(100):
    step()
  r = run(8., (0., .25 * 1., 0.), axis=1, measure_from=1.)
  report("sidestep 0.25 m/s across the strips", r, ("track", "tilt", "pel roll", "slip"))
