"""Measure the forward lean of a PoultrySurface-Lean checkpoint and how low the hands get.

  PYTHONPATH=. python scripts/poultry/probe_lean.py logs/rsl_rl/DUET_G1_23dof_poultry/<run>/model_N.pt

64 robots on a flat floor, 0.6 kg per hand. For each (pelvis height, lean)
pair the robot stands, leans, lets its arms hang straight down, holds for 3 s
and stands up again. Printed: the lean and height reached, how far the wrist
and a point 15 cm beyond it (about where Dex3 fingertips are) stay above the
floor, how far in front of the ankles that point is, drift, and falls.
"""

import os
import sys
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

TASK = "Unitree-G1-23Dof-Duet-PoultrySurface-Lean"
DEV, N = "cuda:0", 64
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
cfg.events["reset_base"].func = envs_mdp.reset_root_state_uniform
cfg.events["reset_base"].params = {"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}}
cfg.events.pop("randomize_terrain", None)
cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
del cfg.observations["critic"].terms["height_scan"]
cfg.events["hand_payload"].params["ranges"] = (0.6, 0.6)
cfg.events["torso_payload"].params["ranges"] = (0.0, 0.0)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=30.0)
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=DEV)
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

tw = u.command_manager.get_term("twist")
bh = u.command_manager.get_term("base_height")
ln = u.command_manager.get_term("lean")
arm = u.action_manager.get_term("upper_body_pose")
robot = u.scene["robot"]
bn = robot.body_names
hand = bn.index("left_wrist_roll_rubber_hand")
elbow = [i for i, n in enumerate(bn) if n.startswith("left_elbow")][0]
ankle = bn.index("left_ankle_roll_link")
torso = bn.index("torso_link")
home = arm.default_pose.clone()


def hang(lean: float) -> torch.Tensor:
  """Arm targets that point both arms straight down when the torso leans by ``lean``."""
  goal = home.clone()
  for j, n in enumerate(arm.joint_names):
    if "shoulder_pitch" in n:
      goal[:, j] = -lean
    elif "elbow" in n:
      goal[:, j] = 1.4
    elif "shoulder_roll" in n:
      goal[:, j] = 0.25 if n.startswith("left") else -0.25
    else:
      goal[:, j] = 0.0
  return goal


def step(h: float, lean: float, arms: torch.Tensor) -> torch.Tensor:
  global obs
  tw.time_left[:] = 100.0
  bh.time_left[:] = 100.0
  ln.time_left[:] = 100.0
  tw.is_standing_env[:] = True
  tw.is_heading_env[:] = False
  tw.vel_command_b[:] = 0.0
  bh._squat_target[:] = h
  bh._walk_target[:] = h
  ln.target[:] = lean
  arm.set_external_target(arms)
  obs, _, done, _ = env.step(policy(obs))
  return done.bool()


obs = env.get_observations()
print("height cmd | lean cmd | lean reached | pelvis | wrist above floor | tip above floor | tip ahead of ankle | drift | fell while leaning | fell standing up")
for h, lean in ((0.78, 0.0), (0.50, 0.0), (0.50, 0.5), (0.50, 0.9), (0.50, 1.25), (0.44, 1.25), (0.58, 1.25)):
  with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    fell = torch.zeros(N, dtype=torch.bool, device=DEV)
    for _ in range(100):
      step(0.78, 0.0, home)
    p0 = robot.data.root_link_pos_w[:, :2].clone()
    for k in range(250):  # go down and lean
      fell |= step(h, lean, home)
    for k in range(250):  # arms down, hold
      fell |= step(h, lean, home + min(1.0, k / 100.0) * (hang(lean) - home))
    g = quat_apply_inverse(robot.data.body_link_quat_w[:, torso], robot.data.gravity_vec_w)
    pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
    hp, ep, ap = (robot.data.body_link_pos_w[:, i] for i in (hand, elbow, ankle))
    fwd = hp - ep
    tip = hp + 0.15 * fwd / fwd.norm(dim=1, keepdim=True)
    heading = quat_apply(robot.data.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=DEV).expand(N, 3))
    heading[:, 2] = 0
    heading = heading / heading.norm(dim=1, keepdim=True)
    ok = ~fell
    n_ok = max(int(ok.sum()), 1)
    row = (
      f"   {h:.2f}    |  {lean:4.2f} rad ({lean * 57.3:2.0f} deg) | {float(pitch[ok].sum()) / n_ok:+6.1f} deg | "
      f"{float(robot.data.root_link_pos_w[ok, 2].sum()) / n_ok:.3f} | {100 * float(hp[ok, 2].sum()) / n_ok:9.1f} cm      | "
      f"{100 * float(tip[ok, 2].sum()) / n_ok:7.1f} cm      | {100 * float(((tip - ap) * heading).sum(1)[ok].sum()) / n_ok:8.1f} cm        | "
      f"{100 * float((robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1)[ok].sum()) / n_ok:4.1f} cm | {int(fell.sum()):3d} of {N}          |"
    )
    fell_up = torch.zeros(N, dtype=torch.bool, device=DEV)
    for k in range(250):  # arms home, stand up
      fell_up |= step(0.78, 0.0, home)
    print(row + f" {int((fell_up & ~fell).sum()):3d} of {N}")
