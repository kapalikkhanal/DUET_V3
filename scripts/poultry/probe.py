"""Measure what a PoultrySurface checkpoint actually does, on a flat floor.

  PYTHONPATH=. python scripts/poultry/probe.py logs/rsl_rl/DUET_G1_23dof_poultry/<run>/model_5000.pt

128 robots, no pushes, arms still, 0.6 kg per hand. Safe to run while a
training is going on the same GPU (takes about two minutes). Prints:

  Standing   torso pitch (+ = leaning forward), knee angle, drift, balance steps
  Walking    achieved speed for forward / backward / sideways / turn commands,
             steps per second (a 0.85 s stride is 2.35), foot lift, falls
  Stopping   speed and foot stagger after the command drops from 0.4 m/s to 0

What to look for at 5,000 iterations: standing pitch within +-3 deg; sideways
and turn commands achieved to at least ~70%; no falls; stagger after stopping
under ~3 cm.
"""
import os, sys, math
from dataclasses import asdict
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import mjlab.tasks, src.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, matrix_from_quat
TASK = "Unitree-G1-23Dof-Duet-PoultrySurface"; DEV = "cuda:0"; N = 128
ckpt = sys.argv[1]
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = N
cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
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
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)
tw = u.command_manager.get_term("twist"); bh = u.command_manager.get_term("base_height")
robot = u.scene["robot"]; sensor = u.scene["feet_ground_contact"]
torso = robot.find_bodies(("torso_link",))[0][0]
sites = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
jn = robot.joint_names
J = {k: jn.index(k) for k in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint")}
act = u.action_manager.get_term("joint_pos")

def step(vx, vy, wz, h=0.78):
  tw.time_left[:] = 100.; bh.time_left[:] = 100.
  tw.is_standing_env[:] = (vx == 0 and vy == 0 and wz == 0); tw.is_heading_env[:] = False
  tw.vel_command_b[:, 0] = vx; tw.vel_command_b[:, 1] = vy; tw.vel_command_b[:, 2] = wz
  bh._squat_target[:] = h; bh._walk_target[:] = h
  global obs
  obs, _, done, _ = env.step(policy(obs))
  return done

def pitch_deg():
  q = robot.data.body_link_quat_w[:, torso]
  g = quat_apply_inverse(q, robot.data.gravity_vec_w)
  a = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
  # independent: where a point 0.4 m up the torso sits relative to the torso origin, along heading
  up = quat_apply(q, torch.tensor([0., 0., 0.4], device=DEV).expand(N, 3))
  fwd = quat_apply(q, torch.tensor([1., 0., 0.], device=DEV).expand(N, 3)); fwd[:, 2] = 0; fwd = fwd / fwd.norm(dim=1, keepdim=True)
  return a, (up * fwd).sum(1)  # deg (+ forward), metres the chest top is ahead (+) of the waist

obs = env.get_observations()
print("== Standing 6 s at 0.78 m ==")
with torch.inference_mode():
  for k in range(100): step(0, 0, 0)
  p0 = robot.data.root_link_pos_w[:, :2].clone(); air = 0.; P = []; X = []; Q = {k: [] for k in J}; falls = 0
  for k in range(300):
    falls += int(step(0, 0, 0).sum())
    a, x = pitch_deg(); P.append(a); X.append(x)
    air += float((sensor.data.current_contact_time <= 0).any(1).float().mean())
    for k2, i in J.items(): Q[k2].append(robot.data.joint_pos[:, i])
  P = torch.stack(P); X = torch.stack(X)
  drift = (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1)
  print(f"torso pitch {float(P.mean()):+.1f} deg (+ = forward), robot-to-robot range {float(P.mean(0).min()):+.1f}..{float(P.mean(0).max()):+.1f}; sway (std over time) {float(P.std(0).mean()):.2f} deg")
  print(f"chest top is {100 * float(X.mean()):+.1f} cm ahead of the waist (+ = leaning forward, independent check)")
  print("joint angles: " + ", ".join(f"{k.replace('_joint','')} {float(torch.stack(v).mean()):+.2f}" for k, v in Q.items()) + "   (default hip -0.10 knee 0.30 ankle -0.20)")
  print(f"pelvis height {float(robot.data.root_link_pos_w[:, 2].mean()):.3f} m; drift in 6 s {100 * float(drift.mean()):.1f} cm (max {100 * float(drift.max()):.1f}); a foot off the ground {100 * air / 300:.1f}% of the time; falls {falls}")

print("\n== Walking (5 s each, measured over the last 3 s) ==")
for label, (vx, vy, wz) in {"forward 0.2": (0.2,0,0), "forward 0.4": (0.4,0,0), "forward 0.6": (0.6,0,0), "backward 0.3": (-0.3,0,0), "sideways 0.25": (0,0.25,0), "turn 0.6": (0,0,0.6)}.items():
  with torch.inference_mode():
    for k in range(50): step(0, 0, 0)
    acc = dict(vx=0., vy=0., wz=0., n=0); lift = torch.zeros(N, device=DEV); was = torch.zeros(N, 2, dtype=torch.bool, device=DEV)
    apex = torch.zeros(N, 2, device=DEV); z0 = robot.data.site_pos_w[:, sites, 2].clone(); P = []; amax = 0.; falls = 0; dbl = 0.
    for k in range(250):
      falls += int(step(vx, vy, wz).sum())
      if k < 100: continue
      a = sensor.data.current_contact_time <= 0
      lift += (a & ~was).sum(1); was = a
      apex = torch.maximum(apex, robot.data.site_pos_w[:, sites, 2] - z0)
      v = robot.data.root_link_lin_vel_b; w = robot.data.root_link_ang_vel_b
      acc["vx"] += float(v[:, 0].mean()); acc["vy"] += float(v[:, 1].mean()); acc["wz"] += float(w[:, 2].mean()); acc["n"] += 1
      P.append(pitch_deg()[0]); amax = max(amax, float(act.raw_action.abs().max())); dbl += float(a.all(1).float().mean())
    n = acc["n"]
    print(f"{label:14s} -> vx {acc['vx']/n:+.2f} vy {acc['vy']/n:+.2f} yaw {acc['wz']/n:+.2f} | steps/s {float(lift.mean())/3:.2f} | foot lift {100*float(apex.mean()):.1f} cm | pitch {float(torch.stack(P).mean()):+.1f} deg | both feet in air {100*dbl/n:.1f}% | |a|max {amax:.1f} | falls {falls}")

print("\n== Stop from 0.4 m/s ==")
with torch.inference_mode():
  for k in range(200): step(0.4, 0, 0)
  last_air = torch.zeros(N, device=DEV); rows = []
  for k in range(200):
    step(0, 0, 0)
    a = (sensor.data.current_contact_time <= 0).any(1)
    last_air = torch.where(a, torch.full_like(last_air, (k + 1) * 0.02), last_air)
    if k + 1 in (25, 50, 75, 100, 150, 200):
      f = robot.data.site_pos_w[:, sites]; off = quat_apply_inverse(robot.data.root_link_quat_w, f[:, 0] - f[:, 1])
      sp = robot.data.root_link_lin_vel_b[:, :2].norm(dim=1)
      rows.append(f"  t={0.02*(k+1):.1f}s speed {float(sp.mean()):.3f} m/s, foot stagger {100*float(off[:,0].abs().mean()):.1f} cm (max {100*float(off[:,0].abs().max()):.1f}), pitch {float(pitch_deg()[0].mean()):+.1f} deg")
  print("\n".join(rows))
  print(f"last foot down: median {float(last_air.median()):.2f} s, 90% by {float(last_air.quantile(0.9)):.2f} s, still stepping after 3 s: {100*float((last_air > 3.0).float().mean()):.0f}% of robots")
