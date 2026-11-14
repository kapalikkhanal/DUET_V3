"""Print a PoultrySurface training run as one row per N iterations.

  python scripts/poultry/progress.py [RUN_DIR] [--every 250]

RUN_DIR defaults to the newest folder in logs/rsl_rl/DUET_G1_23dof_poultry. Each
value is the mean over the ``--window`` iterations ending at that row.

Columns
  reward     mean episode return
  len s      mean episode length (20 s = the robot survived the whole episode)
  falls/min  falls per robot-minute during training (with pushes and noise)
  std        policy action noise
  lin / yaw  velocity tracking error (m/s, rad/s)
  h err      pelvis height error (m)
  apex cm    lowest point of the sole at the top of the swing, floor-relative
             (training average: standing and slope robots count as 0, so it
             reads below the real lift; use it for the trend)
  steps/s    foot touchdowns per second while a velocity is commanded
             (0 = not walking; a 0.85 s stride is 2.35)
  pitch      torso pitch while standing tall, deg (+ = leaning forward)   [91.5k robot: +5.9]
  knee       knee angle while standing tall, rad (default pose 0.30)      [91.5k robot: 0.70]
  stag cm    fore-aft foot stagger while standing
  step %     share of standing time with a foot off the ground (balance steps)
  |a|max     largest raw policy output (deploy guard trips above 15)
  terrain    mean terrain level (0-9)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("run_dir", nargs="?", default=None)
p.add_argument("--every", type=int, default=250)
p.add_argument("--window", type=int, default=20)
p.add_argument("--num-envs", type=int, default=4096)
a = p.parse_args()

root = Path(__file__).resolve().parents[2] / "logs" / "rsl_rl" / "DUET_G1_23dof_poultry"
run = Path(a.run_dir) if a.run_dir else max(root.iterdir(), key=lambda d: d.stat().st_mtime)
acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]


def series(*suffixes: str):
  """(steps, values) of the first tag ending with one of ``suffixes``."""
  for suffix in suffixes:
    for tag in tags:
      if tag.endswith(suffix):
        ev = acc.Scalars(tag)
        return np.array([e.step for e in ev]), np.array([e.value for e in ev])
  return np.array([]), np.array([])


def at(name_suffixes, it: int) -> float:
  steps, vals = series(*name_suffixes)
  m = (steps <= it) & (steps > it - a.window)
  return float(vals[m].mean()) if m.any() else float("nan")


M = "Episode_Metrics/"
steps, _ = series("Train/mean_reward")
if len(steps) == 0:
  sys.exit(f"no training scalars in {run} yet")
last = int(steps.max())
print(f"{run.name}: iteration {last}\n")
print("|  iter | reward | len s | falls/min |  std |  lin |  yaw | h err | apex cm | pitch | knee | stag cm | step % | |a|max | terrain | steps/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = list(range(a.every, last + 1, a.every))
if not rows or rows[-1] != last:
  rows.append(last)
for it in rows:
  stand = max(at((M + "standing_fraction",), it), 1e-6)
  tall = at((M + "standing_tall_fraction",), it)
  tall = max(tall, 1e-6)
  # steps_per_second is zero unless a velocity is commanded.
  steps_s = at((M + "steps_per_second",), it) / max(at((M + "moving_fraction",), it), 1e-6)
  # Episode_Termination/fell_over = robots that fell per policy step (50 Hz).
  falls = at(("Episode_Termination/fell_over",), it) * 50 * 60 / a.num_envs
  print(
    f"| {it:5d} | {at(('Train/mean_reward',), it):6.1f} | {at(('Train/mean_episode_length',), it) * 0.02:5.1f} "
    f"| {falls:9.2f} | {at(('Policy/mean_noise_std', 'mean_noise_std', 'mean_std'), it):4.2f} "
    f"| {at((M + 'lin_vel_track_error',), it):4.2f} | {at((M + 'yaw_track_error',), it):4.2f} "
    f"| {at((M + 'height_track_error',), it):5.3f} | {100 * at((M + 'poultry_swing_apex_m',), it):7.2f} "
    f"| {at((M + 'stand_pitch_deg',), it) / tall:+5.1f} | {at((M + 'stand_knee_rad',), it) / tall:4.2f} "
    f"| {100 * at((M + 'stand_stagger_m',), it) / stand:7.1f} | {100 * at((M + 'stand_foot_air',), it) / stand:6.1f} "
    f"| {at((M + 'action_abs_max',), it):6.1f} | {at(('Curriculum/terrain_levels',), it):7.2f} | {steps_s:7.2f} |"
  )
