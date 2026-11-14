"""Render the egg-tray task: one robot, scripted routine, telemetry overlay.

Every frame is labelled with the segment, the command, tray tilt, tray
acceleration and eggs kept. ``--compare fixed rl`` runs the listed modes with
the same seed and puts them side by side; ``--closeup`` frames the tray.

  PYTHONPATH=. python scripts/eggtray/play.py --terrain poultry --compare fixed rl \
      --checkpoint logs/rsl_rl/EggTray_G1_23dof/<run>/model_N.pt --out logs/eggtray/videos/compare.mp4
  PYTHONPATH=. python scripts/eggtray/play.py --mode rl --checkpoint ... --closeup --out closeup.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

LABEL = {
  "fixed": "DUET 91500 + fixed arms",
  "feedback": "DUET 91500 + feedback leveller",
  "rl": "DUET 91500 + arm RL",
  "fixed@squatreach50k": "DUET 50k (squat-reach) + fixed arms",
  "rl@squatreach50k": "DUET 50k (squat-reach) + arm RL",
}


def _font(size):
  for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
    if os.path.exists(path):
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


def record(mode, a, seed):
  cfg = C.make_cfg(a.terrain, a.eggs, a.contact, num_envs=1, litter=a.litter)
  cfg.viewer.width, cfg.viewer.height = a.width, a.height
  if a.closeup:
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 1.1, -38.0, 160.0
  cfg.events["reset_base"].params["pose_range"] = {"x": (0, 0), "y": (0, 0), "z": (0, 0.01), "yaw": (0, 0)}
  wrapped = C.make_env(cfg, render=True, seed=seed)
  u = wrapped.unwrapped
  u.seed(seed)
  ctrl = C.make_controller(mode, u, a.checkpoint)
  frames = []
  font, small = _font(22), _font(17)

  def cb(k, t, name, uu):
    if k % 2:
      return
    s = C.state(uu).sample()
    img = Image.fromarray(uu.render())
    d = ImageDraw.Draw(img)
    cmd = uu.command_manager.get_command("twist")[0].tolist()
    d.rectangle([0, 0, img.width, 92], fill=(255, 255, 255))
    d.text((14, 8), LABEL.get(mode, mode), fill=(20, 20, 20), font=font)
    d.text((14, 38), f"{name}   cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}   t {t:4.1f} s",
           fill=(60, 60, 60), font=small)
    kept = int((~s["lost"][0]).sum())
    if 0 <= k - uu._eggtray_last_push < 25:
      d.rectangle([img.width - 130, 8, img.width - 14, 40], fill=(200, 40, 40))
      d.text((img.width - 118, 12), "PUSH", fill=(255, 255, 255), font=font)
    d.text((14, 62), f"tilt {float(torch.rad2deg(s['tilt'][0])):4.1f} deg   |acc| {float(s['acc'][0].norm()):4.2f} m/s2   "
           f"eggs {kept}/{s['lost'].shape[1]}", fill=(60, 60, 60), font=small)
    frames.append(np.asarray(img))

  C.run_routine(wrapped, ctrl, None, frame_cb=cb, pushes=C.PUSH_SETS[a.pushes])
  wrapped.close()
  return frames


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--terrain", default="poultry", choices=("flat", "poultry"))
  ap.add_argument("--contact", default="built_up")
  ap.add_argument("--eggs", type=int, default=6, choices=(6, 12))
  ap.add_argument("--litter", type=float, default=0.03)
  ap.add_argument("--mode", default="rl", help="arm mode, optionally @DUET legs, e.g. fixed@squatreach50k")
  ap.add_argument("--pushes", default="none", choices=sorted(C.PUSH_SETS))
  ap.add_argument("--compare", nargs="*", default=None, help="modes to render side by side")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--closeup", action="store_true")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--out", default="logs/eggtray/videos/eggtray.mp4")
  a = ap.parse_args()
  import imageio.v2 as imageio

  modes = a.compare or [a.mode]
  clips = [record(m, a, a.seed) for m in modes]
  n = min(len(c) for c in clips)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  writer = imageio.get_writer(a.out, fps=25, codec="libx264", quality=8, macro_block_size=1)
  for i in range(n):
    writer.append_data(np.concatenate([c[i] for c in clips], axis=1))
  writer.close()
  print(f"video -> {a.out} ({n} frames, modes {modes})")


if __name__ == "__main__":
  main()
