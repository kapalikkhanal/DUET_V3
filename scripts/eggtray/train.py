"""Train the egg-tray arm stabiliser with the repository's training script.

A thin wrapper around ``scripts/train.py`` (same CLI). Before handing over it
checks that the frozen DUET checkpoint is the expected file and that the
poultry-floor assets exist for the Poultry tasks. Checkpoints go to
``logs/rsl_rl/EggTray_G1_23dof/<run>/`` with an ``arm_policy.onnx`` beside each
one; nothing under ``checkpoints/``, ``exported/`` or ``deploy/`` is written.

  PYTHONPATH=. python scripts/eggtray/train.py Unitree-G1-23Dof-EggTray-Poultry \
      --env.scene.num-envs 1024 --agent.max-iterations 500 --agent.run-name main --agent.seed 1
"""

from __future__ import annotations

import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)


def main() -> None:
  if len(sys.argv) < 2 or not sys.argv[1].startswith("Unitree-G1-23Dof-EggTray"):
    sys.exit(__doc__)
  from src.tasks.eggtray.frozen_duet import DUET_CHECKPOINT, DUET_CHECKPOINT_SHA256, resolve, sha256_file

  digest = sha256_file(resolve(DUET_CHECKPOINT))
  if digest != DUET_CHECKPOINT_SHA256:
    sys.exit(f"{DUET_CHECKPOINT} has SHA-256 {digest}, expected {DUET_CHECKPOINT_SHA256}")
  print(f"[INFO] frozen DUET checkpoint {DUET_CHECKPOINT} (sha256 {digest[:12]}...) verified")
  if "Poultry" in sys.argv[1]:
    from src.tasks.eggtray.config.g1_23dof.env_cfgs import floor_assets

    if not os.path.exists(os.path.join(floor_assets(), "floor_meta.json")):
      sys.exit(
        f"Poultry floor assets not found at {floor_assets()}. Build them with\n"
        "  python scripts/poultry_floor/make_poultry_floor.py --out logs/eggtray/poultry_floor --no-verify"
      )
  os.chdir(REPO)
  sys.argv[0] = os.path.join(REPO, "scripts", "train.py")
  runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
  main()
