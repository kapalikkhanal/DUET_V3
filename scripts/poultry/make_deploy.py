"""Turn a PoultrySurface checkpoint into a deploy folder for the C++ controller.

  PYTHONPATH=. python scripts/poultry/make_deploy.py RUN_DIR CHECKPOINT NAME [GAIT_PERIOD]

  RUN_DIR     e.g. logs/rsl_rl/DUET_G1_23dof_poultry/2026-11-14_11-35-49_trial2k
  CHECKPOINT  e.g. model_1999.pt
  NAME        folder created under deploy/robots/g1_23dof/config/policy/poultry/
  GAIT_PERIOD stride period the checkpoint was trained with, s (default: the
              task's current value; checkpoints up to 19,500 used 0.85)

Writes exported/policy.onnx (with the training contract in its metadata),
exported/checkpoint.json and params/deploy.yaml. Gains, action scale, default
pose and observation layout are read from the training environment, and the
command ranges are the ones the curriculum had reached at that checkpoint, so
the robot is never commanded outside what the policy has seen. The ONNX file
is compared against the checkpoint on random inputs before anything is kept.

Then point ``policy_dir`` in deploy/robots/g1_23dof/config/config.yaml at the
new folder and run scripts/check_deploy_consistency.py (printed at the end).
"""

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import onnxruntime as ort
import torch
import yaml

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.duet.config.g1_23dof import poultry_surface as P
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.rl.export_metadata import build_deploy_metadata

TASK = "Unitree-G1-23Dof-Duet-PoultrySurface"
ROOT = Path(__file__).resolve().parents[2]
# Policy joint order -> Unitree SDK motor index.
JOINT_IDS_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
# Joint-target clamp (rad) applied by the controller, as in every earlier deploy.yaml.
TARGET_CLIP = [
  [-3.22, 3.11], [-0.85, 3.29], [-2.98, 2.98], [-0.74, 3.34], [-1.99, 1.97], [-0.98, 0.96],
  [-3.33, 3.11], [-3.29, 0.86], [-2.98, 2.98], [-0.74, 3.33], [-2.02, 1.97], [-0.95, 0.99],
  [-2.86, 2.86],
]
# Training observation term -> (controller term, extra fields).
OBS = {
  "base_ang_vel": ("base_ang_vel", {}),
  "projected_gravity": ("projected_gravity", {}),
  "command": ("velocity_commands", {"params": {"command_name": "base_velocity"}}),
  "phase": ("gait_phase", {"params": {"period": float(sys.argv[4]) if len(sys.argv) > 4 else P.GAIT_PERIOD,
                                      "settle": P.SETTLE_TIME}}),
  "joint_pos": ("joint_pos_rel", {}),
  "joint_vel": ("joint_vel_rel", {}),
  "actions": ("last_action", {"clip": [-15.0, 15.0]}),
  "height_command": ("base_height_command", {}),
  "lean_command": ("lean_command", {}),
}


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def inward(lo: float, hi: float, digits: int = 2) -> list[float]:
  """Round a range toward zero, so the deployed limit is never wider than trained."""
  f = 10**digits
  return [float(np.ceil(lo * f) / f), float(np.floor(hi * f) / f)]


def main() -> int:
  run_dir, ckpt_name, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
  ckpt = run_dir / ckpt_name
  iteration = int(re.search(r"(\d+)", ckpt_name).group(1))
  out = ROOT / "deploy/robots/g1_23dof/config/policy/poultry" / name
  (out / "exported").mkdir(parents=True, exist_ok=True)
  (out / "params").mkdir(parents=True, exist_ok=True)

  # A checkpoint of the lean task has one more input per frame (see add_lean_input.py).
  global TASK
  n_in = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]["mlp.0.weight"].shape[1]
  if n_in == 72 * P.HISTORY:
    TASK += "-Lean"

  # The TRAINING config, so the curriculum can be advanced to this checkpoint.
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 4
  agent = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"), clip_actions=agent.clip_actions)
  u = env.unwrapped
  u.common_step_counter = iteration * ITER
  u.curriculum_manager.compute(torch.arange(4, device=u.device))

  runner = load_runner_cls(TASK)(env, asdict(agent), device="cuda:0")
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  onnx_path = out / "exported" / "policy.onnx"
  runner.export_policy_to_onnx(str(onnx_path.parent), onnx_path.name)
  meta = build_deploy_metadata(u, f"{run_dir.name}/{ckpt_name}")
  attach_metadata_to_onnx(str(onnx_path), meta)

  # ONNX against the checkpoint.
  policy = runner.get_inference_policy(device="cuda:0")
  obs = env.get_observations()
  x = torch.randn(100, obs["actor"].shape[1], device="cuda:0") * 2.0
  td = obs.clone()[:1].expand(100).clone()
  td["actor"] = x
  with torch.inference_mode():
    ref = policy(td).cpu().numpy()
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  feed, xs = sess.get_inputs()[0].name, x.cpu().numpy()
  got = np.concatenate([sess.run(None, {feed: xs[i:i + 1]})[0] for i in range(len(xs))])  # batch of 1
  err = float(np.abs(got - ref).max())
  if err > 1e-3:
    print(f"ONNX does not reproduce the checkpoint (max error {err:.2e}); nothing written.")
    return 1

  twist = u.command_manager.get_term("twist").cfg.ranges
  height = u.command_manager.get_term("base_height")
  floor = float(np.ceil(height.current_floor * 100) / 100)
  top = float(height.cfg.height_range[1])
  walk_min = float(np.ceil(height.cfg.walk_min_height * 100) / 100)

  n_act = len(meta["action_joint_names"])
  observations = {}
  for term, dim in zip(meta["observation_names"], meta["observation_dims"], strict=True):
    key, extra = OBS[term]
    observations[key] = {
      "params": extra.get("params", {}),
      "clip": extra.get("clip"),
      "scale": [1.0] * (dim // P.HISTORY),
      "history_length": P.HISTORY,
    }
  deploy = {
    "joint_ids_map": JOINT_IDS_MAP,
    "step_dt": meta["step_dt"],
    "stiffness": meta["joint_stiffness"],
    "damping": meta["joint_damping"],
    "default_joint_pos": meta["default_joint_pos"],
    "safety": {"max_abs_action": 15.0, "max_abs_action_hold_s": 0.06},
    "commands": {
      "base_velocity": {
        "ranges": {
          "lin_vel_x": inward(*twist.lin_vel_x),
          "lin_vel_y": inward(*twist.lin_vel_y),
          "ang_vel_z": inward(*twist.ang_vel_z),
          "heading": None,
        }
      },
      "base_height": {
        "range": [floor, top],
        "standstill": top,
        "walk_min": walk_min,
        "settle": P.SETTLE_TIME,
      },
    },
    "actions": {
      "JointPositionAction": {
        "clip": TARGET_CLIP,
        "joint_names": [".*"],
        "joint_ids": list(range(n_act)),
        "scale": meta["action_scale"],
        "offset": meta["action_offset"],
      }
    },
    "observations": observations,
  }
  if "lean" in u.command_manager.active_terms:
    lean_term = u.command_manager.get_term("lean")
    lean_max = float(np.floor(lean_term.max_lean * 100) / 100)
    stoop_lo = float(np.ceil(lean_term.stoop_floor * 100) / 100)
    deploy["commands"]["lean"] = {
      "max": lean_max,  # largest lean the curriculum had reached, rad
      "pick": lean_max,  # lean asked for by button X
      "rate": lean_term.cfg.rate,
      "settle": P.SETTLE_TIME,
      "height_range": [stoop_lo, float(lean_term.cfg.stoop_height_range[1])],
      "height": max(stoop_lo, 0.36),  # pelvis height held while leaning (D-pad Up/Down adjusts)
    }
  header = (
    f"# PoultrySurface {run_dir.name}/{ckpt_name} (iteration {iteration}).\n"
    f"# {P.HISTORY} frames per observation term: {meta['obs_dim']} inputs, {n_act} actions. "
    f"Gait period {OBS['phase'][1]['params']['period']} s.\n"
    f"# Command ranges are what the curriculum had reached at this checkpoint.\n"
    f"# Written by scripts/poultry/make_deploy.py; contract hash {meta['deploy_config_hash']}.\n"
  )
  yaml_path = out / "params" / "deploy.yaml"
  yaml_path.write_text(header + yaml.safe_dump(deploy, sort_keys=False, default_flow_style=None, width=100))
  (out / "exported" / "checkpoint.json").write_text(json.dumps({
    "checkpoint": f"{run_dir.name}/{ckpt_name}",
    "iteration": iteration,
    "checkpoint_sha256": sha256(ckpt),
    "policy_sha256": sha256(onnx_path),
    "onnx_vs_checkpoint_inputs": 100,
    "onnx_vs_checkpoint_max_abs_error": err,
  }, indent=2) + "\n")

  rel = out.relative_to(ROOT)
  print(f"\nwrote {rel}  (ONNX vs checkpoint max error {err:.1e})")
  print(f"  velocity x {deploy['commands']['base_velocity']['ranges']['lin_vel_x']}, "
        f"y {deploy['commands']['base_velocity']['ranges']['lin_vel_y']}, "
        f"yaw {deploy['commands']['base_velocity']['ranges']['ang_vel_z']}; "
        f"height {floor}-{top} m, walking above {walk_min} m")
  print(f"check:  python scripts/check_deploy_consistency.py --onnx {rel}/exported/policy.onnx "
        f"--deploy-yaml {rel}/params/deploy.yaml --expect-obs-dim {meta['obs_dim']}")
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
