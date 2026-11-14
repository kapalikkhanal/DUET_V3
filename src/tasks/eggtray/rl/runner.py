"""On-policy runner for the egg-tray arm stabiliser.

Adds to ``MjlabOnPolicyRunner``:

1. A start-up audit that the frozen DUET network (inside the environment) has
   no trainable parameters and shares none with the PPO optimizer, and that the
   agent's action space is exactly the 10 arm residuals.
2. An ONNX export of the ARM policy next to every checkpoint, with metadata
   (observation layout, joint order, residual bounds, carry reference, and the
   hash of the DUET checkpoint it was trained with). It is a separate file from
   DUET's policy; nothing in ``exported/`` or ``deploy/`` is touched.
"""

from __future__ import annotations

import os

import torch

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner


def audit_frozen_duet(runner) -> dict:
  env = runner.env.unwrapped
  duet = env.action_manager.get_term("duet_lower_body")
  arm = env.action_manager.get_term("arm_residual")
  frozen = list(duet.policy.parameters())
  if any(p.requires_grad for p in frozen):
    raise RuntimeError("frozen DUET parameters require grad")
  frozen_ids = {id(p) for p in frozen}
  opt_ids = {id(p) for g in runner.alg.optimizer.param_groups for p in g["params"]}
  if frozen_ids & opt_ids:
    raise RuntimeError("frozen DUET parameters are registered with the PPO optimizer")
  dims = dict(zip(env.action_manager.active_terms, env.action_manager.action_term_dim))
  if env.action_manager.total_action_dim != 10 or dims.get("arm_residual") != 10:
    raise RuntimeError(f"agent action space must be the 10 arm residuals, got {dims}")
  robot = env.scene["robot"]
  leg_ids = set(duet.joint_ids.tolist())
  if leg_ids & set(arm.joint_ids.tolist()):
    raise RuntimeError("arm residual overlaps the DUET joints")
  return {
    "duet_checkpoint": duet.policy.source,
    "duet_iter": duet.policy.source_iter,
    "duet_params_frozen": sum(p.numel() for p in frozen),
    "arm_policy_params_trained": sum(p.numel() for p in opt_ids_params(runner)),
    "action_terms": dims,
    "duet_joints": [robot.joint_names[i] for i in sorted(leg_ids)],
    "arm_joints": arm.joint_names,
  }


def opt_ids_params(runner):
  for g in runner.alg.optimizer.param_groups:
    yield from g["params"]


def arm_policy_metadata(env) -> dict:
  from src.tasks.eggtray.frozen_duet import DUET_CHECKPOINT, DUET_CHECKPOINT_SHA256

  arm = env.action_manager.get_term("arm_residual")
  om = env.observation_manager
  names = om.active_terms["actor"]
  dims = [int(d[0]) for d in om.group_obs_term_dim["actor"]]
  return {
    "task": "EggTray-ArmResidual-G1-23Dof",
    "observation_names": names,
    "observation_dims": dims,
    "obs_dim": sum(dims),
    "action_joint_names": arm.joint_names,
    "action_scale": arm.cfg.scale,
    "max_residual_rad": arm.cfg.max_residual,
    "max_target_rate_rad_s": arm.cfg.max_rate,
    "carry_reference": arm.base_reference[0].tolist(),
    "carry_nominal": arm.q_nominal[0].tolist(),
    "step_dt": env.step_dt,
    "duet_checkpoint": DUET_CHECKPOINT,
    "duet_checkpoint_sha256": DUET_CHECKPOINT_SHA256,
  }


class EggTrayOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg: dict, log_dir=None, device: str = "cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    report = audit_frozen_duet(self)
    print(f"[INFO] frozen-DUET audit passed: {report}")

  def save(self, path: str, infos=None):
    super().save(path, infos)
    folder = os.path.dirname(path)
    self.export_policy_to_onnx(folder, "arm_policy.onnx")
    meta = {k: (",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
            for k, v in arm_policy_metadata(self.env.unwrapped).items()}
    attach_metadata_to_onnx(os.path.join(folder, "arm_policy.onnx"), meta)


def load_arm_policy(path: str, device: str = "cpu") -> torch.nn.Module:
  """Inference module for a saved arm policy (normaliser + MLP mean), no env needed."""
  from src.tasks.eggtray.frozen_duet import FrozenDuetPolicy

  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  return FrozenDuetPolicy(ckpt["actor_state_dict"]).to(device)
