"""The trained DUET locomotion policy as a frozen, zero-dimension action term.

The 13 lower-body joint targets (12 legs + waist_yaw) come only from
``checkpoints/poultry_v3_squat_91500/model_91500.pt``. The term has
``action_dim == 0``, so the learning agent's action vector cannot reach these
joints; each control step it reads the ``duet`` observation group -- a copy of
the PoultrySurface actor group, i.e. the deployed 71-D layout in its original
order -- runs the checkpoint's actor and applies the result exactly as the
DUET task does (per-joint scale, default-pose offset, encoder-bias
correction).

Freezing is structural, not a flag: the network lives inside the environment,
is built from the checkpoint with ``requires_grad=False``, runs under
``torch.no_grad`` and is never registered with the PPO optimizer. On build the
checkpoint's SHA-256 and its deployment contract (joint order, action scale,
offsets, obs/action sizes) are checked, so an older checkpoint cannot be
loaded by mistake.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
DUET_CHECKPOINT = "checkpoints/poultry_v3_squat_91500/model_91500.pt"
DUET_CHECKPOINT_SHA256 = "0e1e2cbf7201fa1c3140753cf8bfc2059266afbf9bd76c7d686c02e1518f979f"
DUET_EXPORT_DIR = "exported/poultry_v3_squat_91500"
# Other DUET checkpoints with the identical deploy contract (71 -> 13, same
# joint order, scale, offsets), selectable for comparisons only. Training the
# arm policy always uses DUET_CHECKPOINT.
DUET_POLICIES = {
  "91500": (DUET_CHECKPOINT, DUET_CHECKPOINT_SHA256, DUET_EXPORT_DIR),
  "squatreach50k": (
    "checkpoints/rough_squatreach_ft_50000/model_50000.pt",
    "683354ac5f4aeb265f7ca38a6eeda4d775a9132cc727001538755f92ed61e25e",
    "exported/rough_squatreach_ft_50000",
  ),
}
DUET_OBS_DIM = 71
DUET_ACTION_DIM = 13
# rsl_rl EmpiricalNormalization default; the ONNX export agrees to ~3e-6 with it.
NORMALIZER_EPS = 1e-2


def sha256_file(path: str | Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def resolve(path: str | Path) -> Path:
  p = Path(path)
  return p if p.is_absolute() else REPO_ROOT / p


class FrozenDuetPolicy(nn.Module):
  """Observation normaliser + MLP actor of an rsl_rl 5.x checkpoint, inference only."""

  def __init__(self, actor_state_dict: dict[str, torch.Tensor]):
    super().__init__()
    linear = sorted(
      {int(k.split(".")[1]) for k in actor_state_dict if k.startswith("mlp.")}
    )
    layers: list[nn.Module] = []
    for i, idx in enumerate(linear):
      w = actor_state_dict[f"mlp.{idx}.weight"]
      layer = nn.Linear(w.shape[1], w.shape[0])
      layer.weight.data.copy_(w)
      layer.bias.data.copy_(actor_state_dict[f"mlp.{idx}.bias"])
      layers.append(layer)
      if i < len(linear) - 1:
        layers.append(nn.ELU())
    self.mlp = nn.Sequential(*layers)
    self.register_buffer("obs_mean", actor_state_dict["obs_normalizer._mean"].clone())
    self.register_buffer("obs_std", actor_state_dict["obs_normalizer._std"].clone())
    self.requires_grad_(False)
    self.eval()

  @property
  def obs_dim(self) -> int:
    return self.mlp[0].in_features

  @property
  def action_dim(self) -> int:
    return self.mlp[-1].out_features

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp((obs - self.obs_mean) / (self.obs_std + NORMALIZER_EPS))

  @classmethod
  def load(
    cls, path: str | Path, device: str = "cpu", expected_sha256: str | None = None
  ) -> "FrozenDuetPolicy":
    path = resolve(path)
    if expected_sha256 is not None:
      digest = sha256_file(path)
      if digest != expected_sha256:
        raise RuntimeError(
          f"{path}: SHA-256 {digest} does not match the frozen DUET checkpoint "
          f"{expected_sha256}. Refusing to load a different locomotion policy."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = cls(ckpt["actor_state_dict"]).to(device)
    policy.source = str(path)
    policy.source_iter = int(ckpt.get("iter", -1))
    return policy


def load_contract(export_dir: str | Path = DUET_EXPORT_DIR) -> dict:
  with open(resolve(export_dir) / "deploy_contract.json") as f:
    return json.load(f)


@dataclass(kw_only=True)
class FrozenDuetActionCfg(ActionTermCfg):
  """Frozen DUET lower-body controller; consumes no policy action dimensions."""

  actuator_names: tuple[str, ...]
  scale: dict[str, float]
  checkpoint: str = DUET_CHECKPOINT
  expected_sha256: str | None = DUET_CHECKPOINT_SHA256
  contract_dir: str | None = DUET_EXPORT_DIR
  obs_group: str = "duet"
  clip_actions: float | None = 30.0
  """Same clip as PoultrySurface training (``poultry_surface_ppo_runner_cfg``);
  the last-action input is clipped separately at +-15 by the obs term."""
  extreme_action_threshold: float = 15.0
  """The deployed runaway guard trips above |a| = 15; counted, never altered."""
  extras: dict = field(default_factory=dict)

  def build(self, env):
    return FrozenDuetAction(self, env)


class FrozenDuetAction(ActionTerm):
  cfg: FrozenDuetActionCfg

  def __init__(self, cfg: FrozenDuetActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self._joint = JointPositionAction(
      JointPositionActionCfg(
        entity_name=cfg.entity_name,
        actuator_names=cfg.actuator_names,
        scale=cfg.scale,
        use_default_offset=True,
      ),
      env,
    )
    self.policy = FrozenDuetPolicy.load(cfg.checkpoint, self.device, cfg.expected_sha256)
    if (self.policy.obs_dim, self.policy.action_dim) != (DUET_OBS_DIM, DUET_ACTION_DIM):
      raise RuntimeError(
        f"checkpoint is {self.policy.obs_dim}->{self.policy.action_dim}, expected 71->13"
      )
    if cfg.contract_dir is not None:
      self._check_contract(load_contract(cfg.contract_dir))
    self._empty = torch.zeros(self.num_envs, 0, device=self.device)
    self.extreme_count = torch.zeros(self.num_envs, device=self.device)
    self._obs_checked = False

  def _check_contract(self, contract: dict) -> None:
    names = self._joint.target_names
    if names != contract["action_joint_names"]:
      raise RuntimeError(f"DUET joint order {names} != contract {contract['action_joint_names']}")
    scale = self._joint.scale
    scale = scale[0].cpu().numpy() if isinstance(scale, torch.Tensor) else np.full(len(names), scale)
    if not np.allclose(scale, contract["action_scale"], atol=1e-3):
      raise RuntimeError(f"DUET action scale {scale} != contract {contract['action_scale']}")
    offset = self._joint.offset[0].cpu().numpy()
    if not np.allclose(offset, contract["action_offset"], atol=1e-4):
      raise RuntimeError(f"DUET action offset {offset} != contract {contract['action_offset']}")

  def use_policy(self, name: str) -> None:
    """Swap in another registered DUET checkpoint (comparisons only), with the
    same hash and deploy-contract checks as at build time."""
    path, sha, export = DUET_POLICIES[name]
    policy = FrozenDuetPolicy.load(path, self.device, sha)
    self._check_contract(load_contract(export))
    self.policy = policy
    self.policy_name = name

  # -- ActionTerm interface ---------------------------------------------------

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    """The last 13-D DUET action (read by the ``duet`` group's ``actions`` term)."""
    return self._joint.raw_action

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint.target_ids

  def duet_observation(self) -> torch.Tensor:
    obs_buf = getattr(self._env, "obs_buf", None)
    if isinstance(obs_buf, dict) and self.cfg.obs_group in obs_buf:
      obs = obs_buf[self.cfg.obs_group]
    else:  # before the first reset
      obs = self._env.observation_manager.compute_group(self.cfg.obs_group)
    if not self._obs_checked:
      if obs.shape[-1] != DUET_OBS_DIM:
        raise RuntimeError(f"duet observation is {obs.shape[-1]}-D, expected 71")
      self._obs_checked = True
    return obs

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions  # zero-width slice: the agent has no say over these joints
    with torch.no_grad():
      a = self.policy(self.duet_observation())
      self.extreme_count += (a.abs() > self.cfg.extreme_action_threshold).any(-1).float()
      if self.cfg.clip_actions is not None:
        a = a.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
    self._joint.process_actions(a)

  def apply_actions(self) -> None:
    self._joint.apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._joint.reset(env_ids)
