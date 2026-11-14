"""Residual arm action: ``arm_target = carry_reference + RL_residual``.

The only term with a non-zero action dimension in the egg-tray task, and it
only addresses the ten arm joints (matched by name; the constructor rejects
anything else). Safety layers, applied in order every control step:

1. residual = clamp(scale * a, -max_residual, +max_residual), per joint;
2. the change of the commanded target is rate-limited to ``max_rate`` rad/s;
3. the target is clamped to the joint's soft limits (90% of range);
4. torque is bounded by the actuator's effort limit (25 N m, mjlab actuator).

The carry reference is per environment: the nominal pose plus gravity
feed-forward plus, optionally, a small tray-consistent offset (a random tray
twist mapped through ``CarryGeometry.tray_twist_to_joints``), so the policy
learns around a family of carry poses rather than one -- a later GR00T
handover will not land on exactly the nominal pose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from src.tasks.eggtray.carry import CarryPoseCfg, compute_carry_geometry
from src.tasks.eggtray.tray_model import TrayCfg

_ARM_NAME = re.compile(r"^(left|right)_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint$")


@dataclass(kw_only=True)
class ArmResidualActionCfg(ActionTermCfg):
  carry: CarryPoseCfg
  tray: TrayCfg
  scale: float = 0.05
  """rad per unit of policy output."""
  max_residual: float = 0.15
  """rad, per joint. Conservative: about 8.6 deg."""
  max_rate: float = 2.0
  """rad/s cap on the commanded target's rate of change."""
  carry_twist_noise: tuple[float, ...] = (0.0,) * 6
  """Half-ranges (m, m, m, rad, rad, rad) of the random tray offset per episode
  (torso frame, about the tray origin). Zero = always the nominal pose."""
  enabled: bool = True
  """False pins the targets at the carry reference (fixed-arm baseline)."""
  extras: dict = field(default_factory=dict)

  def build(self, env):
    return ArmResidualAction(self, env)


class ArmResidualAction(ActionTerm):
  cfg: ArmResidualActionCfg

  def __init__(self, cfg: ArmResidualActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    geo = compute_carry_geometry(cfg.carry, cfg.tray)
    ids, names = self._entity.find_joints(list(geo.arm_joint_names), preserve_order=True)
    if len(ids) != 10 or not all(_ARM_NAME.match(n) for n in names):
      raise ValueError(f"arm residual must address exactly the 10 arm joints, got {names}")
    self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._joint_names = names
    B = self.num_envs
    nominal = torch.tensor(geo.q_nominal, device=self.device, dtype=torch.float32)
    ff = cfg.carry.feedforward_dict()
    feedforward = torch.tensor([ff.get(n, 0.0) for n in names], device=self.device)
    self.q_nominal = nominal.expand(B, -1).clone()
    self.base_reference = (nominal + feedforward).expand(B, -1).clone()
    self.reference = self.base_reference.clone()
    self.twist_to_joints = torch.tensor(geo.tray_twist_to_joints, device=self.device, dtype=torch.float32)
    limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids]
    self._lo, self._hi = limits[..., 0], limits[..., 1]
    self._raw = torch.zeros(B, 10, device=self.device)
    self.residual = torch.zeros(B, 10, device=self.device)
    self.target = self.reference.clone()
    self._max_step = cfg.max_rate * env.step_dt

  @property
  def action_dim(self) -> int:
    return 10

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  @property
  def joint_names(self) -> list[str]:
    return self._joint_names

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    if self.cfg.enabled:
      self.residual = (actions * self.cfg.scale).clamp(-self.cfg.max_residual, self.cfg.max_residual)
    else:
      self.residual = torch.zeros_like(self._raw)
    desired = self.reference + self.residual
    step = (desired - self.target).clamp(-self._max_step, self._max_step)
    self.target = torch.minimum(torch.maximum(self.target + step, self._lo), self._hi)

  def apply_actions(self) -> None:
    bias = self._entity.data.encoder_bias[:, self._joint_ids]
    self._entity.set_joint_position_target(self.target - bias, joint_ids=self._joint_ids)

  def reset(self, env_ids=None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    noise = torch.tensor(self.cfg.carry_twist_noise, device=self.device)
    if bool((noise > 0).any()):
      n = self.reference[ids].shape[0]
      twist = (torch.rand(n, 6, device=self.device) * 2 - 1) * noise
      self.reference[ids] = self.base_reference[ids] + twist @ self.twist_to_joints.T
    else:
      self.reference[ids] = self.base_reference[ids]
    self._raw[ids] = 0.0
    self.residual[ids] = 0.0
    # The arms are reset to the nominal pose; with the feed-forward the first
    # target is the static-equilibrium target, so there is no first-step jump.
    self.target[ids] = self.base_reference[ids]
