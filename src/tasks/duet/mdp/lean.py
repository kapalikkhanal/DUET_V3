"""Forward lean ("stoop") for picking things off the floor.

Measured with the 26.5k policy (2026-11-15): squatting does not bring the hands
to the floor. Below a 0.30 m pelvis height the fingertips stop about 13 cm
above it, because the folded knees are in the way of the arms. A person bends
at the hips instead; with the pelvis at 0.45-0.50 m and the torso about 70 deg
forward the hands of this robot reach the floor in front of the feet. The
23-DOF G1 has no waist pitch joint, so the lean is the whole pelvis pitching
forward while the hips move back over the heels.

``LeanCommand`` is a new policy input: the torso pitch to hold, in radians,
0 = upright. It is only non-zero while the robot stands (after the settle
window); it ramps at a limited rate in both directions, and it returns to zero
when a velocity is commanded. While a lean is asked for, the height command is
kept inside a "stoop band". The controller must apply the same three rules.

Everything that used to mean "be upright" is measured against the commanded
lean instead of against vertical, and reduces to the old term when the lean is
zero, so a policy that never sees a lean command is trained exactly as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.common.mdp.curriculums import piecewise_linear
from src.tasks.common.mdp.rewards import idle_position_anchor
from src.tasks.duet.mdp import settle

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
LEAN = "lean"  # name of the LeanCommand term in cfg.commands


@dataclass(kw_only=True)
class LeanCommandCfg(CommandTermCfg):
  max_lean_stages: tuple[tuple[int, float], ...] = ((0, 1.25),)
  """Largest lean drawn, (policy step, rad) knots: it opens up gradually."""
  min_lean: float = 0.25
  lean_fraction: float = 0.4
  """Share of command draws that ask for a lean; the rest ask for none."""
  rate: float = 0.8
  """How fast the commanded value moves toward its target, rad/s."""
  stoop_height_range: tuple[float, float] = (0.42, 0.62)
  """Pelvis height commands allowed while a lean is asked for, m."""
  height_command_name: str = "base_height"

  def build(self, env: ManagerBasedRlEnv) -> LeanCommand:
    return LeanCommand(self, env)


class LeanCommand(CommandTerm):
  cfg: LeanCommandCfg

  def __init__(self, cfg: LeanCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target = torch.zeros(self.num_envs, device=self.device)
    self._value = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._value

  @property
  def max_lean(self) -> float:
    return piecewise_linear(int(self._env.common_step_counter), self.cfg.max_lean_stages)

  @property
  def active(self) -> torch.Tensor:
    """A lean is asked for or still being ramped out. [B] bool"""
    return (self.target > 0.0) | (self._value[:, 0] > 0.02)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._value[ids] = 0.0  # the robot is respawned upright
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    hi = max(self.max_lean, self.cfg.min_lean)
    lean = torch.empty(n, device=self.device).uniform_(self.cfg.min_lean, hi)
    ask = torch.rand(n, device=self.device) < self.cfg.lean_fraction
    self.target[env_ids] = torch.where(ask, lean, torch.zeros_like(lean))

  def _update_command(self) -> None:
    standing = self._env.command_manager.get_term(settle.HELD).standing
    goal = torch.where(standing, self.target, torch.zeros_like(self.target))
    step = self.cfg.rate * self._env.step_dt
    value = self._value[:, 0]
    self._value[:, 0] = value + (goal - value).clamp(-step, step)
    # Keep the height command inside the stoop band while a lean is asked for.
    # A lean with a deep squat is not reachable (the thighs are in the way).
    height = self._env.command_manager.get_term(self.cfg.height_command_name)
    ids = (self.target > 0.0).nonzero(as_tuple=False).flatten()
    if len(ids):
      lo, hi = self.cfg.stoop_height_range
      height._squat_target[ids] = height._squat_target[ids].clamp(lo, hi)
      still = ids[standing[ids]]
      height._height[still] = height._squat_target[still]

  def _update_metrics(self) -> None:
    pass


def _lean(env: ManagerBasedRlEnv) -> LeanCommand:
  return env.command_manager.get_term(LEAN)


def _torso_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  return quat_apply_inverse(quat, asset.data.gravity_vec_w)


def _tilt_from_commanded(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Angle (rad) between the torso and where the lean command wants it. [B]"""
  g = _torso_gravity(env, asset_cfg)
  c = _lean(env).command[:, 0]
  # Gravity seen by a torso pitched forward by c is (sin c, 0, -cos c).
  cos_angle = g[:, 0] * torch.sin(c) - g[:, 2] * torch.cos(c)
  return torch.acos(cos_angle.clamp(-1.0, 1.0))


##
# Rewards. Each equals the term it replaces when the lean command is zero.
##


def orientation_lean_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
  height_command_name: str | None = None,
  relax_below: float = 0.30,
  full_above: float = 0.55,
  relax_factor: float = 0.30,
) -> torch.Tensor:
  """``body_orientation_l2`` about the commanded lean instead of vertical."""
  g = _torso_gravity(env, asset_cfg)
  c = _lean(env).command[:, 0]
  cost = torch.square(g[:, 0] - torch.sin(c)) + torch.square(g[:, 1])
  if height_command_name is not None:
    h = env.command_manager.get_command(height_command_name)[:, 0]
    frac = ((h - relax_below) / max(full_above - relax_below, 1e-6)).clamp(0.0, 1.0)
    cost = cost * (relax_factor + (1.0 - relax_factor) * frac)
  return cost


def stand_upright_lean(
  env: ManagerBasedRlEnv,
  deadband: float = 0.026,
  height_command_name: str = "base_height",
  fade_below: float = 0.55,
  full_above: float = 0.70,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """``settle.stand_upright`` about the commanded lean; this is what tracks it.

  Linear in the angle between the torso and the commanded lean. Without a lean
  it fades out for squats exactly as before; while a lean is asked for it is
  at full strength at every height.
  """
  tilt = _tilt_from_commanded(env, asset_cfg)
  h = env.command_manager.get_command(height_command_name)[:, 0]
  gate = ((h - fade_below) / max(full_above - fade_below, 1e-6)).clamp(0.0, 1.0)
  gate = torch.maximum(gate, _lean(env).active.float())
  moving = env.command_manager.get_term(settle.HELD).moving
  return (tilt - deadband).clamp_min(0.0) * gate * (~moving)


class idle_position_anchor_lean(idle_position_anchor):
  """The idle position anchor, released while the robot leans.

  The anchor charges pelvis travel from where the robot stopped. Leaning moves
  the pelvis 10-15 cm back over the heels, which is the balance strategy, not
  drift. While a lean is active the anchor is re-latched every step (so it
  costs nothing), and it latches again where the robot stands once upright.
  """

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    ids = _lean(env).active.nonzero(as_tuple=False).flatten()
    if len(ids):
      self.reset(ids)
    return super().__call__(env, **kwargs)


##
# Termination.
##


def fell_over_lean(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  lean_margin: float = 0.5,
  min_height: float = 0.28,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Fallen: the old 60 deg rule, widened forward by the commanded lean.

  Without a lean command this is ``bad_orientation(limit_angle)``. With one,
  the pelvis may pitch forward up to the command plus ``lean_margin``; it has
  fallen if it pitches further, pitches backward, rolls onto its side, or the
  pelvis drops below ``min_height`` above the lower foot (collapsed or lying
  on its front, which the angle alone would not catch at a 70 deg command).
  ``asset_cfg.site_ids`` are the foot sites.
  """
  asset: Entity = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b
  upright_fall = torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) > limit_angle
  value = _lean(env).command[:, 0]
  pitch = torch.atan2(g[:, 0], -g[:, 2])
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  height = asset.data.root_link_pos_w[:, 2] - foot_z.min(dim=1).values
  lean_fall = (
    (g[:, 1].abs() > math.sin(math.radians(50.0)))
    | (pitch < -0.5)
    | (pitch > value + lean_margin)
    | (height < min_height)
  )
  return torch.where(value > 0.05, lean_fall, upright_fall)


##
# Metrics. Zero when no lean is asked for: divide by ``lean_fraction``.
##


def _at_lean(env: ManagerBasedRlEnv) -> torch.Tensor:
  """A lean is asked for and the command has finished ramping to it. [B] bool"""
  term = _lean(env)
  return (term.target > 0.0) & ((term.command[:, 0] - term.target).abs() < 1e-3)


def lean_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _at_lean(env).float()


def lean_command_rad(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _lean(env).command[:, 0] * _at_lean(env)


def lean_error_deg(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Angle between the torso and the commanded lean while holding one, deg."""
  return torch.rad2deg(_tilt_from_commanded(env, asset_cfg)) * _at_lean(env)


def lean_max_rad(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
  """Curriculum log: the largest lean currently drawn."""
  del env_ids
  return torch.tensor([_lean(env).max_lean])
