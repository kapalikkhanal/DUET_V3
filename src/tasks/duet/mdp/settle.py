"""Stop -> settle -> stand, and the from-scratch schedules of PoultrySurface.

Measured on the real robot with the 91.5k policy (farm and lab, 2026-11-10): it
stopped in whatever staggered stance it had and took a median 1.9 s (lab 0.6 s)
to go quiet. The cause is in the rewards, not the weights of the network: every
standing term (foot motion, base travel, position anchor, standing posture)
switched on at the very step the stick was released, while the robot was still
moving, so braking on the spot and freezing was the cheapest answer.

``HeldTwistCommand`` fixes that with one idea: the terms that mean "you are
standing now" read a copy of the velocity command that goes to zero
``settle_time`` seconds late. For that one second the robot is neither paid for
walking nor charged for stepping, and the gait clock stays visible, so it can
take the step or two that put its feet side by side. ``stand_feet_square`` is
what makes that worth doing. Nothing else in the reward set changes.

The robot applies the same rule in ``gait_phase`` (deploy ``observations.h``,
``settle`` in deploy.yaml, which must equal ``settle_time`` here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.common.mdp.curriculums import piecewise_linear

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
HELD = "twist_held"  # name of the HeldTwistCommand term in cfg.commands


##
# The delayed copy of the velocity command.
##


@dataclass(kw_only=True)
class HeldTwistCommandCfg(CommandTermCfg):
  """The twist command, except that it reaches zero ``settle_time`` late.

    moving    twist norm >= ``idle_threshold``: equal to the twist
    settling  twist is zero, for less than ``settle_time``: the last moving twist
    standing  twist has been zero for ``settle_time`` or longer: zero

  A fresh episode starts STANDING (the robot is spawned at rest).
  """

  twist_command_name: str = "twist"
  settle_time: float = 1.0
  idle_threshold: float = 0.1

  def build(self, env: ManagerBasedRlEnv) -> HeldTwistCommand:
    return HeldTwistCommand(self, env)


class HeldTwistCommand(CommandTerm):
  cfg: HeldTwistCommandCfg

  def __init__(self, cfg: HeldTwistCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.idle_time = torch.full((n,), cfg.settle_time, device=dev)
    self._idle = torch.ones(n, dtype=torch.bool, device=dev)
    self._held = torch.zeros(n, 3, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self._held

  @property
  def moving(self) -> torch.Tensor:
    return ~self._idle

  @property
  def settling(self) -> torch.Tensor:
    return self._idle & (self.idle_time < self.cfg.settle_time)

  @property
  def standing(self) -> torch.Tensor:
    return self._idle & (self.idle_time >= self.cfg.settle_time)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.idle_time[ids] = self.cfg.settle_time
    self._idle[ids] = True
    self._held[ids] = 0.0
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Nothing is sampled: the state follows the twist command.

  def _update_command(self) -> None:
    twist = self._env.command_manager.get_command(self.cfg.twist_command_name)[:, :3]
    idle = torch.norm(twist, dim=1) < self.cfg.idle_threshold
    # In-place: rollouts run under torch.inference_mode, and a buffer rebound
    # there could not be written by a later reset outside it.
    self.idle_time.copy_(
      torch.where(idle, self.idle_time + self._env.step_dt, torch.zeros_like(self.idle_time))
    )
    self._idle.copy_(idle)
    keep = self.settling.unsqueeze(1)  # hold the last moving twist
    self._held.copy_(
      torch.where(idle.unsqueeze(1), torch.where(keep, self._held, torch.zeros_like(twist)), twist)
    )

  def _update_metrics(self) -> None:
    pass


def _held(env: ManagerBasedRlEnv) -> HeldTwistCommand:
  return env.command_manager.get_term(HELD)


##
# Reward.
##


def _feet_stagger(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Fore-aft distance between the two feet, in the base frame (m). [B]"""
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  offset = quat_apply_inverse(asset.data.root_link_quat_w, feet[:, 0] - feet[:, 1])
  return offset[:, 0].abs()


def stand_feet_square(
  env: ManagerBasedRlEnv, deadband: float = 0.02, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Fore-aft stagger between the feet (m) once the robot is STANDING.

  The only term that pays for the squaring step: it can be reduced only by
  moving a foot, which is free during the settle window and charged afterwards.
  """
  return (_feet_stagger(env, asset_cfg) - deadband).clamp_min(0.0) * _held(env).standing


def gait_swing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period: float,
  stance_fraction: float = 0.56,
  offset: tuple[float, float] = (0.0, 0.5),
) -> torch.Tensor:
  """Foot in the air during its scheduled swing; zero for a robot that stands.

  +1 (averaged over the stride) for lifting each foot in its swing window, -1
  per foot that is airborne while it is scheduled to be planted. Zero unless a
  velocity is commanded.

  Why not the shared ``feet_gait``: that term scores agreement between contact
  and the schedule in BOTH phases, so a robot that never lifts a foot still
  collects the stance share, 56% of it. At 4,000 iterations of the full run the
  robot answered a pure sideways command of 0.25 or 0.40 m/s by standing still
  (0.00 m/s, no steps), although it sidestepped at half speed when a forward
  command had it stepping already. Stepping on the spot paid only the missing
  44% and cost the foot-lift penalties, so there was no first move toward a
  sidestep. Here standing under a walk command earns nothing.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_contact_time <= 0.0
  phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
  phase = (phase + torch.as_tensor(offset, device=env.device).view(1, -1)) % 1.0
  swing = phase >= stance_fraction
  good = (swing & air).float().mean(dim=1) / (1.0 - stance_fraction)
  bad = (~swing & air).float().mean(dim=1)
  return (good - bad) * _held(env).moving


def stand_upright(
  env: ManagerBasedRlEnv,
  deadband: float = 0.026,
  height_command_name: str = "base_height",
  fade_below: float = 0.55,
  full_above: float = 0.70,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Torso tilt angle (rad, beyond ``deadband``) while no velocity is commanded.

  The 2,000-iteration trial stood leaning 9.7 deg forward with its hips pushed
  back, and the 91.5k policy leaned 5.9 deg on the robot. The quadratic
  orientation term is almost flat near zero (that 9.7 deg costs 0.07 per
  second); this one is linear in the angle, so the pull does not vanish near
  upright. The 1.5 deg deadband leaves room to balance a payload held forward.
  Fades out below a 0.70 m height command: a squat needs the torso to lean.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  g = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  tilt = torch.asin(g[:, :2].norm(dim=1).clamp(max=1.0))
  h = env.command_manager.get_command(height_command_name)[:, 0]
  gate = ((h - fade_below) / max(full_above - fade_below, 1e-6)).clamp(0.0, 1.0)
  return (tilt - deadband).clamp_min(0.0) * gate * (~_held(env).moving)


##
# Events and curriculum.
##


def random_stops(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  prob: float = 0.15,
  command_name: str = "twist",
) -> None:
  """Turn a random share of walking robots into standing ones mid-command.

  Stopping is what the hardware trial failed on, and command resampling alone
  produces a walk-to-stop transition only about once per episode.
  """
  term = env.command_manager.get_term(command_name)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  moving = term.command[env_ids].norm(dim=1) > 0.1
  stop_ids = env_ids[moving & (torch.rand(len(env_ids), device=env.device) < prob)]
  term.is_standing_env[stop_ids] = True
  term.command[stop_ids] = 0.0


def reward_weight_schedule(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, reward_name: str, stages
) -> torch.Tensor:
  """Ramp one reward weight along (policy_step, weight) knots."""
  del env_ids
  weight = piecewise_linear(env.common_step_counter, stages)
  env.reward_manager.get_term_cfg(reward_name).weight = weight
  return torch.tensor([weight])


def litter_depth_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages,
  action_term_name: str = "poultry_surface",
) -> torch.Tensor:
  """Deepest loose litter drawn per episode, (policy_step, metres) knots."""
  del env_ids
  term = env.action_manager.get_term(action_term_name)
  deepest = piecewise_linear(env.common_step_counter, stages)
  term.cfg.litter_depth_range = (0.0, deepest)
  return torch.tensor([deepest])


##
# Metrics. Per-step values, averaged over the episode by the manager. The
# ``stand_*`` ones are zero when not standing: divide by ``standing_fraction``
# (or ``standing_tall_fraction``) to get the mean while standing.
##


def standing_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _held(env).standing.float()


def _standing_tall(env: ManagerBasedRlEnv, min_height: float = 0.75) -> torch.Tensor:
  """STANDING with a full-height command, i.e. not a commanded squat."""
  h = env.command_manager.get_command("base_height")[:, 0]
  return _held(env).standing & (h >= min_height)


def standing_tall_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _standing_tall(env).float()


def stand_pitch_deg(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Signed torso pitch (deg, + = leaning forward) while standing tall."""
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  g = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])) * _standing_tall(env)


def stand_knee_rad(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Mean knee angle (rad) while standing tall; the default pose is 0.30."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=1) * _standing_tall(env)


def stand_stagger_m(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Fore-aft foot stagger (m) while STANDING."""
  return _feet_stagger(env, asset_cfg) * _held(env).standing


def stand_foot_air(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1 on a step where a foot is off the ground while STANDING (a balance step)."""
  sensor: ContactSensor = env.scene[sensor_name]
  any_air = (sensor.data.current_contact_time <= 0.0).any(dim=1)
  return (any_air & _held(env).standing).float()


def steps_per_second(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Foot touchdowns per second while a velocity is commanded (0 = not walking).

  A 0.85 s stride is 2.35 steps per second. Divide by the moving share.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  touchdown = sensor.compute_first_contact(env.step_dt).float().sum(dim=1)
  return touchdown / env.step_dt * _held(env).moving


def moving_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _held(env).moving.float()


def action_abs_max(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Largest |raw policy output| this step (the deploy guard trips above 15)."""
  return env.action_manager.get_term("joint_pos").raw_action.abs().amax(dim=1)
