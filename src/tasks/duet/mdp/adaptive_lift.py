"""Continuous support-relative swing commands, LiftFirst shaping and flat landings."""
import math
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from src.tasks.common.mdp.curriculums import piecewise_linear
from src.tasks.common.mdp.rewards import idle_base_motion, idle_feet_still

from . import settle
from .lean import idle_position_anchor_lean

from .lift_terrain import LAYOUT, LOOSE, MATERIALS, column_kinds, material_at, split_heights, split_offset
from .poultry_surface import PoultrySurfaceAction, PoultrySurfaceActionCfg, litter_wrench, point_kinematics

# A landing counts as heel-first when the sole was this far toe-up just before contact.
HEEL_FIRST_RAD = math.radians(3.)


@dataclass(kw_only=True)
class LiftCommandCfg(CommandTermCfg):
  height_range: tuple[float, float] = (.04, .11)
  rate_mps: float = .035
  independent_fraction: float = .25

  def build(self, env):
    return LiftCommand(self, env)


class LiftCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.goal = torch.full((self.num_envs, 2), sum(cfg.height_range) / 2, device=self.device)
    self.value = self.goal.clone()

  @property
  def command(self):
    return self.value

  def _resample_command(self, env_ids):
    pair = torch.empty(len(env_ids), 2, device=self.device).uniform_(*self.cfg.height_range)
    independent = torch.rand(len(env_ids), device=self.device) < self.cfg.independent_fraction
    pair[:, 1] = torch.where(independent, pair[:, 1], pair[:, 0])
    self.goal[env_ids] = pair

  def reset(self, env_ids):
    extras = super().reset(env_ids)
    self.value[env_ids] = self.goal[env_ids]
    return extras

  def _update_command(self):
    limit = self.cfg.rate_mps * self._env.step_dt
    self.value.add_((self.goal - self.value).clamp(-limit, limit))

  def _update_metrics(self):
    pass


def clearance_commands(env):
  command = env.command_manager.get_term("clearance")
  low, high = command.cfg.height_range
  return 2 * (command.command - low) / (high - low) - 1


@dataclass(kw_only=True)
class PoultryTwistCfg(UniformVelocityCommandCfg):
  """Velocity command with dedicated in-place turns and pure sidesteps."""

  turn_fraction: float = .2
  side_fraction: float = .2

  def build(self, env):
    return PoultryTwist(self, env)


class PoultryTwist(UniformVelocityCommand):
  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    # Standing draws are kept. Of the moving draws, a fifth each become a pure
    # turn or a pure sidestep at 35-100% of the current curriculum range, in
    # both directions. The rest stay mixed. (As in the 47k->50k fine-tune.)
    draw = torch.rand(len(env_ids), device=self.device)
    moving = ~self.is_standing_env[env_ids]
    turn_end = self.cfg.turn_fraction
    side_end = turn_end + self.cfg.side_fraction
    turns = env_ids[moving & (draw < turn_end)]
    sides = env_ids[moving & (draw >= turn_end) & (draw < side_end)]
    for ids, axis, limits in ((turns, 2, self.cfg.ranges.ang_vel_z),
                              (sides, 1, self.cfg.ranges.lin_vel_y)):
      sample = torch.rand(len(ids), device=self.device) * 2 - 1
      speed = torch.where(sample < 0, limits[0], limits[1]) * (.35 + .65 * sample.abs())
      self.is_heading_env[ids] = False
      self.vel_command_b[ids] = 0
      self.vel_command_b[ids, axis] = speed


def track_lateral_velocity(env, std, command_name="twist"):
  """HOMIE's separate y-velocity term, exp(-(vy - vy_cmd)^2 / std^2)."""
  vy_cmd = env.command_manager.get_command(command_name)[:, 1]
  vy = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
  return torch.exp(-(vy - vy_cmd).square() / std**2)


@dataclass(kw_only=True)
class LiftSurfaceCfg(PoultrySurfaceActionCfg):
  # Loose layer per episode: uniform from this depth to the curriculum's deepest
  # (litter_depth_range[1], moved by settle.litter_depth_schedule).
  loose_min_depth: float = .02
  compacted_scale_range: tuple[float, float] = (.8, 1.2)

  def build(self, env):
    return LiftSurface(self, env)


class LiftSurface(PoultrySurfaceAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.layout = torch.as_tensor(LAYOUT, device=self.device)
    self.materials = torch.tensor(MATERIALS, device=self.device)
    self.col_kind = torch.tensor(column_kinds(self.terrain_cfg), device=self.device)
    heights, widths = split_heights(self.terrain_cfg)
    self.split_h = torch.as_tensor(heights, device=self.device)
    self.split_w = torch.as_tensor(widths, device=self.device)
    self.compacted_scale = torch.ones_like(self.litter_depth)
    self.loose_depth = torch.full_like(self.litter_depth, cfg.loose_min_depth)
    self.previous_contact = torch.zeros_like(self.was_air)
    self.tracking = torch.zeros_like(self.was_air)
    self.swing_target = torch.zeros_like(self.peak)
    self.air_pitch = torch.zeros_like(self.peak)
    # Lowest and highest support under the sole over the current swing.
    self.swing_bed_lo = torch.zeros_like(self.peak)
    self.swing_bed_hi = torch.zeros_like(self.peak)
    self.last_bed = torch.zeros((self.num_envs, 2, self.endpoints.shape[1]), device=self.device)
    joints, _ = self._entity.find_joints(".*_(hip|knee|ankle).*_joint")
    if len(joints) != 12:
      raise ValueError("Mechanical power requires twelve G1 leg joints")
    self.leg_dofs = self._entity.data.indexing.joint_v_adr[joints].long()

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.compacted_scale[ids] = torch.empty_like(self.compacted_scale[ids]).uniform_(*self.cfg.compacted_scale_range)
    deepest = max(self.cfg.litter_depth_range[1], self.cfg.loose_min_depth)
    self.loose_depth[ids] = torch.empty_like(self.loose_depth[ids]).uniform_(self.cfg.loose_min_depth, deepest)
    for buf in (self.previous_contact, self.tracking, self.swing_target, self.air_pitch,
                self.swing_bed_lo, self.swing_bed_hi):
      buf[ids] = 0

  def world_geometry(self):
    data = self._entity.data
    rot = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
    points, velocity = point_kinematics(
      data.body_link_pos_w[:, self.body_ids], rot, data.body_link_lin_vel_w[:, self.body_ids],
      data.body_link_ang_vel_w[:, self.body_ids], self.endpoints, self.radii)
    return points, velocity, rot

  def substrate(self, points):
    material, valid = material_at(points, self.terrain_cfg, self.layout, self.col_kind)
    bed = self.materials[material, 0] + split_offset(points, self.terrain_cfg, self.split_h, self.split_w)
    depth = torch.where(material == LOOSE, self.loose_depth[:, None, None],
                        self.materials[material, 1] * self.compacted_scale[:, None, None])
    return bed, depth * valid, valid

  def geometry(self):
    points, velocity, rot = self.world_geometry()
    bed, depth, valid = self.substrate(points)
    self.last_bed.copy_(bed)
    points[..., 2] -= bed
    return points, velocity, rot, valid, depth

  def apply_actions(self):
    points, velocity, _ = self.world_geometry()
    bed, depth, _ = self.substrate(points)
    # Keep world-space moment arms while measuring burial above each point's local bed.
    force, torque = litter_wrench(points, velocity, self._entity.data.body_com_pos_w[:, self.body_ids],
                                 depth, self.strength, self.cfg.max_drag_force, floor_height=bed)
    self.last_force.copy_(force)
    self.last_torque.copy_(torque)
    self._entity.write_external_wrench_to_sim(force, torque, body_ids=self.body_ids)

  def sample(self):
    key = (self._env.common_step_counter, self._env._sim_step_counter)
    if self._sample_key == key:
      return self._sample
    s = super().sample()
    takeoff = s["active"] & s["air"] & self.previous_contact
    # mjlab computes rewards before advancing commands. This is the target seen at takeoff.
    command = self._env.command_manager.get_command("clearance")
    self.swing_target.copy_(torch.where(takeoff, command, self.swing_target))
    # Contact chatter at toe-off can otherwise finish several fictitious swings.
    flight = self._env.scene["feet_ground_contact"].data.last_air_time
    s["tracked_landing"] = self.tracking & s["landing"] & (flight >= .06)
    # Sole pitch over the last airborne step, kept through the landing step.
    self.air_pitch.copy_(torch.where(s["air"], s["toe_down"], self.air_pitch))
    s["heel_first"] = s["tracked_landing"] & (self.air_pitch < -HEEL_FIRST_RAD)
    # A swing that steps up or down (split-level strips, compacted bed edges) is
    # not scored for apex: its peak above the LOCAL bed includes the step, so
    # the apex term would charge the lift that clears it.
    lo, hi = self.last_bed.amin(-1), self.last_bed.amax(-1)
    self.swing_bed_lo.copy_(torch.where(takeoff, lo, torch.minimum(self.swing_bed_lo, lo)))
    self.swing_bed_hi.copy_(torch.where(takeoff, hi, torch.maximum(self.swing_bed_hi, hi)))
    s["level_landing"] = s["tracked_landing"] & (self.swing_bed_hi - self.swing_bed_lo < .005)
    self.tracking.copy_((self.tracking | takeoff) & s["air"] & s["active"])
    self.previous_contact.copy_(s["contact"])
    return s


def apex_error(apex, target):
  # Smooth L1 in units of 2 cm, with bounded influence from exceptional swings.
  error = torch.sqrt(1 + ((apex - target) / .02).square()) - 1
  return error / (1 + error / 8)


def swing_apex(env):
  surface = env.action_manager.get_term("poultry_surface")
  s = surface.sample()
  return (apex_error(surface.last_apex, surface.swing_target) * s["level_landing"]).sum(-1) / env.step_dt


def lift_first(env):
  surface = env.action_manager.get_term("poultry_surface")
  s = surface.sample()
  # Translating early below the command is costly, descending for touchdown is free.
  # The landing apex objective handles overshoot.
  air_time = env.scene["feet_ground_contact"].data.current_air_time
  lifting = air_time < .65 * surface.cfg.period * (1 - surface.cfg.stance_fraction)
  speed = s["velocity"][..., :2].norm(dim=-1).mean(-1)
  deficit = (surface.swing_target - s["lowest"]).clamp_min(0)
  return (deficit * speed * surface.tracking * lifting).sum(-1)


def loose_plow(env):
  """Horizontal sole travel buried in litter, including before liftoff.

  Burial counts only below the commanded clearance: when the command is lower
  than the layer (4 cm asked on 8 cm of litter), travelling at the commanded
  height is obeying the command and is not charged.
  """
  s = env.action_manager.get_term("poultry_surface").sample()
  command = env.command_manager.get_command("clearance")[..., None]
  allowed = torch.minimum(s["layer"], command)
  buried = (allowed - s["points"][..., 2]).clamp_min(0).minimum(allowed)
  speed = s["velocity"][..., :2].norm(dim=-1)
  return ((speed * buried).mean(-1) * s["active"]).sum(-1)


def leg_power_w(env):
  dofs = env.action_manager.get_term("poultry_surface").leg_dofs
  data = env.sim.data
  return (data.qfrc_actuator[:, dofs] * data.qvel[:, dofs]).abs().sum(-1)


def power_cost(env):
  return (leg_power_w(env) / 100).clamp_max(4)


def touchdown_load(env):
  sensor = env.scene["feet_ground_contact"].data
  loading = (sensor.current_contact_time > 0) & (sensor.current_contact_time <= .04 + 1e-6)
  gate = loading & (sensor.last_air_time >= .06) & env.command_manager.get_term("twist_held").moving[:, None]
  peak = sensor.force_history[..., 2].abs().amax(-1)
  return (((peak - 320) / 150).clamp(0, 4) * gate).sum(-1)


def touchdown_tilt(env):
  """Level sole over the last 1.5 cm of descent and at the landing step.

  The 47k->50k landing fine-tune form (heel-first steps ~100% -> <10%). The
  window ends at contact: there is no standing or loading-phase term, because
  on soft litter a loaded heel sinks and an absolute-pitch cost while standing
  could not be removed (the 50k rocking). Heights are above the local bed, so
  a deep litter layer does not widen the window. Litter beds are level; the
  heightfield columns are excluded by ``active``.
  """
  surface = env.action_manager.get_term("poultry_surface")
  s = surface.sample()
  z = s["rotation"][..., :, 2]
  angle = torch.atan2(z[..., :2].norm(dim=-1), z[..., 2])
  cost = ((angle - math.radians(1.5)).clamp_min(0) / math.radians(5)).clamp_max(3)
  low = s["points"][..., 2].argmin(-1, keepdim=True)
  vz = s["velocity"][..., 2].gather(-1, low).squeeze(-1)
  approaching = surface.tracking & s["air"] & (s["lowest"] < .015) & (vz < -.04)
  return (cost * (s["tracked_landing"] | approaching)).sum(-1)


def litter_truth(env):
  """Critic only: per-foot litter depth, bed elevation, litter-tile flag, drag strength."""
  surface = env.action_manager.get_term("poultry_surface")
  s = surface.sample()
  return torch.cat((s["layer"].mean(-1) / .08, surface.last_bed.mean(-1) / .012,
                    s["valid"].float().mean(-1), surface.strength[:, None] / 1000), dim=-1)


def swing_error_sum_m(env):
  """Absolute apex error summed over level swings; divide by level_swings."""
  surface = env.action_manager.get_term("poultry_surface")
  completed = surface.sample()["level_landing"]
  return ((surface.last_apex - surface.swing_target).abs() * completed).sum(-1)


def swing_count(env):
  return env.action_manager.get_term("poultry_surface").sample()["tracked_landing"].sum(-1)


def level_swing_count(env):
  return env.action_manager.get_term("poultry_surface").sample()["level_landing"].sum(-1)


def heel_first_count(env):
  return env.action_manager.get_term("poultry_surface").sample()["heel_first"].sum(-1)


def landing_pitch_sum_deg(env):
  """Toe-down pitch just before contact, summed over completed swings (negative = heel first)."""
  surface = env.action_manager.get_term("poultry_surface")
  s = surface.sample()
  return (torch.rad2deg(surface.air_pitch) * s["tracked_landing"]).sum(-1)


##
# Sustained pulls: give way to a strong push or pull instead of resisting it.
##


@dataclass(kw_only=True)
class PullForceCfg(ActionTermCfg):
  """A hand-like push or pull: horizontal, held 1-3 s, ramped on and off.

  The velocity pushes are instant kicks; nothing in training lasted long
  enough to need a step, and the stand-still terms (position anchor, feet
  still, feet square) made leaning back against a pull cheaper than stepping
  with it. Magnitudes are uniform from 0 to ``max_force`` times the curriculum
  scale. Light pulls are still resisted; above ``release_force`` and for
  ``settle_time`` after, the stand-still terms are off and the anchor
  re-latches where the robot ends up. 80 N at the torso moves the centre of
  pressure ~0.23 m on a 35 kg robot, past the feet: it has to step.
  """

  body_names: tuple[str, ...] = ("torso_link", "pelvis")
  max_force: float = 80.
  duration_range: tuple[float, float] = (1., 3.)
  gap_range: tuple[float, float] = (3., 8.)
  ramp_time: float = .25
  release_force: float = 20.
  settle_time: float = 1.
  initial_scale: float = 0.

  def build(self, env):
    return PullForce(self, env)


class PullForce(ActionTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.body_ids, _ = self._entity.find_bodies(cfg.body_names, preserve_order=True)
    n = self.num_envs
    self.scale = cfg.initial_scale
    self._raw_action = torch.zeros((n, 0), device=self.device)
    self.target = torch.zeros((n, 3), device=self.device)
    self.force = torch.zeros_like(self.target)
    self.body = torch.zeros(n, dtype=torch.long, device=self.device)
    self.elapsed = torch.zeros(n, device=self.device)
    self.duration = torch.zeros(n, device=self.device)
    self.gap = torch.empty(n, device=self.device).uniform_(*cfg.gap_range)
    self.pulling = torch.zeros(n, dtype=torch.bool, device=self.device)
    self.since_strong = torch.full((n,), 1e3, device=self.device)
    self.wrench = torch.zeros((n, len(self.body_ids), 3), device=self.device)

  @property
  def action_dim(self):
    return 0

  @property
  def raw_action(self):
    return self._raw_action

  @property
  def released(self):
    """Strong pull now, or one ended less than ``settle_time`` ago."""
    return self.since_strong < self.cfg.settle_time

  def process_actions(self, actions):
    # Once per policy step: advance the pull schedule.
    dt = self._env.step_dt
    self.gap.sub_(dt * ~self.pulling)
    start = ~self.pulling & (self.gap <= 0)
    ids = start.nonzero(as_tuple=False).flatten()
    if len(ids):
      magnitude = torch.rand(len(ids), device=self.device) * self.cfg.max_force * self.scale
      yaw = torch.rand(len(ids), device=self.device) * 2 * math.pi
      self.target[ids] = torch.stack((magnitude * yaw.cos(), magnitude * yaw.sin(), 0 * yaw), -1)
      self.body[ids] = torch.randint(len(self.body_ids), (len(ids),), device=self.device)
      self.duration[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.duration_range)
      self.elapsed[ids] = 0
      self.pulling[ids] = True
    self.elapsed.add_(dt * self.pulling)
    done = self.pulling & (self.elapsed >= self.duration)
    self.pulling &= ~done
    self.gap[done] = torch.empty(int(done.sum()), device=self.device).uniform_(*self.cfg.gap_range)
    envelope = (torch.minimum(self.elapsed, self.duration - self.elapsed) / self.cfg.ramp_time).clamp(0, 1)
    self.force.copy_(self.target * (envelope * self.pulling)[:, None])
    strong = self.force.norm(dim=-1) > self.cfg.release_force
    self.since_strong.copy_(torch.where(strong, 0., self.since_strong + dt))
    self.wrench.zero_()
    self.wrench[torch.arange(self.num_envs, device=self.device), self.body] = self.force

  def apply_actions(self):
    self._entity.write_external_wrench_to_sim(self.wrench, torch.zeros_like(self.wrench), body_ids=self.body_ids)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.pulling[ids] = False
    self.force[ids] = 0
    self.wrench[ids] = 0
    self.since_strong[ids] = 1e3
    self.gap[ids] = torch.empty_like(self.gap[ids]).uniform_(*self.cfg.gap_range)
    self._entity.write_external_wrench_to_sim(self.wrench[ids], torch.zeros_like(self.wrench[ids]),
                                              env_ids=env_ids, body_ids=self.body_ids)


def pull_released(env):
  if "pull_force" not in env.action_manager.active_terms:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env.action_manager.get_term("pull_force").released


def pull_scale_schedule(env, env_ids, stages):
  """Pull magnitude as a fraction of max_force along (policy_step, fraction) knots."""
  del env_ids
  term = env.action_manager.get_term("pull_force")
  term.scale = piecewise_linear(env.common_step_counter, stages)
  return torch.tensor([term.scale])


class idle_position_anchor_pull(idle_position_anchor_lean):
  """The standing position anchor, released during and just after a strong pull.

  Re-latched every released step, so it latches again where the robot stands
  once the pull is over and it has settled.
  """

  def __call__(self, env, **kwargs):
    ids = pull_released(env).nonzero(as_tuple=False).flatten()
    if len(ids):
      self.reset(ids)
    return super().__call__(env, **kwargs)


def idle_feet_still_pull(env, **kwargs):
  return idle_feet_still(env, **kwargs) * ~pull_released(env)


def idle_base_motion_pull(env, **kwargs):
  return idle_base_motion(env, **kwargs) * ~pull_released(env)


def stand_feet_square_pull(env, **kwargs):
  return settle.stand_feet_square(env, **kwargs) * ~pull_released(env)


# The stand-still terms these replace, checked by the task config.
idle_feet_still_pull.base = idle_feet_still
idle_base_motion_pull.base = idle_base_motion
stand_feet_square_pull.base = settle.stand_feet_square
idle_position_anchor_pull.base = idle_position_anchor_lean


def pull_state(env):
  """Critic only: the applied pull in the pelvis frame (/100 N) and the release flag."""
  term = env.action_manager.get_term("pull_force")
  force_b = quat_apply_inverse(env.scene["robot"].data.root_link_quat_w, term.force)
  return torch.cat((force_b / 100, term.released.float()[:, None]), -1)


def pull_force_n(env):
  return env.action_manager.get_term("pull_force").force.norm(dim=-1)


def pull_released_fraction(env):
  return pull_released(env).float()
