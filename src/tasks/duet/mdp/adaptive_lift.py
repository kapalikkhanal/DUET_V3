"""Continuous support-relative swing commands and LiftFirst shaping."""
from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .lift_terrain import LAYOUT, MATERIALS, material_at
from .poultry_surface import PoultrySurfaceAction, PoultrySurfaceActionCfg, litter_wrench, point_kinematics


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
class LiftSurfaceCfg(PoultrySurfaceActionCfg):
  def build(self, env):
    return LiftSurface(self, env)


class LiftSurface(PoultrySurfaceAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.layout = torch.as_tensor(LAYOUT, device=self.device)
    self.materials = torch.tensor(MATERIALS, device=self.device)
    self.depth_scale = torch.ones_like(self.litter_depth)
    self.previous_contact = torch.zeros_like(self.was_air)
    self.tracking = torch.zeros_like(self.was_air)
    self.swing_target = torch.zeros_like(self.peak)
    joints, _ = self._entity.find_joints(".*_(hip|knee|ankle).*_joint")
    if len(joints) != 12:
      raise ValueError("Mechanical power requires twelve G1 leg joints")
    self.leg_dofs = self._entity.data.indexing.joint_v_adr[joints].long()

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.depth_scale[ids] = torch.empty_like(self.depth_scale[ids]).uniform_(.8, 1.2)
    self.previous_contact[ids] = False
    self.tracking[ids] = False
    self.swing_target[ids] = 0

  def world_geometry(self):
    data = self._entity.data
    rot = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
    points, velocity = point_kinematics(
      data.body_link_pos_w[:, self.body_ids], rot, data.body_link_lin_vel_w[:, self.body_ids],
      data.body_link_ang_vel_w[:, self.body_ids], self.endpoints, self.radii)
    return points, velocity, rot

  def substrate(self, points):
    material, valid = material_at(points, self.terrain_cfg, self.layout)
    return self.materials[material, 0], self.materials[material, 1] * self.depth_scale[:, None, None], valid

  def geometry(self):
    points, velocity, rot = self.world_geometry()
    bed, depth, valid = self.substrate(points)
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
  return (apex_error(surface.last_apex, surface.swing_target) * s["tracked_landing"]).sum(-1) / env.step_dt


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
  s = env.action_manager.get_term("poultry_surface").sample()
  buried = torch.minimum((-s["clearance"]).clamp_min(0), s["layer"])
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
  s = env.action_manager.get_term("poultry_surface").sample()
  normal = s["rotation"][..., :, 2]
  angle = torch.atan2(normal[..., :2].norm(dim=-1), normal[..., 2])
  return ((angle - .0873).clamp_min(0) * s["tracked_landing"]).sum(-1) / env.step_dt


def swing_error_sum_m(env):
  surface = env.action_manager.get_term("poultry_surface")
  completed = surface.sample()["tracked_landing"]
  return ((surface.last_apex - surface.swing_target).abs() * completed).sum(-1)


def swing_count(env):
  return env.action_manager.get_term("poultry_surface").sample()["tracked_landing"].sum(-1)
