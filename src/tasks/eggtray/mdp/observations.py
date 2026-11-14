"""Observation terms for the arm stabiliser.

Actor terms use only what the robot can sense: arm joint encoders, the robot
IMU, the operator's walking commands, the gait clock DUET already uses, a
tray-mounted IMU (orientation estimate, gyro, accelerometer) and the arm
policy's own previous action. Critic-only terms are marked privileged.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.eggtray.mdp.state import state


def _arm(env):
  return env.action_manager.get_term("arm_residual")


def arm_joint_pos(env) -> torch.Tensor:
  """Arm encoder angles (with encoder bias) relative to this episode's carry reference."""
  term = _arm(env)
  return env.scene["robot"].data.joint_pos_biased[:, term.joint_ids] - term.reference


def arm_joint_vel(env) -> torch.Tensor:
  return env.scene["robot"].data.joint_vel[:, _arm(env).joint_ids]


def carry_reference_offset(env) -> torch.Tensor:
  """Episode carry reference minus the nominal pose (0 unless randomised)."""
  term = _arm(env)
  return term.reference - term.q_nominal


def tray_gravity(env) -> torch.Tensor:
  """Gravity direction in the tray frame: the tray IMU's attitude estimate."""
  quat = env.scene["tray"].data.root_link_quat_w
  g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(quat.shape[0], -1)
  return quat_apply_inverse(quat, g)


# -- Privileged (critic only) -------------------------------------------------


def tray_lin_vel_b(env) -> torch.Tensor:
  td = env.scene["tray"].data
  return quat_apply_inverse(td.root_link_quat_w, td.root_link_lin_vel_w)


def egg_rel_pos(env) -> torch.Tensor:
  return state(env).egg_relative()[0].flatten(1)


def egg_rel_vel(env) -> torch.Tensor:
  return state(env).egg_relative()[1].flatten(1)


def grasp_strain(env) -> torch.Tensor:
  return state(env).grasp_strain()


def duet_action(env) -> torch.Tensor:
  return env.action_manager.get_term("duet_lower_body").raw_action


def litter_depth(env) -> torch.Tensor:
  return env.action_manager.get_term("poultry_surface").litter_depth[:, None]
