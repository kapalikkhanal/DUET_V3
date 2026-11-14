"""Reward and metric terms for the egg-tray stabiliser.

Few terms, each with one job (weights and the reasoning are in
``config/g1_23dof/env_cfgs.py``):

  payload    egg_lost, egg_motion, egg_impact      (privileged sim state)
  tray       tray_acc (positive kernel), tray_tilt_excess, tray_ang_vel
  arm        action rate, joint acceleration, residual size, grasp strain
  safety     is_terminated (falls, lost eggs, tray tipped) -- mjlab term

Tilt is penalised only beyond a deadband, so a small deliberate tilt that
keeps the eggs seated during acceleration costs nothing.

Damage proxies (``egg_impact``) are contact forces between egg and tray, not a
calibrated breakage model; see docs/eggtray_rl.md.
"""

from __future__ import annotations

import math

import torch

from src.tasks.eggtray.mdp.state import state


def _kept(s) -> torch.Tensor:
  return (~s["lost"]).float()


def tray_acc_exp(env, std: float = 2.0) -> torch.Tensor:
  """exp(-|a|^2 / std^2), a = gravity-compensated tray acceleration (m/s^2)."""
  a = state(env).sample()["acc"]
  return torch.exp(-a.square().sum(-1) / std**2)


def tray_tilt_excess(env, deadband_deg: float = 3.0) -> torch.Tensor:
  """Squared tilt (rad) beyond a deadband."""
  tilt = state(env).sample()["tilt"]
  return (tilt - math.radians(deadband_deg)).clamp_min(0.0).square()


def tray_ang_vel_l2(env) -> torch.Tensor:
  """Squared roll/pitch rate of the tray (its own frame; yaw rate excluded)."""
  return state(env).sample()["ang_vel_b"][:, :2].square().sum(-1)


def egg_lost(env) -> torch.Tensor:
  """Number of eggs that left their cups this step."""
  return state(env).sample()["newly_lost"].float().sum(-1)


def egg_motion_l2(env) -> torch.Tensor:
  """Sum of squared egg speeds relative to their cups (rattle), seated eggs only."""
  s = state(env).sample()
  return (s["egg_rel_vel"].square().sum(-1) * _kept(s)).sum(-1)


def egg_impact(env, threshold: float = 1.5) -> torch.Tensor:
  """Net egg contact force above ``threshold`` N (resting load ~0.6 N), summed."""
  s = state(env).sample()
  return ((s["egg_force"] - threshold).clamp_min(0.0) * _kept(s)).sum(-1)


def arm_residual_l2(env) -> torch.Tensor:
  """Squared residual as a fraction of its bound: stay near the carry pose."""
  term = env.action_manager.get_term("arm_residual")
  return (term.residual / term.cfg.max_residual).square().sum(-1)


def arm_action_rate_l2(env) -> torch.Tensor:
  am = env.action_manager
  return (am.action - am.prev_action).square().sum(-1)


def arm_joint_acc_l2(env) -> torch.Tensor:
  term = env.action_manager.get_term("arm_residual")
  return env.scene["robot"].data.joint_acc[:, term.joint_ids].square().sum(-1)


def grasp_strain_l2(env) -> torch.Tensor:
  """Squared weld stretch (m^2): internal squeeze from inconsistent arm targets."""
  return state(env).sample()["grasp_strain"].square().sum(-1)


# -- Metrics (Episode_Metrics/*, per-step means) -------------------------------


def eggtray_metric(env, name: str) -> torch.Tensor:
  s = state(env).sample()
  if name == "acc":
    return s["acc"].norm(dim=-1)
  if name == "acc_imu":
    return s["acc_imu"].norm(dim=-1)
  if name == "jerk":
    return s["jerk"].norm(dim=-1)
  if name == "tilt_deg":
    return torch.rad2deg(s["tilt"])
  if name == "ang_vel":
    return s["ang_vel_b"][:, :2].norm(dim=-1)
  if name == "eggs_kept":
    return _kept(s).mean(-1)
  if name == "egg_force_max":
    return (s["egg_force"] * _kept(s)).amax(-1)
  if name == "egg_speed":
    return (s["egg_rel_vel"].norm(dim=-1) * _kept(s)).mean(-1)
  if name == "grasp_strain_mm":
    return 1000 * s["grasp_strain"].amax(-1)
  if name == "residual_rad":
    return env.action_manager.get_term("arm_residual").residual.abs().mean(-1)
  if name == "duet_extreme":
    return env.action_manager.get_term("duet_lower_body").extreme_count
  raise ValueError(name)
