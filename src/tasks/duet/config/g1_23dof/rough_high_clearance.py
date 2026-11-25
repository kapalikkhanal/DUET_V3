"""Conventional high-clearance reward on the full lean_47000 task."""

from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import settle

from .env_cfgs import unitree_g1_23dof_duet_rough_env_cfg
from .poultry_lean import unitree_g1_23dof_duet_poultry_lean_env_cfg
from .poultry_surface import LIFT_TERMS, LIFT_WEIGHT_STAGES, poultry_surface_ppo_runner_cfg


def rough_high_clearance_env_cfg(play=False):
  cfg = unitree_g1_23dof_duet_poultry_lean_env_cfg(play=play)
  # Litter depth and physical drag remain part of terrain exposure. The
  # lowest-sole, near-surface travel, apex and foot-level reward terms go away.
  for name in LIFT_TERMS:
    del cfg.rewards[name]
    if not play:
      del cfg.curriculum[f"weight_{name}"]

  clearance = unitree_g1_23dof_duet_rough_env_cfg(play=play).rewards["foot_clearance"]
  clearance.params["target_height"] = 0.14  # conventional foot site, world z (m)
  cfg.rewards["foot_clearance"] = clearance
  if not play:
    stages = tuple((i * ITER, fraction * clearance.weight) for i, fraction in LIFT_WEIGHT_STAGES)
    clearance.weight = stages[0][1]
    cfg.curriculum["weight_foot_clearance"] = CurriculumTermCfg(
      func=settle.reward_weight_schedule, params={"reward_name": "foot_clearance", "stages": stages},
    )
  return cfg


def rough_high_clearance_ppo_runner_cfg():
  # RSL-RL numbers a fresh run from zero, so the final file is model_47000.pt.
  return replace(
    poultry_surface_ppo_runner_cfg(),
    experiment_name="DUET_G1_23dof_rough_high_clearance",
    max_iterations=47_001,
  )
