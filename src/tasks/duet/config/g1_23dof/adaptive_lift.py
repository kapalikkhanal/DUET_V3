"""Scratch Adaptive_Lift with the DUET lean, arm and payload training contract."""
from dataclasses import replace

import torch
from mjlab.envs import mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...duet_env_cfg import ITER
from ...mdp import adaptive_lift as lift
from ...mdp import settle
from ...mdp.lift_terrain import TILE_NAMES, LiftTile
from .env_cfgs import LOWER_BODY_ACTUATORS
from .poultry_lean import LEAN_MAX, STOOP_HEIGHT, unitree_g1_23dof_duet_poultry_lean_env_cfg
from .poultry_surface import LIFT_WEIGHT_STAGES, poultry_surface_ppo_runner_cfg

TASK = "Adaptive_Lift"
TERRAIN_STAGES = (1000, 2500, 4000)


def reset_tile(env, env_ids, play):
  stage = len(TERRAIN_STAGES) if play else sum(env.common_step_counter >= i * ITER for i in TERRAIN_STAGES)
  terrain = env.scene.terrain
  column = torch.randint(stage + 1, (len(env_ids),), device=env.device)
  row = terrain.terrain_origins.shape[0] // 2
  terrain.terrain_types[env_ids] = column
  terrain.terrain_levels[env_ids] = row
  terrain.env_origins[env_ids] = terrain.terrain_origins[row, column]


def env_cfg(play=False):
  cfg = unitree_g1_23dof_duet_poultry_lean_env_cfg(play=play)
  cfg.commands["clearance"] = lift.LiftCommandCfg(resampling_time_range=(2., 5.))
  for group in ("actor", "critic"):
    cfg.observations[group].terms["clearance_commands"] = ObservationTermCfg(func=lift.clearance_commands)

  if not play:
    lean = cfg.commands["lean"]
    lean.min_lean = 0.
    lean.max_lean_stages = ((0, 0.), (6000 * ITER, 0.), (14000 * ITER, LEAN_MAX))
    lean.stoop_floor_stages = ((0, .60), (6000 * ITER, .60), (14000 * ITER, STOOP_HEIGHT[0]))

  cfg.actions["poultry_surface"] = lift.LiftSurfaceCfg(
    entity_name="robot", period=cfg.actions["poultry_surface"].period,
    stance_fraction=cfg.actions["poultry_surface"].stance_fraction,
    drag_strength_range=(400., 900.), max_drag_force=45.)
  terrain = cfg.scene.terrain
  terrain.terrain_generator = replace(
    terrain.terrain_generator, size=(12., 8.), num_rows=3, num_cols=len(TILE_NAMES),
    border_width=5., sub_terrains={name: LiftTile(kind=i, proportion=1.) for i, name in enumerate(TILE_NAMES)})
  terrain.max_init_terrain_level = 0
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"] = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                         params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}})
  cfg.events = {"lift_tile": EventTermCfg(func=reset_tile, mode="reset", params={"play": play}), **cfg.events}
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("litter_depth", None)
  cfg.events["foot_friction"].params["ranges"] = (.50, 1.05)
  for sensor in cfg.scene.sensors:
    if sensor.name == "feet_ground_contact":
      sensor.history_length = cfg.decimation

  # One swing apex objective. Early travel and buried travel shape the path.
  # Landing orientation replaces the fixed near-floor foot-level objective.
  cfg.rewards.pop("poultry_foot_level")
  cfg.curriculum.pop("weight_poultry_foot_level", None)
  for name, func, weight in (
    ("poultry_clearance", lift.lift_first, -4.),
    ("poultry_swing_height", lift.swing_apex, -.16),
    ("poultry_plow", lift.loose_plow, -6.),
    ("touchdown_tilt", lift.touchdown_tilt, -.12),
    ("touchdown_load", lift.touchdown_load, -.45),
    ("leg_power", lift.power_cost, -.08),
  ):
    cfg.rewards[name] = RewardTermCfg(func=func, weight=weight if play else 0.)
    if not play:
      cfg.curriculum["weight_" + name] = CurriculumTermCfg(
        func=settle.reward_weight_schedule,
        params={"reward_name": name, "stages": tuple((i * ITER, f * weight) for i, f in LIFT_WEIGHT_STAGES)})
  surface = cfg.actions["poultry_surface"]
  cfg.rewards["feet_air_time"].params["threshold"] = surface.period * (1 - surface.stance_fraction)
  cfg.rewards["foot_slip"].weight = -.35
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LOWER_BODY_ACTUATORS)
  cfg.metrics["poultry_plow_cost"] = MetricsTermCfg(func=lift.loose_plow)
  cfg.metrics["leg_power_w"] = MetricsTermCfg(func=lift.leg_power_w)
  cfg.metrics["swing_error_sum_m"] = MetricsTermCfg(func=lift.swing_error_sum_m)
  cfg.metrics["completed_swings"] = MetricsTermCfg(func=lift.swing_count)
  return cfg


def runner_cfg():
  cfg = poultry_surface_ppo_runner_cfg()
  cfg.experiment_name = TASK
  cfg.resume = False
  return cfg
