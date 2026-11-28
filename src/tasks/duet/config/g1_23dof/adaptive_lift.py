"""Scratch Adaptive_Lift with the DUET lean, arm and payload training contract.

Terrain, friction, litter drag and spawn are the 47k lean task's, except that
its flat and soft_floor columns become litter tiles with a support bed. The
heightfield columns (slopes, rough, waves) and the terrain-level curriculum
are unchanged.
"""
import math
from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...duet_env_cfg import ITER
from ...mdp import adaptive_lift as lift
from ...mdp import settle
from ...mdp.lift_terrain import LiftTile
from .env_cfgs import LOWER_BODY_ACTUATORS
from .poultry_lean import LEAN_MAX, STOOP_HEIGHT, unitree_g1_23dof_duet_poultry_lean_env_cfg
from .poultry_surface import LIFT_WEIGHT_STAGES, poultry_surface_ppo_runner_cfg

TASK = "Adaptive_Lift"
# Litter columns, replacing the 47k flat (0.15) and soft_floor (0.60) columns.
# With 20 columns: rigid 3, compacted 4, loose 5, mixed 3; the remaining 5 are
# the 47k heightfield columns (slope 1, inverted slope 1, rough 1, waves 2).
TILE_PROPORTIONS = {"rigid": .15, "compacted": .20, "loose": .25, "mixed": .15}
REPLACED_TILES = ("flat", "soft_floor")
# Deepest loose litter per episode, (iteration, m). Shallow while the walk is
# learned, then to 8 cm (the terrain_v1 deep litter) by 4,000, as the 47k's
# 2 -> 6 cm ramp. Each episode draws uniformly from 2 cm to this depth.
LOOSE_DEPTH_STAGES = ((0, .03), (1_500, .03), (4_000, .08))


def env_cfg(play=False):
  cfg = unitree_g1_23dof_duet_poultry_lean_env_cfg(play=play)
  cfg.commands["clearance"] = lift.LiftCommandCfg(resampling_time_range=(2., 5.))
  for group in ("actor", "critic"):
    cfg.observations[group].terms["clearance_commands"] = ObservationTermCfg(func=lift.clearance_commands)
  # Privileged litter state for the value function; the actor stays blind.
  cfg.observations["critic"].terms["litter_truth"] = ObservationTermCfg(func=lift.litter_truth)

  if not play:
    lean = cfg.commands["lean"]
    lean.min_lean = 0.
    lean.max_lean_stages = ((0, 0.), (6000 * ITER, 0.), (14000 * ITER, LEAN_MAX))
    lean.stoop_floor_stages = ((0, .60), (6000 * ITER, .60), (14000 * ITER, STOOP_HEIGHT[0]))
    # Dedicated in-place turns and pure sidesteps (a fifth of moving draws each).
    cfg.commands["twist"] = lift.PoultryTwistCfg(**vars(cfg.commands["twist"]))

  old = cfg.actions["poultry_surface"]
  deepest = LOOSE_DEPTH_STAGES[-1][1] if play else LOOSE_DEPTH_STAGES[0][1]
  # Drag strength and force cap stay at the 47k values (500-1500 N s/m^2, 150 N).
  cfg.actions["poultry_surface"] = lift.LiftSurfaceCfg(
    entity_name="robot", period=old.period, stance_fraction=old.stance_fraction,
    max_layer_height=LOOSE_DEPTH_STAGES[-1][1], litter_depth_range=(0., deepest))
  generator = cfg.scene.terrain.terrain_generator
  tiles = {name: LiftTile(kind=i, proportion=TILE_PROPORTIONS[name])
           for i, name in enumerate(TILE_PROPORTIONS)}
  kept = {name: sub for name, sub in generator.sub_terrains.items() if name not in REPLACED_TILES}
  cfg.scene.terrain.terrain_generator = replace(generator, sub_terrains={**tiles, **kept})
  if not play:
    cfg.curriculum["litter_depth"].params["stages"] = tuple((i * ITER, d) for i, d in LOOSE_DEPTH_STAGES)
  for sensor in cfg.scene.sensors:
    if sensor.name == "feet_ground_contact":
      sensor.history_length = cfg.decimation

  # One swing apex objective. Early travel and buried travel shape the path.
  # A level landing (approach window) replaces the near-floor foot-level term.
  # Touchdown weights are the 47k->50k landing fine-tune's.
  cfg.rewards.pop("poultry_foot_level")
  cfg.curriculum.pop("weight_poultry_foot_level", None)
  for name, func, weight in (
    ("poultry_clearance", lift.lift_first, -4.),
    ("poultry_swing_height", lift.swing_apex, -.16),
    ("poultry_plow", lift.loose_plow, -6.),
    ("touchdown_tilt", lift.touchdown_tilt, -2.),
    ("touchdown_load", lift.touchdown_load, -1.2),
    ("leg_power", lift.power_cost, -.08),
  ):
    cfg.rewards[name] = RewardTermCfg(func=func, weight=weight if play else 0.)
    if not play:
      cfg.curriculum["weight_" + name] = CurriculumTermCfg(
        func=settle.reward_weight_schedule,
        params={"reward_name": name, "stages": tuple((i * ITER, f * weight) for i, f in LIFT_WEIGHT_STAGES)})

  # Turning and sidestepping, HOMIE's G1 terms: yaw exp(-4 err^2) at 2.0 (was
  # exp(-2 err^2) at 1.75, nearly flat for small yaw errors) and a separate
  # lateral term exp(-4 err^2) at 1.0 next to the combined xy term.
  cfg.rewards["track_angular_velocity"].weight = 2.
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(.25)
  cfg.rewards["track_lateral_velocity"] = RewardTermCfg(
    func=lift.track_lateral_velocity, weight=1., params={"std": math.sqrt(.25)})

  surface = cfg.actions["poultry_surface"]
  cfg.rewards["feet_air_time"].params["threshold"] = surface.period * (1 - surface.stance_fraction)
  cfg.rewards["foot_slip"].weight = -.35
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LOWER_BODY_ACTUATORS)
  cfg.metrics["poultry_plow_cost"] = MetricsTermCfg(func=lift.loose_plow)
  cfg.metrics["leg_power_w"] = MetricsTermCfg(func=lift.leg_power_w)
  cfg.metrics["swing_error_sum_m"] = MetricsTermCfg(func=lift.swing_error_sum_m)
  cfg.metrics["completed_swings"] = MetricsTermCfg(func=lift.swing_count)
  cfg.metrics["heel_first_landings"] = MetricsTermCfg(func=lift.heel_first_count)
  cfg.metrics["landing_pitch_sum_deg"] = MetricsTermCfg(func=lift.landing_pitch_sum_deg)
  return cfg


def runner_cfg():
  cfg = poultry_surface_ppo_runner_cfg()
  cfg.experiment_name = TASK
  cfg.resume = False
  return cfg
