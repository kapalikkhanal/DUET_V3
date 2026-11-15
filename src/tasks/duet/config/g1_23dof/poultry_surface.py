"""Poultry-surface task: walking on litter, squats, externally driven arms.

Trained FROM SCRATCH in one run since 2026-11-14. The 91.5k policy was reached
through a chain of resumes (walking, then arms and height, then litter and
foot lift); on the farm it worked but wobbled, took balance steps, marched and
did not come back to a square stance. The task keeps that policy's rewards and
changes nine things, all in ``_train_from_scratch`` at the bottom of this file:

  1  gait period 0.60 s -> 0.85 s (a calm walk instead of a march)
  2  stiff leg gains (GR00T decoupled-WBC table) with a uniform 0.25 action scale
  3  the actor sees its last 6 observations instead of only the current one
  4  motor commands arrive 0-20 ms late, and PD gains vary per episode
  5  one second to settle after the stick is released, then stand square
  6  litter depth and foot-lift penalties ramp in, so nothing needs a resume
  7  a torso-upright term while standing and a stronger foot-squaring term
     (added after the first 2,000-iteration trial of 1-6, which walked forward
     well but stood leaning 9.7 deg forward with its feet 5-6 cm staggered)
  8  a gait reward that pays nothing for standing still under a walk command
     (added at 4,000 iterations of the full run: pure sideways commands were
     answered by standing; turning and backward walking were already learned)
  9  a wider foot-spacing band while a sideways velocity is commanded, and a
     stronger upright term (added at 19,000: sideways walking was capped at
     0.23 m/s by the spacing band, and the stance leaned 2.4-4.2 deg forward)

``legacy=True`` builds the task exactly as the 91.5k policy was trained. The
egg-tray task (frozen 91.5k legs) and any evaluation of the old checkpoints use
that. Other DUET variants (Flat, Rough, ablations) keep their configs.
"""

import copy
from dataclasses import replace

from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.duet.mdp as mdp
from src.assets.robots.unitree_g1.g1_23dof_deploy_constants import (
  ARM_ACTUATOR_GROUPS,
  G1_23DOF_DEPLOY_ACTION_SCALE,
  G1_23DOF_POULTRY_PD_GAINS,
  LEG_ACTUATOR_GROUPS,
  get_g1_23dof_deploy_robot_cfg,
)
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import settle
from src.tasks.duet.mdp.poultry_surface import (
  PoultrySurfaceActionCfg,
  action_excess_l2,
  backward_speed_deficit,
  joint_pos_limits_widened,
  poultry_clearance,
  poultry_foot_level,
  poultry_metric,
  poultry_plow,
  poultry_swing_height,
  stop_backward_walkers,
)
from .env_cfgs import (
  HEIGHT_RANGE,
  SQUAT_REACH_POSES,
  _arm_pose,
  unitree_g1_23dof_duet_rough_env_cfg,
)
from .rl_cfg import ROUGH_REWARD_FLOOR, unitree_g1_23dof_duet_ppo_runner_cfg

SQUAT_FLOOR = 0.22  # m; was HEIGHT_RANGE[0] = 0.24

# (1) Stride period. The 91.5k policy was trained at 0.60 s and measured
# 1.63 Hz on the robot: 3.3 foot strikes a second with a 0.24 s swing. At 0.85 s
# the swing is 0.37 s. deploy.yaml's gait_phase ``period`` must match.
GAIT_PERIOD = 0.85
# (3) Actor observation history, frames at 50 Hz. HOMIE and GR00T use 6.
HISTORY = 6
# (4) Motor command delay, physics steps of 5 ms: 0-20 ms, one value per episode.
DELAY_MAX_LAG = 4
# (5) Seconds between releasing the stick and the standing terms switching on.
# deploy.yaml's gait_phase ``settle`` must match.
SETTLE_TIME = 1.0
# (6) Schedules in training iterations. The walk is learned on almost bare
# floor with no foot-lift penalty; both then ramp in. With everything at full
# strength from iteration 0 a 2,000-iteration trial learned to stand still.
LITTER_DEPTH_STAGES = ((0, 0.02), (1_500, 0.02), (4_000, 0.06))  # deepest litter, m
LIFT_WEIGHT_STAGES = ((0, 0.0), (800, 0.0), (3_000, 1.0))  # fraction of the final weight
LIFT_TERMS = (
  "poultry_clearance", "poultry_plow", "poultry_swing_height", "poultry_foot_level",
)
# Raw-output threshold of action_excess_l2. The deepest squat needs the knee
# 2.5 rad from default: a raw output of ~10-11 at the 0.25 rad scale, where the
# old per-joint scale (knee 0.35) needed ~7 and the threshold was 8. The deploy
# guard stays at 15.
ACTION_EXCESS_THRESHOLD = 12.0


def unitree_g1_23dof_duet_poultry_surface_env_cfg(
  play=False, litter_enabled=True, *, litter_depth_mode=None,
  litter_depth_range=(0.0, 0.06), legacy=False,
):
  """Random episode depths in training; legacy row depths in play/evaluation.

  Pin a depth in either mode with litter_depth_mode="random" and
  litter_depth_range=(d, d), or set those fields on actions["poultry_surface"].

  ``legacy=True``: the task exactly as the 91.5k policy was trained (soft
  gains, 0.60 s gait, no history). Use it to run or evaluate the old
  checkpoints; they do not work in the current task and vice versa.
  """
  cfg = unitree_g1_23dof_duet_rough_env_cfg(play=play)
  # Cap every training stage as well as the unstaged play command range.
  if "command_vel" in cfg.curriculum:
    for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
      lo, hi = stage["lin_vel_x"]
      stage["lin_vel_x"] = (max(lo, -0.6), hi)
  twist = cfg.commands["twist"]
  lo, hi = twist.ranges.lin_vel_x
  twist.ranges.lin_vel_x = (max(lo, -0.6), hi)
  cfg.rewards["backward_speed_deficit"] = RewardTermCfg(
    func=backward_speed_deficit, weight=-1.5,
  )
  cfg.events["stop_backward_walkers"] = EventTermCfg(
    func=stop_backward_walkers, mode="interval", interval_range_s=(1.0, 3.0),
    params={"prob": 0.3},
  )

  # Restore squat balance in this Rough-derived task, including medium-height
  # squats and a fading centering penalty at the nominal standing height.
  cfg.rewards["squat_com_centering"] = RewardTermCfg(
    func=mdp.squat_com_centering, weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "height_command_name": "base_height",
      "squat_below": 0.65,
      "full_above": 0.85,
      "foot_half_length": 0.09,
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )
  cfg.rewards["body_orientation_l2"].params.update(
    relax_below=0.35, full_above=0.60, relax_factor=0.15,
  )
  arm = cfg.actions["upper_body_pose"]
  arm.squat_reach_poses = SQUAT_REACH_POSES + (
    _arm_pose(pitch=0.0, roll=1.2, yaw=0.0, elbow=0.30),
    _arm_pose(pitch=-0.6, roll=0.9, yaw=0.0, elbow=0.50),
  )
  arm.squat_reach_prob = 0.75
  arm.squat_reach_below = 0.65
  cfg.commands["base_height"].deep_squat_fraction = 0.6
  # Deeper squat (2026-10-12): the floor ramps 0.24 -> 0.22 m over iterations
  # 90k-92k (knots are policy steps, 24 per iteration), so a resume from 90k
  # is not asked for the new depth at once. Paired with the 97% squat-joint
  # limit band below; at 90% the kinematic floor was 0.29 m.
  base_height = cfg.commands["base_height"]
  base_height.height_range = (SQUAT_FLOOR, base_height.height_range[1])
  if base_height.floor_stages:
    base_height.floor_stages = tuple(base_height.floor_stages) + (
      (90_000 * 24, HEIGHT_RANGE[0]), (92_000 * 24, SQUAT_FLOOR),
    )
  elif base_height.floor_curriculum_start <= HEIGHT_RANGE[0]:
    base_height.floor_curriculum_start = SQUAT_FLOOR  # pinned (play/eval): full depth
  base_height.deep_squat_range = (SQUAT_FLOOR, 0.65)
  # Same penalty as joint_pos_limits (-10), but hip pitch, knee and ankle pitch
  # may use 97% of their range; every other joint keeps the 90% soft limit.
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=joint_pos_limits_widened, weight=cfg.rewards["joint_pos_limits"].weight,
    params={"factor": 0.97},
  )

  # The new task uses identifiable terrain columns even in play. Membership is
  # checked at each foot point's actual xy, including crossings between tiles.
  proportions = {
    "flat": 0.15, "soft_floor": 0.60,
    "hf_pyramid_slope": 0.05, "hf_pyramid_slope_inv": 0.05,
    "random_rough": 0.05, "wave_terrain": 0.10,
  }
  generator = cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = replace(
    generator, curriculum=True, num_cols=20,
    sub_terrains={
      name: replace(sub, proportion=proportions[name])
      for name, sub in generator.sub_terrains.items()
    },
  )
  cfg.actions["poultry_surface"] = PoultrySurfaceActionCfg(
    entity_name="robot", litter_enabled=litter_enabled,
    litter_depth_mode=("row" if play else "random") if litter_depth_mode is None else litter_depth_mode,
    litter_depth_range=litter_depth_range,
  )
  # V1 saved rollouts: sole apex ~1 cm, ankle stays near default, and a level
  # 10 cm swing competes with ~0.23/s posture reward. These tolerances reduce
  # that estimated cost to ~0.04/s while retaining the balance-joint settings.
  cfg.rewards["pose"].params["std_walking"].update({
    r".*hip_pitch.*": 0.8, r".*knee.*": 0.8, r".*ankle_pitch.*": 0.4,
  })
  # Lowest sole above the stance floor; contact-independent travel cost. The
  # old world-z/midfoot foot_clearance is removed here (a separate key keeps the
  # curves from reading as the old term) and untouched for every other task.
  cfg.rewards.pop("foot_clearance")
  # Boost 1 (2026-10-10, at 60k): -2.0 -> -3.0. Boost 2 (at 84k): lift flat at
  # ~7.3 cm from 70k to 85k against the 9 cm target; -3.0 -> -4.0.
  lift_target = {"above_litter": 0.05, "min_target": 0.09, "max_target": 0.11}
  cfg.rewards["poultry_clearance"] = RewardTermCfg(
    func=poultry_clearance, weight=-4.0, params=lift_target.copy(),
  )
  # Offline V1 rates at d=4 cm: ~0.012-0.015 raw plow on bare flat/soft floor.
  # Weight -6 makes that ~0.075-0.09/s, above the relaxed posture cost and well
  # below individual tracking rewards (~1.4-1.95/s). Layer-relative cost is
  # larger for buried toes. Measured on v1 at 0.35-0.5 m/s: -0.08 to -0.12/s
  # on bare flat/soft floor, -0.18 to -0.25/s on the 4 cm litter row.
  cfg.rewards["poultry_plow"] = RewardTermCfg(func=poultry_plow, weight=-6.0)
  # At 3.33 landings/s, a fully missed target costs about 0.2/s. Unlike just
  # raising an L1 target, this supplies height pressure to an already-low gait.
  # Boost 1 (at 60k): -0.06 -> -0.12. Boost 2 (at 84k): -0.12 -> -0.20.
  cfg.rewards["poultry_swing_height"] = RewardTermCfg(
    func=poultry_swing_height, weight=-0.20, params=lift_target.copy(),
  )
  cfg.rewards["poultry_foot_level"] = RewardTermCfg(func=poultry_foot_level, weight=-1.0)
  # Raw outputs drifted past the deploy guard in all three 100k lines (66k,
  # 66k, 75k; worst joint ankle pitch). At raw 12 this costs ~20% of a step's
  # reward, at the 15 clip ~60%; below 8 it is zero. -0.05 was too weak: 85k
  # still reached 17.8 (guard trip, ankle pitch), because the 15 clip hid
  # everything above it from the penalty. Now -0.2, with the clip at 30.
  cfg.rewards["action_excess_l2"] = RewardTermCfg(
    func=action_excess_l2, weight=-0.2, params={"threshold": 8.0},
  )
  # Actions are clipped at 30 (not 15) so the penalty above sees the real
  # magnitude; the last-action input stays within the deploy guard's +-15 so a
  # rare outlier cannot swamp the frozen-normalizer inputs.
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["actions"] = replace(terms["actions"], clip=(-15.0, 15.0))
  for key, name in {
    "poultry_swing_apex_m": "apex",  # held last completed apex, episode mean
    "poultry_swing_clearance_m": "clearance",
    "poultry_plow_cost": "plow",
    "poultry_toe_low_travel_mps": "low_travel",
    "poultry_toe_contact_slide_mps": "contact_slide",
    "poultry_liftoff_toe_down_deg": "liftoff_toe_down",
    "poultry_drag_force_n": "drag_force",
    "poultry_layer_height_m": "layer",
    "poultry_falls": "falls",
  }.items():
    cfg.metrics[key] = MetricsTermCfg(func=poultry_metric, params={"name": name})
  if not legacy:
    _train_from_scratch(cfg, play)
  return cfg


def _train_from_scratch(cfg, play):
  """The changes of 2026-11-14 (see the module docstring), applied in place."""
  feet = lambda: SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))  # noqa: E731

  # (1) Gait period: the clock the actor sees, the gait reward, and the swing
  # schedule the foot-level term reads.
  # (8) The gait reward pays for swinging on schedule and nothing for standing
  # still under a walk command (see settle.gait_swing). A perfect gait earns
  # 1.0 per second; the shared feet_gait it replaces earned 0.5, of which a
  # robot that never stepped kept 0.28.
  gait = cfg.rewards["foot_gait"].params
  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=settle.gait_swing, weight=1.0,
    params={
      "sensor_name": gait["sensor_name"], "period": GAIT_PERIOD,
      "stance_fraction": gait["threshold"], "offset": tuple(gait["offset"]),
    },
  )
  cfg.actions["poultry_surface"].period = GAIT_PERIOD

  # (2) + (4) Stiff gains with every motor behind a command delay, and one
  # action unit = 0.25 rad on every policy joint.
  cfg.scene.entities = {
    "robot": get_g1_23dof_deploy_robot_cfg(G1_23DOF_POULTRY_PD_GAINS, DELAY_MAX_LAG)
  }
  cfg.actions["joint_pos"].scale = G1_23DOF_DEPLOY_ACTION_SCALE
  cfg.rewards["action_excess_l2"].params["threshold"] = ACTION_EXCESS_THRESHOLD
  # The lag is drawn once per episode for the whole robot. Leg gain ranges are
  # AGILE's; the arm range is wide because on the robot the arms are driven by
  # whatever the manipulation side sends.
  cfg.events["actuator_delay"] = EventTermCfg(
    func=dr.sync_actuator_delays, mode="reset",
    params={"lag_range": (2, 2) if play else (0, DELAY_MAX_LAG)},
  )
  if not play:
    cfg.events["leg_gains"] = EventTermCfg(
      func=dr.pd_gains, mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=LEG_ACTUATOR_GROUPS),
        "kp_range": (0.9, 1.1), "kd_range": (0.8, 1.2),
      },
    )
    cfg.events["arm_gains"] = EventTermCfg(
      func=dr.pd_gains, mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=ARM_ACTUATOR_GROUPS),
        "kp_range": (0.5, 1.5), "kd_range": (0.5, 1.5),
      },
    )

  # (3) Each actor term carries its last HISTORY frames, oldest first, giving
  # HISTORY x 71 inputs; the critic keeps the current frame only. The base
  # config shares one term object between the two groups, so split them first.
  cfg.observations["critic"].terms = copy.deepcopy(cfg.observations["critic"].terms)
  cfg.observations["actor"].history_length = HISTORY
  cfg.observations["actor"].flatten_history_dim = True

  # (9) Foot spacing: the 0.20-0.35 m band stays for standing, forward walking
  # and turning; a sideways command gets 0.14-0.48 m, because the narrow band
  # capped sidestepping at 0.23 m/s (see settle.feet_spacing).
  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=settle.feet_spacing, weight=cfg.rewards["feet_distance"].weight,
    params={
      "min_distance": cfg.rewards["feet_distance"].params["min_distance"],
      "max_distance": cfg.rewards["feet_distance"].params["max_distance"],
      "asset_cfg": feet(),
    },
  )

  # (5) Stop -> settle -> stand. ``twist_held`` is the velocity command with its
  # fall to zero delayed by SETTLE_TIME; every term that means "you are standing
  # now" reads it instead of the raw command. See mdp/settle.py.
  cfg.commands = {
    "twist": cfg.commands["twist"],
    settle.HELD: settle.HeldTwistCommandCfg(
      resampling_time_range=(1e9, 1e9), settle_time=SETTLE_TIME
    ),
    # After twist_held, which it reads: the squat target replaces the walking
    # height only once the robot has settled.
    "base_height": cfg.commands["base_height"],
  }
  cfg.commands["base_height"].velocity_command_name = settle.HELD
  for group in ("actor", "critic"):
    # The clock stays visible while settling; it reads zero once STANDING.
    terms = cfg.observations[group].terms
    terms["phase"] = replace(
      terms["phase"], params={"period": GAIT_PERIOD, "command_name": settle.HELD}
    )
  for name in (
    "pose", "idle_base_motion", "idle_position_anchor", "idle_feet_still",
    "com_support", "squat_com_centering",
  ):
    cfg.rewards[name].params["command_name"] = settle.HELD
  # Fore-aft stagger of the feet once standing; a squaring step is free during
  # the settle window. At -2.0 with a 2 cm deadband the 2k trial stopped
  # quickly but kept a 5-6 cm stagger (12 cm worst), so -5.0 and 1 cm: a 10 cm
  # stagger now costs 0.45/s for as long as the robot stands.
  cfg.rewards["stand_feet_square"] = RewardTermCfg(
    func=settle.stand_feet_square, weight=-5.0,
    params={"deadband": 0.01, "asset_cfg": feet()},
  )
  # (7) Stand upright. The 2k trial stood 9.7 deg forward with its hips back
  # (the 91.5k policy: 5.9 deg on the robot). 9.7 deg costs 0.29/s here against
  # 0.07/s in body_orientation_l2. Not applied to squats.
  # Free band 1.5 -> 0.5 deg at 5,500 iterations of the full run: the stance
  # had moved from +0.4 deg (4,000) to 2.0 deg BACKWARD, which the wider band
  # charged almost nothing for (0.017/s; now 0.052/s).
  # Weight -2.0 -> -5.0 at 19,000: with pushes, payloads and arm motion at
  # full strength the stance wandered between +2.4 and +4.2 deg forward
  # (16,000-19,000) although this was already its largest standing penalty
  # (0.13/s at 4.2 deg; now 0.32/s).
  cfg.rewards["stand_upright"] = RewardTermCfg(
    func=settle.stand_upright, weight=-5.0,
    params={
      "deadband": 0.009,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  if not play:
    # More walk-to-stop transitions than command resampling alone gives.
    cfg.events["random_stops"] = EventTermCfg(
      func=settle.random_stops, mode="interval", interval_range_s=(2.0, 4.0),
      params={"prob": 0.15},
    )

  # (6) One run from scratch. The squat floor's schedule ends at 0.22 m instead
  # of at 0.24 m with a 90k-92k ramp for the resumed run; litter depth and the
  # four foot-lift penalties start low and ramp in.
  base_height = cfg.commands["base_height"]
  if base_height.floor_stages:
    stages = tuple(k for k in base_height.floor_stages if k[0] < 90_000 * ITER)
    base_height.floor_stages = stages[:-1] + ((stages[-1][0], SQUAT_FLOOR),)
  if play:
    base_height.floor_curriculum_start = SQUAT_FLOOR  # pinned: full depth
  else:
    cfg.actions["poultry_surface"].litter_depth_range = (0.0, LITTER_DEPTH_STAGES[0][1])
    cfg.curriculum["litter_depth"] = CurriculumTermCfg(
      func=settle.litter_depth_schedule,
      params={"stages": tuple((i * ITER, d) for i, d in LITTER_DEPTH_STAGES)},
    )
    for name in LIFT_TERMS:
      final = cfg.rewards[name].weight
      cfg.rewards[name].weight = final * LIFT_WEIGHT_STAGES[0][1]
      cfg.curriculum[f"weight_{name}"] = CurriculumTermCfg(
        func=settle.reward_weight_schedule,
        params={
          "reward_name": name,
          "stages": tuple((i * ITER, f * final) for i, f in LIFT_WEIGHT_STAGES),
        },
      )

  cfg.metrics.update({
    "standing_fraction": MetricsTermCfg(func=settle.standing_fraction),
    "standing_tall_fraction": MetricsTermCfg(func=settle.standing_tall_fraction),
    "moving_fraction": MetricsTermCfg(func=settle.moving_fraction),
    "stand_pitch_deg": MetricsTermCfg(
      func=settle.stand_pitch_deg,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    "stand_knee_rad": MetricsTermCfg(
      func=settle.stand_knee_rad,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",))},
    ),
    "stand_stagger_m": MetricsTermCfg(func=settle.stand_stagger_m, params={"asset_cfg": feet()}),
    "stand_foot_air": MetricsTermCfg(
      func=settle.stand_foot_air, params={"sensor_name": "feet_ground_contact"}
    ),
    "steps_per_second": MetricsTermCfg(
      func=settle.steps_per_second, params={"sensor_name": "feet_ground_contact"}
    ),
    "action_abs_max": MetricsTermCfg(func=settle.action_abs_max),
  })


def poultry_surface_ppo_runner_cfg(legacy=False):
  cfg = unitree_g1_23dof_duet_ppo_runner_cfg(
    experiment_name="DUET_G1_23dof_poultry",
    reward_floor=ROUGH_REWARD_FLOOR,
    max_iterations=100_001 if legacy else 50_001,
  )
  # Only matters when a run is resumed; see rl_cfg.DuetPpoAlgorithmCfg. The
  # 91.5k fine-tunes froze the normalizers of a mature checkpoint. The
  # from-scratch run is resumed while its schedules are still moving (squat
  # depth until 10.5k), so its normalizers keep updating as they did before
  # the stop.
  cfg.algorithm.freeze_obs_normalizer_on_load = legacy
  # Healthy checkpoints never exceed |a| ~13 (60k-65.5k); from 66k rare states
  # produced |a| up to 1000, blowing up the physics and the value loss. 30
  # bounds that while leaving 15-30 visible to action_excess_l2 (the deploy
  # runaway guard trips above 15); the last-action input is clipped at 15.
  cfg.clip_actions = 30.0
  return cfg
