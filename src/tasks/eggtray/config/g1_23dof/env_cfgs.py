"""Egg-tray arm-stabiliser environment for the G1-23DOF.

Built on ``unitree_g1_23dof_duet_poultry_surface_env_cfg`` so the robot model,
PD gains, lower-body action scale, contact parameters, litter-drag layer and
command generators are DUET's own. Then:

* the 13 lower-body actions come from the frozen DUET checkpoint
  (``duet_lower_body``, zero action dims), which reads the ``duet`` group -- a
  copy of the PoultrySurface actor group, i.e. the deployed 71-D layout;
* the arm disturbance generator is replaced by ``arm_residual`` (10 dims, the
  only learned action);
* a tray with one free egg per cup is held by both hands through soft welds;
* actor/critic groups, rewards, terminations and metrics are the egg task's.

Terrains: ``flat`` (debugging) or ``poultry`` -- the generated broiler-house
floor from ``scripts/poultry_floor`` (``built_up`` contact preset by default,
``fresh_deep`` as the out-of-distribution test), with the PoultrySurface drag
layer of loose litter on top.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.duet.mdp as duet_mdp
import src.tasks.eggtray.mdp as mdp
from src.tasks.common.poultry_farm_terrain import ASSETS as DEFAULT_FLOOR_ASSETS
from src.tasks.common.poultry_farm_terrain import PoultryFarmTerrainCfg
from src.tasks.duet.config.g1_23dof.poultry_surface import (
  unitree_g1_23dof_duet_poultry_surface_env_cfg,
)
from src.tasks.eggtray.carry import (
  CarryPoseCfg,
  DEFAULT_CARRY_POSE,
  add_tray_welds,
  compute_carry_geometry,
)
from src.tasks.eggtray.frozen_duet import FrozenDuetActionCfg
from src.tasks.eggtray.tray_model import TrayCfg, build_egg_spec, build_tray_spec

REPO_ROOT = Path(__file__).resolve().parents[5]
REPO_FLOOR_ASSETS = REPO_ROOT / "logs" / "eggtray" / "poultry_floor" / "meshes" / "poultry_floor"

TRAY_6 = TrayCfg(rows=2, cols=3)
TRAY_12 = TrayCfg(rows=4, cols=3)

# Feed-forward on the arm PD targets (arm gains are ~14 N m/rad). Stage 1
# cancels gravity sag so the loaded arms settle at the carry pose (it alone
# leaves the tray 4.9 deg nose-down: DUET leans its torso under the load);
# stage 2 rotates the tray through the tray-twist map until it is level in the
# world while standing (1.0 deg left). Measured on flat ground, six eggs,
# 0.25 kg hand surplus, by ``scripts/eggtray/phase_a_verify.py --calibrate``.
# Re-run it if the carry pose, tray or hands change.
CARRY_FEEDFORWARD: tuple[tuple[str, float], ...] = (
  ("left_shoulder_pitch_joint", -0.1362),
  ("left_shoulder_roll_joint", 0.0141),
  ("left_shoulder_yaw_joint", 0.0205),
  ("left_elbow_joint", -0.2478),
  ("left_wrist_roll_joint", -0.0044),
  ("right_shoulder_pitch_joint", -0.1347),
  ("right_shoulder_roll_joint", -0.0141),
  ("right_shoulder_yaw_joint", -0.0224),
  ("right_elbow_joint", -0.2465),
  ("right_wrist_roll_joint", 0.0041),
)

CARRY = CarryPoseCfg(joint_pos=DEFAULT_CARRY_POSE, feedforward=CARRY_FEEDFORWARD)
# 12 eggs: the 4-row tray is gripped at its middle, moved 3 cm forward so its
# rear edge stays clear of the torso.
CARRY_12 = replace(CARRY, tray_offset=(0.03, 0.0, 0.0))

# Weld softness. timeconst = 2 physics steps (the stiffest MuJoCo recommends);
# measured static tray sag in the hands ~1.5 mm with six eggs.
WELD_SOLREF = (0.01, 1.0)

# Walking commands for carrying: inside DUET's trained envelope
# (deploy contract: x -0.6..1.0, y +-0.5, yaw +-1.0), moderated for eggs.
CARRY_TWIST = {"lin_vel_x": (-0.4, 0.6), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.6, 0.6)}
CARRY_HEIGHT = (0.70, 0.78)  # pelvis height; DUET's nominal stand is 0.78

ARM_NOISE = {"pos": 0.01, "vel": 0.5, "ang_vel": 0.2, "gravity": 0.05,
             "tray_gravity": 0.03, "tray_gyro": 0.1, "tray_acc": 0.5}


def floor_assets() -> str:
  """$DUET_POULTRY_FLOOR_ASSETS, else the repo build, else the documented default."""
  env = os.environ.get("DUET_POULTRY_FLOOR_ASSETS")
  if env:
    return env
  return str(REPO_FLOOR_ASSETS) if REPO_FLOOR_ASSETS.exists() else DEFAULT_FLOOR_ASSETS


def _floor_size(asset_dir: str) -> tuple[float, float]:
  meta = Path(asset_dir) / "floor_meta.json"
  if meta.exists():
    p = json.loads(meta.read_text())["params"]
    return float(p["length"]), float(p["width"])
  return 16.0, 6.0  # make_poultry_floor.py defaults; the build fails loudly if missing


def poultry_floor_terrain(cfg: ManagerBasedRlEnvCfg, contact: str = "built_up", pans: bool = True) -> None:
  """The generated broiler-house floor as the only tile (as eval_poultry_floor.py)."""
  assets = floor_assets()
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=_floor_size(assets), num_rows=1, num_cols=1, curriculum=True, border_width=2.0,
    sub_terrains={"poultry_farm": PoultryFarmTerrainCfg(
      proportion=1.0, asset_dir=assets, contact=contact, pans=pans, coarsen=2)},
  )
  cfg.scene.terrain.max_init_terrain_level = 0


def unitree_g1_23dof_eggtray_env_cfg(
  play: bool = False,
  terrain: str = "poultry",
  tray: TrayCfg = TRAY_6,
  carry: CarryPoseCfg = CARRY,
  floor_contact: str = "built_up",
  litter_depth_range: tuple[float, float] = (0.0, 0.06),
  duet_obs_noise: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """Egg-tray stabiliser task.

  Args:
    play: evaluation/visualisation settings (no obs noise, no pushes, eggs and
      tray tipping do not end the episode).
    terrain: ``flat`` or ``poultry``.
    tray: tray geometry; ``TRAY_6`` or ``TRAY_12``.
    carry: carry pose and grasp; replaceable by a GR00T-produced pose.
    floor_contact: litter contact preset for ``poultry`` (built_up, fresh_deep,
      training, rigid, floor).
    litter_depth_range: loose-litter drag layer depth, sampled per episode.
    duet_obs_noise: corrupt DUET's own 71-D inputs as in its training
      (default: on in training, off in play).
  """
  # legacy=True: the frozen legs are the 91.5k checkpoint, which needs the task
  # it was trained in (the PoultrySurface task itself was rebuilt 2026-11-14).
  cfg = unitree_g1_23dof_duet_poultry_surface_env_cfg(play=play, legacy=True)
  geo = compute_carry_geometry(carry, tray)
  n_eggs = tray.num_eggs

  # -- Scene -----------------------------------------------------------------
  # Spec positions only keep the bodies apart in the compiled model's qpos0,
  # from which MuJoCo Warp sizes its contact buffer; every reset places them.
  cfg.scene.entities["tray"] = EntityCfg(spec_fn=lambda: build_tray_spec(tray, pos=(0.0, 2.0, 1.0)))
  for i in range(n_eggs):
    cfg.scene.entities[f"egg_{i}"] = EntityCfg(
      spec_fn=lambda i=i: build_egg_spec(tray, pos=(0.1 * i, -2.0, 1.0)),
    )
  cfg.scene.spec_fn = lambda spec: add_tray_welds(spec, geo, solref=WELD_SOLREF)
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")

  if terrain == "flat":
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.sim.nconmax, cfg.sim.njmax = 128, 600
    spawn = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)}
  elif terrain == "poultry":
    poultry_floor_terrain(cfg, floor_contact)
    # Heightfield foot contacts plus ~4 per seated egg. Measured with fixed
    # arms: <= 57 contacts and 312 constraint rows per world.
    cfg.sim.nconmax, cfg.sim.njmax = 192, 900
    # Floor coordinates (as eval_poultry_floor.py): aisle and cake strip, clear of the pans.
    spawn = {"x": (-3.0, 7.0), "y": (-0.8, 2.4), "z": (0.0, 0.02), "yaw": (-math.pi, math.pi)}
  else:
    raise ValueError(f"terrain must be flat|poultry, got {terrain!r}")

  # -- Actions ---------------------------------------------------------------
  lower = cfg.actions["joint_pos"]
  cfg.actions = {
    "duet_lower_body": FrozenDuetActionCfg(
      entity_name="robot", actuator_names=tuple(lower.actuator_names), scale=dict(lower.scale),
    ),
    "arm_residual": mdp.ArmResidualActionCfg(entity_name="robot", carry=carry, tray=tray),
    "poultry_surface": replace(
      cfg.actions["poultry_surface"], litter_depth_mode="random", litter_depth_range=litter_depth_range,
    ),
    "eggtray_state": mdp.EggTrayStateCfg(entity_name="robot", tray=tray, carry=carry),
  }

  # -- Observations ----------------------------------------------------------
  # DUET's own input, untouched except that its last-action term must read the
  # frozen DUET term (the env-level action vector is now the arm residual).
  duet = copy.deepcopy(cfg.observations["actor"])
  duet.terms["actions"] = replace(duet.terms["actions"], params={"action_name": "duet_lower_body"})
  duet.enable_corruption = (not play) if duet_obs_noise is None else duet_obs_noise

  def n(key):
    return Unoise(n_min=-ARM_NOISE[key], n_max=ARM_NOISE[key])

  actor_terms = {
    "arm_joint_pos": ObservationTermCfg(func=mdp.arm_joint_pos, noise=n("pos")),
    "arm_joint_vel": ObservationTermCfg(func=mdp.arm_joint_vel, noise=n("vel"), scale=0.1),
    "base_ang_vel": ObservationTermCfg(
      func=duet_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=n("ang_vel"),
    ),
    "projected_gravity": ObservationTermCfg(func=duet_mdp.projected_gravity, noise=n("gravity")),
    "command": ObservationTermCfg(func=duet_mdp.generated_commands, params={"command_name": "twist"}),
    "height_command": ObservationTermCfg(
      func=duet_mdp.generated_commands, params={"command_name": "base_height"},
    ),
    "phase": ObservationTermCfg(func=duet_mdp.phase, params={"period": 0.6, "command_name": "twist"}),
    "tray_gravity": ObservationTermCfg(func=mdp.tray_gravity, noise=n("tray_gravity")),
    "tray_gyro": ObservationTermCfg(
      func=duet_mdp.builtin_sensor, params={"sensor_name": "tray/tray_gyro"}, noise=n("tray_gyro"),
    ),
    "tray_acc": ObservationTermCfg(
      func=duet_mdp.builtin_sensor, params={"sensor_name": "tray/tray_acc"}, noise=n("tray_acc"),
      scale=0.1,
    ),
    "last_action": ObservationTermCfg(
      func=duet_mdp.last_action, params={"action_name": "arm_residual"},
    ),
    "carry_reference": ObservationTermCfg(func=mdp.carry_reference_offset),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=duet_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "tray_lin_vel": ObservationTermCfg(func=mdp.tray_lin_vel_b),
    "egg_rel_pos": ObservationTermCfg(func=mdp.egg_rel_pos, scale=20.0),
    "egg_rel_vel": ObservationTermCfg(func=mdp.egg_rel_vel),
    "grasp_strain": ObservationTermCfg(func=mdp.grasp_strain, scale=100.0),
    "foot_contact": ObservationTermCfg(
      func=duet_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"},
    ),
    "duet_action": ObservationTermCfg(func=mdp.duet_action, clip=(-15.0, 15.0)),
    "litter_depth": ObservationTermCfg(func=mdp.litter_depth, scale=10.0),
  }
  cfg.observations = {
    "duet": duet,
    "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True,
                                 enable_corruption=not play, history_length=1),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True,
                                  enable_corruption=False, history_length=1),
  }

  # -- Commands --------------------------------------------------------------
  twist = cfg.commands["twist"]
  for axis, rng in CARRY_TWIST.items():
    setattr(twist.ranges, axis, rng)
  twist.rel_standing_envs = 0.2
  height = cfg.commands["base_height"]
  height.height_range = CARRY_HEIGHT
  height.floor_curriculum_start = CARRY_HEIGHT[0]
  height.floor_stages = None
  height.walk_min_height = CARRY_HEIGHT[0]
  height.nominal_env_fraction = 0.5
  height.deep_squat_fraction = 0.0

  # -- Events ----------------------------------------------------------------
  for key in ("push_robot", "randomize_terrain"):
    cfg.events.pop(key, None)
  cfg.events["reset_base"] = EventTermCfg(
    func=duet_mdp.reset_root_state_uniform, mode="reset",
    params={"pose_range": spawn, "velocity_range": {}},
  )
  # Hand mass beyond the sim rubber hand = the real Dex3 + camera surplus only;
  # the tray and eggs are simulated bodies, not a lumped payload.
  cfg.events["hand_payload"].params["ranges"] = (0.25, 0.25) if play else (0.2, 0.35)
  if play:
    cfg.events["torso_payload"].params["ranges"] = (0.0, 0.0)
  else:
    cfg.events["push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity, mode="interval", interval_range_s=(6.0, 10.0),
      params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.3, 0.3)}},
    )
    cfg.events["tray_mass"] = EventTermCfg(
      func=dr.body_mass, mode="reset",
      params={"asset_cfg": SceneEntityCfg("tray", body_names=("tray",)), "operation": "abs",
              "ranges": (0.15, 0.30)},
    )
    cfg.events["tray_com"] = EventTermCfg(
      func=dr.body_com_offset, mode="startup",
      params={"asset_cfg": SceneEntityCfg("tray", body_names=("tray",)), "operation": "add",
              "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.005, 0.005)}},
    )
    cfg.events["tray_friction"] = EventTermCfg(
      func=dr.geom_friction, mode="startup",
      params={"asset_cfg": SceneEntityCfg("tray", geom_names=(".*",)), "operation": "abs",
              "ranges": (0.35, 0.9), "shared_random": True},
    )
    for i in range(n_eggs):
      cfg.events[f"egg_{i}_mass"] = EventTermCfg(
        func=dr.body_mass, mode="reset",
        params={"asset_cfg": SceneEntityCfg(f"egg_{i}", body_names=("egg",)), "operation": "abs",
                "ranges": (0.050, 0.070)},
      )
  cfg.events["reset_carry"] = EventTermCfg(
    func=mdp.reset_carry_state, mode="reset", params={"tray": tray, "carry": carry},
  )

  # -- Rewards ---------------------------------------------------------------
  # Manager scales by dt (0.02 s); per-second magnitudes in comments.
  cfg.rewards = {
    # Primary objective, positive and bounded: 2.0/s at zero acceleration. std
    # 3 m/s^2: fixed arms average ~2.7 (flat) to ~2.9 (litter), where the
    # kernel still has slope.
    "tray_acc": RewardTermCfg(func=mdp.tray_acc_exp, weight=2.0, params={"std": 3.0}),
    # Tilt beyond 3 deg: 10 deg costs 0.3/s, 20 deg 1.8/s.
    "tray_tilt": RewardTermCfg(func=mdp.tray_tilt_excess, weight=-20.0, params={"deadband_deg": 3.0}),
    "tray_ang_vel": RewardTermCfg(func=mdp.tray_ang_vel_l2, weight=-0.2),
    # One-time -10 per egg out of its cup (the episode also ends in training).
    "egg_lost": RewardTermCfg(func=mdp.egg_lost, weight=-500.0),
    "egg_motion": RewardTermCfg(func=mdp.egg_motion_l2, weight=-2.0),
    # Net contact force above 1.5 N (2.5x the egg's weight); walking with
    # fixed arms reaches ~0.9 N at p99 and ~2.7 N at worst on flat ground.
    "egg_impact": RewardTermCfg(func=mdp.egg_impact, weight=-0.5, params={"threshold": 1.5}),
    "action_rate": RewardTermCfg(func=mdp.arm_action_rate_l2, weight=-0.02),
    "arm_joint_acc": RewardTermCfg(func=mdp.arm_joint_acc_l2, weight=-1e-6),
    "residual": RewardTermCfg(func=mdp.arm_residual_l2, weight=-0.1),
    "grasp_strain": RewardTermCfg(func=mdp.grasp_strain_l2, weight=-1000.0),
    "is_terminated": RewardTermCfg(func=duet_mdp.is_terminated, weight=-200.0),
  }

  # -- Terminations ----------------------------------------------------------
  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "fell_over": cfg.terminations["fell_over"],
  }
  if not play:
    cfg.terminations["egg_out_of_cup"] = TerminationTermCfg(func=mdp.egg_out_of_cup)
    cfg.terminations["tray_tipped"] = TerminationTermCfg(func=mdp.tray_tipped, params={"limit_deg": 40.0})

  # -- Metrics ---------------------------------------------------------------
  keep = ("lin_vel_track_error", "yaw_track_error", "poultry_falls")
  cfg.metrics = {k: v for k, v in cfg.metrics.items() if k in keep}
  for name in ("acc", "acc_imu", "jerk", "tilt_deg", "ang_vel", "eggs_kept", "egg_force_max",
               "egg_speed", "grasp_strain_mm", "residual_rad", "duet_extreme"):
    cfg.metrics[f"tray_{name}"] = MetricsTermCfg(func=mdp.eggtray_metric, params={"name": name})

  cfg.curriculum = {}
  cfg.episode_length_s = int(1e9) if play else 20.0
  cfg.viewer.body_name = "torso_link"
  cfg.viewer.distance = 2.2
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 150.0
  return cfg
