"""G1-23dof with deployment-grade PD gains.

Sim2real thesis (HomieDeploy / mjlab-homierl): train with the PD gains the
onboard low-level controller actually runs, so the simulated closed loop is
the deployed closed loop.

The first-principles gains in ``g1_23dof_constants`` (armature x natural
frequency, 10 Hz) come out far softer than any proven G1 deployment stack:
hip_pitch/hip_yaw kp 40.2 vs 100-150 everywhere else, ankle 28.5 vs 40, arms
14.3 vs 40+. Deploying that soft loop (deploy.yaml faithfully ships the
training gains) amplifies unmodeled friction/backlash and slows disturbance
rejection on hardware.

Gain table used here = the unitree_rl_gym G1 table this repo's own FixStand
deploy state already runs (deploy config.yaml):

  legs:      kp 100, 100, 100 (hips), 150 (knee), 40, 40 (ankles)
             kd   2,   2,   2,          4,         2,  2
  waist_yaw: kp 200 / kd 5
  arms:      kp  40 / kd 10 (matches the FixStand/IL arm convention)

Motor properties (armature, effort limits) remain physical and come from
``g1_23dof_constants``. The matching action scale is the unitree_rl_gym
uniform 0.25 (see G1_23DOF_DEPLOY_ACTION_SCALE).
"""

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from src.assets.robots.unitree_g1.g1_23dof_constants import (
  ACTUATOR_5020,
  ACTUATOR_7520_14,
  ACTUATOR_7520_22,
  ARMATURE_5020,
  ARMATURE_7520_14,
  ARMATURE_7520_22,
  FULL_COLLISION,
  HOME_KEYFRAME,
  get_spec,
)

# (kp [N·m/rad], kd [N·m·s/rad]) per joint pattern — deploy config.yaml table.
G1_23DOF_DEPLOY_PD_GAINS: dict[str, tuple[float, float]] = {
  ".*_hip_pitch_joint": (100.0, 2.0),
  ".*_hip_roll_joint": (100.0, 2.0),
  ".*_hip_yaw_joint": (100.0, 2.0),
  ".*_knee_joint": (150.0, 4.0),
  ".*_ankle_pitch_joint": (40.0, 2.0),
  ".*_ankle_roll_joint": (40.0, 2.0),
  "waist_yaw_joint": (200.0, 5.0),
  ".*_shoulder_pitch_joint": (40.0, 10.0),
  ".*_shoulder_roll_joint": (40.0, 10.0),
  ".*_shoulder_yaw_joint": (40.0, 10.0),
  ".*_elbow_joint": (40.0, 10.0),
  ".*_wrist_roll_joint": (40.0, 10.0),
}

# Gain table of the PoultrySurface task (2026-11-14): the leg and waist gains of
# NVIDIA's GR00T decoupled whole-body controller for the G1
# (decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml), which sits in
# the middle of what HOMIE, AMO, FALCON and AGILE use. Arms: the 91.5k policy
# ran them at the soft 14.3 / 0.9 (measured on the robot as kp 14, kd 0.77), an
# underdamped 1.3 Hz pendulum with a 0.6 kg hand; 40 / 3 is damping ratio ~0.4.
G1_23DOF_POULTRY_PD_GAINS: dict[str, tuple[float, float]] = {
  ".*_hip_pitch_joint": (150.0, 2.0),
  ".*_hip_roll_joint": (150.0, 2.0),
  ".*_hip_yaw_joint": (150.0, 2.0),
  ".*_knee_joint": (200.0, 4.0),
  ".*_ankle_pitch_joint": (40.0, 2.0),
  ".*_ankle_roll_joint": (40.0, 2.0),
  "waist_yaw_joint": (250.0, 5.0),
  ".*_shoulder_pitch_joint": (40.0, 3.0),
  ".*_shoulder_roll_joint": (40.0, 3.0),
  ".*_shoulder_yaw_joint": (40.0, 3.0),
  ".*_elbow_joint": (40.0, 3.0),
  ".*_wrist_roll_joint": (40.0, 3.0),
}

# unitree_rl_gym deployment convention: uniform action scale.
G1_23DOF_DEPLOY_ACTION_SCALE = 0.25

# Actuator groups, in this order. ``dr.pd_gains`` and the delay events address
# them by index: 0-4 are the policy's joints, 5 the externally driven arms.
_ACTUATOR_GROUPS: tuple[tuple[tuple[str, ...], float, float], ...] = (
  ((".*_hip_pitch_joint", ".*_hip_yaw_joint"), ARMATURE_7520_14, ACTUATOR_7520_14.effort_limit),
  ((".*_hip_roll_joint",), ARMATURE_7520_22, ACTUATOR_7520_22.effort_limit),
  ((".*_knee_joint",), ARMATURE_7520_22, ACTUATOR_7520_22.effort_limit),
  ((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), ARMATURE_5020 * 2, ACTUATOR_5020.effort_limit * 2),
  (("waist_yaw_joint",), ARMATURE_7520_14, ACTUATOR_7520_14.effort_limit),
  (
    (
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_shoulder_yaw_joint",
      ".*_elbow_joint",
      ".*_wrist_roll_joint",
    ),
    ARMATURE_5020,
    ACTUATOR_5020.effort_limit,
  ),
)
LEG_ACTUATOR_GROUPS = [0, 1, 2, 3, 4]
ARM_ACTUATOR_GROUPS = [5]


def _actuator(
  patterns: tuple[str, ...],
  armature: float,
  effort_limit: float,
  table: dict[str, tuple[float, float]],
) -> BuiltinPositionActuatorCfg:
  gains = {table[p] for p in patterns}
  if len(gains) != 1:
    raise ValueError(f"Actuator group {patterns} mixes different PD gains: {gains}.")
  kp, kd = gains.pop()
  return BuiltinPositionActuatorCfg(
    target_names_expr=patterns,
    stiffness=kp,
    damping=kd,
    effort_limit=effort_limit,
    armature=armature,
  )


def _articulation(
  table: dict[str, tuple[float, float]], delay_max_lag: int = 0
) -> EntityArticulationInfoCfg:
  actuators = [_actuator(p, arm, eff, table) for p, arm, eff in _ACTUATOR_GROUPS]
  if delay_max_lag > 0:
    # hold_prob 1: the lag never re-draws on its own. A reset event
    # (``dr.sync_actuator_delays``) sets one value for the whole robot.
    actuators = [
      DelayedActuatorCfg(
        base_cfg=a, delay_min_lag=0, delay_max_lag=delay_max_lag, delay_hold_prob=1.0
      )
      for a in actuators
    ]
  return EntityArticulationInfoCfg(
    actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9
  )


G1_23DOF_DEPLOY_ARTICULATION = _articulation(G1_23DOF_DEPLOY_PD_GAINS)


def get_g1_23dof_deploy_robot_cfg(
  gains: dict[str, tuple[float, float]] | None = None, delay_max_lag: int = 0
) -> EntityCfg:
  """G1-23dof with deployment-grade PD gains (fresh instance per call).

  Args:
    gains: (kp, kd) per joint pattern; ``None`` keeps the unitree_rl_gym table.
    delay_max_lag: when > 0, every motor command is delayed by 0..this many
      physics steps (5 ms each), one value per robot per episode.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=_articulation(gains or G1_23DOF_DEPLOY_PD_GAINS, delay_max_lag),
  )
