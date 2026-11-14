"""Carrying pose and the tray attachment derived from it.

Everything here is computed once, on the CPU, from forward kinematics of the
G1-23DOF at the carrying pose with the pelvis at the origin. The carry pose is
a plain joint dictionary so a pose produced later by GR00T can replace it.

Grasp model (``grasp``):

* ``two_hand`` (default). The tray hangs between the palms and each hand is
  attached by a soft weld (``eq_weld``) at its palm. Two welds on one 6-DOF body
  form a closed chain with 12 constraint rows; MuJoCo resolves the redundancy
  through its soft, regularised constraint solver, and both welds and the arm
  PD loops are compliant, so inconsistent arm targets produce a bounded
  internal "squeeze" rather than a solver fight. The squeeze is measured
  (weld stretch, ``mdp.state``) and penalised.
* ``right_hand``. One weld at the right palm; the tray is held by its right
  handle and extends to the robot's left. Kept for the payload study in the
  docs: a single 5-DOF arm cannot set all six DOF of the tray pose (the 23-DOF
  G1 has no wrist pitch or yaw).

G1 elbow convention: elbow = 0 is the L-shape (forearm at 90 deg to the upper
arm). With the upper arm hanging and the elbow near 0 the forearm is level, so
the default pose below is the "elbow flexed ~90 deg" carry with the palms at
about elbow height (z ~ 0.89 m when standing at 0.79 m pelvis height).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from src.tasks.eggtray.tray_model import TrayCfg

ARM_JOINTS = (
  r".*_shoulder_pitch_joint",
  r".*_shoulder_roll_joint",
  r".*_shoulder_yaw_joint",
  r".*_elbow_joint",
  r".*_wrist_roll_joint",
)
HAND_BODIES = {"left": "left_wrist_roll_rubber_hand", "right": "right_wrist_roll_rubber_hand"}
PALM_SITES = {"left": "left_palm", "right": "right_palm"}


def symmetric_arm_pose(pitch: float, roll: float, yaw: float, elbow: float, wrist: float = 0.0):
  """Left-side values; the right side mirrors roll / yaw / wrist (as env_cfgs._arm_pose)."""
  return (
    ("left_shoulder_pitch_joint", pitch), ("right_shoulder_pitch_joint", pitch),
    ("left_shoulder_roll_joint", roll), ("right_shoulder_roll_joint", -roll),
    ("left_shoulder_yaw_joint", yaw), ("right_shoulder_yaw_joint", -yaw),
    ("left_elbow_joint", elbow), ("right_elbow_joint", elbow),
    ("left_wrist_roll_joint", wrist), ("right_wrist_roll_joint", -wrist),
  )


# Upper arm ~6 deg forward of vertical, forearm level, hands pointing forward.
DEFAULT_CARRY_POSE = symmetric_arm_pose(pitch=-0.10, roll=0.0, yaw=0.0, elbow=0.10)


@dataclass(frozen=True)
class CarryPoseCfg:
  """Nominal carrying configuration (hashable so geometry can be cached)."""

  joint_pos: tuple[tuple[str, float], ...] = DEFAULT_CARRY_POSE
  grasp: str = "two_hand"
  tray_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Tray origin shift (pelvis frame) from where the grasp puts it."""
  feedforward: tuple[tuple[str, float], ...] = ()
  """Gravity pre-compensation added to the PD targets so the loaded arms
  settle AT ``joint_pos`` (arm gains are soft: ~14 N m/rad). Measured by
  ``scripts/eggtray/phase_a_verify.py --calibrate``."""

  def joint_dict(self) -> dict[str, float]:
    return dict(self.joint_pos)

  def feedforward_dict(self) -> dict[str, float]:
    return dict(self.feedforward)


@dataclass(frozen=True)
class CarryGeometry:
  arm_joint_names: tuple[str, ...]
  """Arm joints in robot (entity) order."""
  q_nominal: np.ndarray
  """[10] carry pose in ``arm_joint_names`` order."""
  tray_pos: np.ndarray
  tray_quat: np.ndarray
  """Tray pose in the pelvis frame (wxyz)."""
  hands: tuple[str, ...]
  """Hand body names (robot-local) holding the tray."""
  weld_relpose: dict[str, np.ndarray] = field(default_factory=dict)
  """hand -> [7] tray pose in the hand frame (MuJoCo weld ``relpose``)."""
  weld_anchor: dict[str, np.ndarray] = field(default_factory=dict)
  """hand -> [3] grip point in the tray frame (weld ``anchor``)."""
  tray_twist_to_joints: np.ndarray | None = None
  """[10, 6] least-squares map from a small tray twist (v, w) in the torso
  frame, about the tray origin, to arm joint offsets that realise it."""


def _robot_model() -> mujoco.MjModel:
  from src.assets.robots.unitree_g1.g1_23dof_constants import get_spec

  return get_spec().compile()


def _pose_mat(pos, quat) -> np.ndarray:
  T = np.eye(4)
  m = np.empty(9)
  mujoco.mju_quat2Mat(m, np.asarray(quat, dtype=np.float64))
  T[:3, :3] = m.reshape(3, 3)
  T[:3, 3] = pos
  return T


def _mat_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  q = np.empty(4)
  mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
  return T[:3, 3].copy(), q


@functools.lru_cache(maxsize=16)
def compute_carry_geometry(carry: CarryPoseCfg, tray: TrayCfg) -> CarryGeometry:
  import re

  model = _robot_model()
  data = mujoco.MjData(model)
  data.qpos[:] = 0.0
  data.qpos[3] = 1.0  # pelvis at the origin, upright
  names = [model.joint(j).name for j in range(model.njnt)]
  arm_names = tuple(n for n in names if any(re.fullmatch(p, n) for p in ARM_JOINTS))
  pose = carry.joint_dict()
  missing = set(arm_names) - set(pose)
  if missing:
    raise ValueError(f"carry pose is missing joints {sorted(missing)}")
  for n in arm_names:
    data.qpos[model.joint(n).qposadr[0]] = pose[n]
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)

  if carry.grasp == "two_hand":
    hands = ("left", "right")
    palms = np.stack([data.site(PALM_SITES[h]).xpos for h in hands])
    centre = palms.mean(0)
  elif carry.grasp == "right_hand":
    hands = ("right",)
    palm = data.site(PALM_SITES["right"]).xpos
    # Tray's right handle tip at the palm: the tray extends to the robot's left.
    centre = palm + np.array([0.0, tray.grip_half_width, 0.0])
  else:
    raise ValueError(f"grasp must be two_hand|right_hand, got {carry.grasp!r}")
  tray_pos = centre - np.array([0.0, 0.0, tray.handle_height]) + np.asarray(carry.tray_offset)
  tray_quat = np.array([1.0, 0.0, 0.0, 0.0])  # level, facing forward
  T_tray = _pose_mat(tray_pos, tray_quat)

  relpose, anchor = {}, {}
  hand_bodies = tuple(HAND_BODIES[h] for h in hands)
  for h, body in zip(hands, hand_bodies):
    T_hand = _pose_mat(data.body(body).xpos, data.body(body).xquat)
    pos, quat = _mat_pose(np.linalg.inv(T_hand) @ T_tray)
    relpose[body] = np.concatenate([pos, quat])
    palm_w = np.append(data.site(PALM_SITES[h]).xpos, 1.0)
    anchor[body] = (np.linalg.inv(T_tray) @ palm_w)[:3]

  # Least-squares tray-twist -> arm-joint map from the hand Jacobians. The torso
  # frame equals the world frame here (pelvis at identity, waist_yaw = 0).
  M = np.zeros((len(arm_names), 6))
  for h, body in zip(hands, hand_bodies):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, model.body(body).id)
    cols, rows = [], []
    for i, n in enumerate(arm_names):
      if n.startswith(h + "_"):
        cols.append(model.joint(n).dofadr[0])
        rows.append(i)
    J = np.vstack([jacp[:, cols], jacr[:, cols]])  # [6, 5]
    r = data.body(body).xpos - tray_pos
    # Hand twist for a tray twist (v, w): v_h = v + w x r, w_h = w.
    skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    A = np.block([[np.eye(3), -skew], [np.zeros((3, 3)), np.eye(3)]])
    damp = 1e-3
    J_pinv = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))
    M[rows] = J_pinv @ A

  return CarryGeometry(
    arm_joint_names=arm_names,
    q_nominal=np.array([pose[n] for n in arm_names]),
    tray_pos=tray_pos,
    tray_quat=tray_quat,
    hands=hand_bodies,
    weld_relpose=relpose,
    weld_anchor=anchor,
    tray_twist_to_joints=M,
  )


def add_tray_welds(
  spec: mujoco.MjSpec,
  geometry: CarryGeometry,
  solref: tuple[float, float] = (0.02, 1.0),
  solimp: tuple[float, ...] = (0.95, 0.99, 0.001, 0.5, 2.0),
  robot: str = "robot",
  tray: str = "tray",
) -> None:
  """Attach the tray to each grasping hand with a soft weld (scene ``spec_fn``)."""
  for body in geometry.hands:
    # Raw eq_data (what the XML compiler would produce from relpose+anchor):
    # [0:3] anchor in body2 (tray), [3:6] the same point in body1 (hand),
    # [6:10] body2 orientation in body1, [10] torquescale. The engine compares
    # body1 + R1 data[3:6] with body2 + R2 data[0:3].
    rel = geometry.weld_relpose[body]
    anchor = geometry.weld_anchor[body]
    data = np.zeros(11)
    data[0:3] = anchor
    data[3:6] = (_pose_mat(rel[:3], rel[3:]) @ np.append(anchor, 1.0))[:3]
    data[6:10] = rel[3:]
    data[10] = 1.0  # torquescale
    eq = spec.add_equality(
      name=f"grasp_{body}",
      type=mujoco.mjtEq.mjEQ_WELD,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      name1=f"{robot}/{body}",
      name2=f"{tray}/tray",
      data=data.tolist(),
    )
    eq.solref = solref
    eq.solimp = solimp


def quat_angle(q: np.ndarray) -> float:
  return 2.0 * math.acos(min(1.0, abs(float(q[0]))))
