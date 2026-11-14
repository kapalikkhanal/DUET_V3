"""Procedural egg tray and egg models (MuJoCo specs, convex collision only).

The tray is a pulp-style carrier: a base plate, one tapered cup per egg made of
four inclined plates (an inverted truncated pyramid), a perimeter rim and two
side handles where the hands hold it. Each egg is its own free rigid body: a
capsule for collision (cheap and well supported by MuJoCo Warp) and an
ellipsoid for display. Eggs stand upright and rest on the four cup walls, not
on the cup floor, so they self-centre and can rattle, ride up a wall, or be
thrown out -- nothing glues them to the tray.

Frames: tray origin at the centre of the base plate's top face, x forward (away
from the robot), y left, z up. Egg body origin at the capsule centre.

Collision filtering. Tray and egg geoms use contact bit 2 only; the robot and
terrain use bit 1. Eggs therefore touch the tray and each other, but nothing
else: the tray is held by the weld constraints (see ``carry.py``), not by
hand-tray contacts, and a lost egg is detected from its pose (``mdp.state``)
rather than by letting it bounce on the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

EGG_CONTACT_BIT = 2


@dataclass(frozen=True)
class TrayCfg:
  """Geometry and mass of the tray and its eggs (SI units).

  ``rows`` run along x (away from the robot), ``cols`` along y (between the
  hands). 2 x 3 = 6 eggs is the default; 4 x 3 = 12 keeps the same width so the
  same carry pose fits.
  """

  rows: int = 2
  cols: int = 3
  pitch: float = 0.055
  """Cup centre spacing. Commercial egg trays use ~50-60 mm."""
  cup_depth: float = 0.030
  wall_angle_deg: float = 60.0
  """Cup wall inclination from horizontal."""
  ridge: float = 0.002
  """Half-gap between adjacent cups at the top."""
  plate_thickness: float = 0.008
  """Collision thickness of every plate. Thicker than real pulp (2-3 mm) on
  purpose: each plate grows outward from the cup surface, so the cup shape is
  unchanged, but a soft contact can no longer push an egg past a plate's
  mid-plane and through it (seen with 3 mm under 20-80 N impacts)."""
  margin: float = 0.006
  """Base plate beyond the outer cups."""
  handle_length: float = 0.022
  handle_height: float = 0.015
  """Handle centre above the tray origin; the grip point."""
  tray_mass: float = 0.22
  """Empty tray, kg (150-300 g is typical for a rigid carrier)."""
  tray_friction: float = 0.6

  egg_radius: float = 0.0215
  """Half the egg width (43 mm wide)."""
  egg_length: float = 0.058
  egg_mass: float = 0.060
  egg_friction: float = 0.5

  @property
  def num_eggs(self) -> int:
    return self.rows * self.cols

  @property
  def cup_half_width(self) -> float:
    return self.pitch / 2 - self.ridge

  @property
  def egg_half_len(self) -> float:
    """Half-length of the capsule's cylindrical part."""
    return max(self.egg_length / 2 - self.egg_radius, 0.0)

  @property
  def half_extent(self) -> tuple[float, float]:
    """Half size of the base plate (x, y)."""
    return (
      self.rows * self.pitch / 2 + self.margin,
      self.cols * self.pitch / 2 + self.margin,
    )

  @property
  def grip_half_width(self) -> float:
    """|y| of the handle tips, i.e. where the palms are."""
    return self.half_extent[1] + self.handle_length

  def compartment_xy(self) -> np.ndarray:
    """Cup centres in the tray frame, [N, 2], row-major (x outer, y inner)."""
    xs = (np.arange(self.rows) - (self.rows - 1) / 2) * self.pitch
    ys = (np.arange(self.cols) - (self.cols - 1) / 2) * self.pitch
    return np.array([(x, y) for x in xs for y in ys], dtype=np.float64)

  def egg_rest_z(self) -> float:
    """Height of an egg's body origin above the tray origin at rest.

    The lower hemisphere (radius r, centre height zc) touches all four walls.
    A wall passes through the cup's top edge (W, D) with inward normal
    (-sin a, cos a); distance from (0, zc) to it equal to r gives
    zc = D + (r - W sin a) / cos a.
    """
    a = math.radians(self.wall_angle_deg)
    W, D, r = self.cup_half_width, self.cup_depth, self.egg_radius
    zc = D + (r - W * math.sin(a)) / math.cos(a)
    # The contact point must lie on the plate and the egg must clear the floor.
    contact_z = zc - r * math.cos(a)
    contact_x = r * math.sin(a)
    bottom = W - D / math.tan(a)
    if not (0.0 < contact_z < D and bottom < contact_x < W and zc - r > 0.0):
      raise ValueError(f"Egg does not seat on the cup walls: {self}")
    return zc + self.egg_half_len

  def egg_inertia(self) -> tuple[float, float, float]:
    """Solid ellipsoid with the egg's length and width (body z = long axis)."""
    a, b, m = self.egg_length / 2, self.egg_radius, self.egg_mass
    ixx = m * (a * a + b * b) / 5
    return (ixx, ixx, m * 2 * b * b / 5)


def _frame_quat(u: np.ndarray, e: np.ndarray) -> np.ndarray:
  """Quaternion of the frame with x = u, y = e, z = u x e."""
  mat = np.stack([u, e, np.cross(u, e)], axis=1)
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, mat.flatten())
  return quat


# Egg-tray contact: timeconst of two physics steps (the stiffest MuJoCo
# recommends at dt = 5 ms) and a high impedance, so a 60 g egg sinks well under
# a millimetre at rest and impacts are not absorbed by deep penetration.
CONTACT_SOLREF = (0.01, 1.0)
CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)


def _set_contact(geom, friction: float) -> None:
  geom.contype = EGG_CONTACT_BIT
  geom.conaffinity = EGG_CONTACT_BIT
  geom.condim = 3
  geom.friction = (friction, 0.005, 0.0001)
  geom.solref = CONTACT_SOLREF
  geom.solimp = CONTACT_SOLIMP


def build_tray_spec(cfg: TrayCfg, pos=(0.0, 0.0, 0.0)) -> mujoco.MjSpec:
  """Tray with a free joint, an IMU site and accelerometer/gyro sensors.

  ``pos`` is only the compiled model's ``qpos0`` placement; resets move it.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="tray", pos=pos)
  body.add_freejoint(name="tray_free")

  hx, hy = cfg.half_extent
  t = cfg.plate_thickness
  a = math.radians(cfg.wall_angle_deg)
  W, D = cfg.cup_half_width, cfg.cup_depth
  slope_len = D / math.sin(a)
  pulp = (0.78, 0.74, 0.66, 1.0)
  rim_rgba = (0.70, 0.66, 0.58, 1.0)

  def box(name, pos, half, quat=(1, 0, 0, 0), rgba=pulp):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat)
    g.rgba = rgba
    g.density = 0.0  # mass comes from the explicit inertial below
    _set_contact(g, cfg.tray_friction)
    return g

  box("base", (0, 0, -t / 2), (hx, hy, t / 2))
  # Four inclined plates per cup. Each plate's inner face lies on the plane
  # through the cup's top edge; the box centre sits half a thickness behind it.
  sides = (
    (np.array([math.cos(a), 0, math.sin(a)]), np.array([0.0, 1, 0])),   # +x wall
    (np.array([-math.cos(a), 0, math.sin(a)]), np.array([0.0, -1, 0])),  # -x wall
    (np.array([0, math.cos(a), math.sin(a)]), np.array([-1.0, 0, 0])),   # +y wall
    (np.array([0, -math.cos(a), math.sin(a)]), np.array([1.0, 0, 0])),   # -y wall
  )
  for k, (cx, cy) in enumerate(cfg.compartment_xy()):
    for s, (u, e) in enumerate(sides):
      normal = np.cross(u, e)
      outward = np.array([u[0], u[1], 0.0]) / math.cos(a)  # unit, horizontal
      mid = np.array([cx, cy, 0.0]) + outward * (W - 0.5 * D / math.tan(a)) + np.array([0, 0, D / 2])
      centre = mid - normal * t / 2
      box(f"cup{k}_wall{s}", centre, (slope_len / 2, W, t / 2), _frame_quat(u, e))
  # Perimeter rim to cup height, and the handles at the grip points.
  for name, pos, half in (
    ("rim_px", (hx - t / 2, 0, D / 2), (t / 2, hy, D / 2)),
    ("rim_nx", (-hx + t / 2, 0, D / 2), (t / 2, hy, D / 2)),
    ("rim_py", (0, hy - t / 2, D / 2), (hx, t / 2, D / 2)),
    ("rim_ny", (0, -hy + t / 2, D / 2), (hx, t / 2, D / 2)),
  ):
    box(name, pos, half, rgba=rim_rgba)
  hl = cfg.handle_length
  for side, sign in (("left", 1.0), ("right", -1.0)):
    box(
      f"handle_{side}",
      (0, sign * (hy + hl / 2), cfg.handle_height),
      (min(hx, 0.04), hl / 2, 0.006),
      rgba=rim_rgba,
    )

  body.explicitinertial = True
  body.mass = cfg.tray_mass
  body.ipos = (0, 0, D / 3)
  body.iquat = (1.0, 0.0, 0.0, 0.0)
  lx, ly, lz = 2 * hx, 2 * hy, D
  m = cfg.tray_mass
  body.inertia = (m * (ly**2 + lz**2) / 12, m * (lx**2 + lz**2) / 12, m * (lx**2 + ly**2) / 12)

  body.add_site(name="imu", pos=(0, 0, 0), size=(0.005, 0.005, 0.005))
  spec.add_sensor(
    name="tray_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  spec.add_sensor(
    name="tray_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  return spec


def build_egg_spec(cfg: TrayCfg, pos=(0.0, 0.0, 0.0)) -> mujoco.MjSpec:
  """One egg: free body, capsule collision, ellipsoid visual."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="egg", pos=pos)
  body.add_freejoint(name="egg_free")
  col = body.add_geom(
    name="egg_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=(cfg.egg_radius, cfg.egg_half_len, 0),
  )
  col.density = 0.0
  col.group = 3
  col.rgba = (0.9, 0.85, 0.75, 0.3)
  _set_contact(col, cfg.egg_friction)
  vis = body.add_geom(
    name="egg_visual", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    size=(cfg.egg_radius, cfg.egg_radius, cfg.egg_length / 2),
  )
  vis.contype = 0
  vis.conaffinity = 0
  vis.density = 0.0
  vis.group = 2
  vis.rgba = (0.93, 0.86, 0.74, 1.0)
  body.explicitinertial = True
  body.mass = cfg.egg_mass
  # Set explicitly: with explicitinertial and add_body(pos=...), MjSpec leaves
  # ipos equal to the body offset, i.e. the centre of mass metres away.
  body.ipos = (0.0, 0.0, 0.0)
  body.iquat = (1.0, 0.0, 0.0, 0.0)
  body.inertia = cfg.egg_inertia()
  return spec
