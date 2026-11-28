"""Litter substrate tiles. The compiled floor and the sole query share LAYOUT.

The tiles replace the 47k task's flat and soft_floor columns. Its heightfield
columns (slopes, rough, waves) stay in the same generator; on those the sole
query reports no valid support, as in the 47k task, and the lift terms rest.
"""
from dataclasses import dataclass, replace

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

TILE_NAMES = ("rigid", "compacted", "loose", "mixed", "split")
RIGID, COMPACTED, LOOSE = 0, 1, 2
SPLIT = 4
# Load-bearing bed elevation and nominal loose layer per material, metres.
# The loose layer is drawn per episode by the surface term (litter curriculum),
# so its entry here is only the nominal value.
MATERIALS = ((0., 0.), (.012, .006), (0., .030))
# Material per cell, indexed [tile kind, x cell, y cell]. Uniform tiles repeat
# one material; the mixed tile has 2 m x 4 m patches of all three. The
# split-level tile is rigid; its strip heights come from split_heights().
LAYOUT = np.array([[[i, i]] * 4 for i in range(3)] +
                  [[[0, 1], [1, 2], [2, 0], [0, 2]]] +
                  [[[0, 0]] * 4])


def _lerp(bounds, t):
  return bounds[0] + (bounds[1] - bounds[0]) * t


@dataclass(kw_only=True)
class LiftTile(SubTerrainCfg):
  kind: int
  # Litter contact as in SoftFlatTerrainCfg (the 47k soft_floor): each range
  # is (hard, soft) and scales with the row difficulty, so the terrain-level
  # curriculum also moves the robot onto softer, sinking litter. The rigid
  # material keeps default hard contact and priority 0, so the randomized foot
  # friction (0.3-1.6) applies there; litter cells use their own friction.
  timeconst_range: tuple[float, float] = (0.02, 0.05)
  dampratio_range: tuple[float, float] = (1.0, 2.0)
  dmin_range: tuple[float, float] = (0.9, 0.6)
  dmax_range: tuple[float, float] = (0.95, 0.9)
  width_range: tuple[float, float] = (0.001, 0.02)
  friction_range: tuple[float, float] = (0.3, 1.0)
  priority: int = 2

  def function(self, difficulty, spec, rng):
    t = float(np.clip(difficulty, 0., 1.))
    grid = LAYOUT[self.kind] if self.kind == 3 else LAYOUT[self.kind, :1, :1]
    dx, dy = np.array(self.size) / grid.shape
    body = spec.body("terrain")
    friction = float(rng.uniform(*self.friction_range))
    shade = .75 - .35 * t
    geometries = []
    for i, j in np.ndindex(grid.shape):
      material = int(grid[i, j])
      bed = MATERIALS[material][0]
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx / 2, dy / 2, .1),
                           pos=((i + .5) * dx, (j + .5) * dy, bed - .1))
      if material == RIGID:
        color = (.55, .55, .55, 1.)
      else:
        geom.priority = self.priority
        geom.solref = (_lerp(self.timeconst_range, t), _lerp(self.dampratio_range, t))
        geom.solimp = (_lerp(self.dmin_range, t), _lerp(self.dmax_range, t),
                       _lerp(self.width_range, t), .5, 2.)
        geom.friction = (friction, .005, .0001)
        tone = shade * (1.1 if material == COMPACTED else .9)
        color = (tone, .8 * tone, .55 * tone, 1.)
      geometries.append(TerrainGeometry(geom=geom, color=color))
    # At the mixed tile's central seam, spawn high enough for both feet.
    origin = np.array([self.size[0] / 2, self.size[1] / 2,
                       max(MATERIALS[m][0] for m in grid.flat)])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class SplitTile(LiftTile):
  """Split-level path: alternating low and raised rigid strips along x.

  Seams run along x every ``strip_width``, so a robot walking along a seam has
  one foot on each level, and walking or sidestepping across the strips steps
  up and down. With random headings each foot is uphill about half the time.
  The step height follows the terrain-level row (the curriculum): 2 cm at
  row 0, quadratic to 7 cm at the last row, so the starting rows 0-5 stay at
  2-3.5 cm. ``fixed_height`` pins it (evaluation).
  """

  kind: int = SPLIT
  height_range: tuple[float, float] = (.02, .07)
  levels: int = 10  # generator rows; row r is difficulty in [r/levels, (r+1)/levels)
  strip_width: float = .6
  fixed_height: float | None = None
  # Evaluation: spawn on a seam with the raised strip on +y (+1) or -y (-1).
  seam_side: int = 0
  # Training spawn points ("spawn" flat patches): most on seams, so robots
  # start, and often stand, with one foot up; the rest on strip centres.
  seam_patches: int = 7
  strip_patches: int = 3

  def height(self, row):
    if self.fixed_height is not None:
      return self.fixed_height
    f = row / max(self.levels - 1, 1)
    return _lerp(self.height_range, f * f)

  def strips(self):
    n = max(int(round(self.size[1] / self.strip_width)), 2)
    return n, self.size[1] / n

  def function(self, difficulty, spec, rng):
    height = self.height(min(int(difficulty * self.levels), self.levels - 1))
    n, width = self.strips()
    body = spec.body("terrain")
    geometries = []
    for k in range(n):
      top = height if k % 2 else 0.
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.size[0] / 2, width / 2, .1),
                           pos=(self.size[0] / 2, (k + .5) * width, top - .1))
      tone = .62 if k % 2 else .5
      geometries.append(TerrainGeometry(geom=geom, color=(tone, tone, .58, 1.)))
    low = n // 2 - (n // 2) % 2  # a low strip near the centre
    if self.seam_side == 0:
      origin = (self.size[0] / 2, (low + .5) * width, 0.)
    else:  # seam between this low strip and the raised one on the chosen side
      origin = (self.size[0] / 2, (low + (self.seam_side > 0)) * width, height)
    # Seams k|k+1 for consecutive k alternate which side is raised. The root is
    # placed at the raised level; the lower foot settles onto its strip.
    x = rng.uniform(.3, .7, self.seam_patches + self.strip_patches) * self.size[0]
    seams = [((low - 2 + k) + 1) * width for k in range(self.seam_patches)]
    centres = [(low - 1 + k + .5) * width for k in range(self.strip_patches)]
    patches = [(x[i], y, height) for i, y in enumerate(seams)]
    patches += [(x[self.seam_patches + i], y, height if (low - 1 + i) % 2 else 0.)
                for i, y in enumerate(centres)]
    return TerrainOutput(origin=np.array(origin), geometries=geometries,
                         flat_patches={"spawn": np.array(patches)})


def split_heights(generator):
  """Raised-strip height and strip width per (row, column); zero off split columns."""
  heights = np.zeros((generator.num_rows, generator.num_cols), dtype=np.float32)
  widths = np.ones(generator.num_cols, dtype=np.float32)
  subs = list(generator.sub_terrains.values())
  cumulative = np.cumsum([s.proportion for s in subs]) / sum(s.proportion for s in subs)
  for col in range(generator.num_cols):
    sub = subs[int(np.min(np.where(col / generator.num_cols + .001 < cumulative)[0]))]
    if isinstance(sub, SplitTile):
      tile = sub if tuple(sub.size) == tuple(generator.size) else replace(sub, size=generator.size)
      if tile.fixed_height is None and tile.levels != generator.num_rows:
        raise ValueError("SplitTile.levels must equal the generator's num_rows")
      heights[:, col] = [tile.height(r) for r in range(generator.num_rows)]
      widths[col] = tile.strips()[1]
  return heights, widths


def split_offset(points, terrain, heights, widths):
  """Raised-strip elevation under absolute xy (0 off split tiles and outside the grid)."""
  u = points[..., 0] / terrain.size[0] + terrain.num_rows / 2
  v = points[..., 1] / terrain.size[1] + terrain.num_cols / 2
  inside = (u >= 0) & (u < terrain.num_rows) & (v >= 0) & (v < terrain.num_cols)
  row = u.floor().long().clamp(0, terrain.num_rows - 1)
  col = v.floor().long().clamp(0, terrain.num_cols - 1)
  strip = ((v - v.floor()) * terrain.size[1] / widths[col]).floor().long()
  return heights[row, col] * ((strip % 2) == 1) * inside


def column_kinds(generator):
  """Tile kind of every terrain column, -1 for non-litter columns.

  Same column assignment as mjlab's curriculum generator.
  """
  subs = list(generator.sub_terrains.values())
  cumulative = np.cumsum([s.proportion for s in subs]) / sum(s.proportion for s in subs)
  kinds = []
  for col in range(generator.num_cols):
    sub = subs[int(np.min(np.where(col / generator.num_cols + .001 < cumulative)[0]))]
    kinds.append(sub.kind if isinstance(sub, LiftTile) else -1)
  return kinds


def material_at(points, terrain, layout, col_kind):
  """Material and validity at absolute xy, including seams and tile crossings."""
  u = points[..., 0] / terrain.size[0] + terrain.num_rows / 2
  v = points[..., 1] / terrain.size[1] + terrain.num_cols / 2
  inside = (u >= 0) & (u < terrain.num_rows) & (v >= 0) & (v < terrain.num_cols)
  kind = col_kind[v.floor().long().clamp(0, terrain.num_cols - 1)]
  valid = inside & (kind >= 0)
  i = ((u - u.floor()) * layout.shape[1]).long().clamp(0, layout.shape[1] - 1)
  j = ((v - v.floor()) * layout.shape[2]).long().clamp(0, layout.shape[2] - 1)
  return torch.where(valid, layout[kind.clamp_min(0), i, j], 0), valid
