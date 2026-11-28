"""Four level-bed substrate tiles. The mesh and the sole query share these grids."""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

TILE_NAMES = ("rigid", "compacted", "loose", "mixed")
# Support elevation, nominal loose layer, friction and positive contact time constant.
MATERIALS = ((0., 0., .55, .020), (.012, .006, .50, .024), (0., .030, .45, .030))
LAYOUT = np.array([[[i, i]] * 4 for i in range(3)] +
                  [[[0, 1], [1, 2], [2, 0], [0, 2]]])


@dataclass(kw_only=True)
class LiftTile(SubTerrainCfg):
  kind: int

  def function(self, difficulty, spec, rng):
    grid = LAYOUT[self.kind] if self.kind == 3 else LAYOUT[self.kind, :1, :1]
    dx, dy = np.array(self.size) / grid.shape
    body = spec.body("terrain")
    geometries = []
    for i, j in np.ndindex(grid.shape):
      bed, _, friction, tau = MATERIALS[grid[i, j]]
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx / 2, dy / 2, .1),
                           pos=((i + .5) * dx, (j + .5) * dy, bed - .1))
      geom.priority = 1  # combine with the randomized priority-1 foot capsules
      geom.friction = (friction, .005, .0001)
      geom.solref = (tau, 1.)
      geometries.append(TerrainGeometry(geom=geom, color=(.52, .43, .32, 1.)))
    # At the central seam, spawn high enough for both feet.
    origin = np.array([self.size[0] / 2, self.size[1] / 2,
                       max(MATERIALS[m][0] for m in grid.flat)])
    return TerrainOutput(origin=origin, geometries=geometries)


def material_at(points, terrain, layout):
  """Query absolute xy, including mixed-tile seams and crossings out of the spawn tile."""
  u = points[..., 0] / terrain.size[0] + terrain.num_rows / 2
  v = points[..., 1] / terrain.size[1] + terrain.num_cols / 2
  valid = (u >= 0) & (u < terrain.num_rows) & (v >= 0) & (v < terrain.num_cols)
  col = v.floor().long().clamp(0, terrain.num_cols - 1)
  i = ((u - u.floor()) * layout.shape[1]).long().clamp_max(layout.shape[1] - 1)
  j = ((v - v.floor()) * layout.shape[2]).long().clamp_max(layout.shape[2] - 1)
  return torch.where(valid, layout[col, i, j], 0), valid
