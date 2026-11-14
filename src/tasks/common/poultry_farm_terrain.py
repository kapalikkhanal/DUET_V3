"""The generated broiler-house litter floor as one mjlab sub-terrain.

Moved verbatim from ``scripts/poultry_floor/eval_poultry_floor.py`` so tasks
under ``src/`` (the egg-tray task) can use the same floor without copying it;
the evaluation script imports it from here. Build the assets first with
``scripts/poultry_floor/make_poultry_floor.py``.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from src.tasks.common.terrains import SoftFlatTerrainCfg

ASSETS = os.path.expanduser("~/Mujoco_kapalikkhanal/unitree_robots/g1/meshes/poultry_floor")
_POULTRY_FLOOR_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "poultry_floor"
)


@dataclass(kw_only=True)
class PoultryFarmTerrainCfg(SoftFlatTerrainCfg):
  """The generated litter floor as one mjlab sub-terrain.

  Subclasses SoftFlatTerrainCfg so the PoultrySurface action treats the whole
  tile as a litter (soft) tile: drag layer and litter metrics apply.
  """
  asset_dir: str = ASSETS
  contact: str = "floor"     # "floor" (as generated: preset), "training" (softest trained floor), "rigid"
  pans: bool = True
  flat: bool = False         # replace the litter relief with a flat surface (ablation)
  # MuJoCo Warp drops contacts above 50 per geom pair; the 19 cm foot capsules overflow
  # that on 4 cm cells. Training heightfields used 10 cm cells, so collide on a coarser
  # block-averaged grid (factor k of the generated resolution).
  coarsen: int = 2

  def function(self, difficulty, spec, rng):
    meta = json.load(open(os.path.join(self.asset_dir, "floor_meta.json")))
    p = meta["params"]
    body = spec.body("terrain")
    L, W, res = p["length"], p["width"], p["resolution"]
    geoms = []

    def hfield(h, cx, cy, half_y, solref, solimp, friction, rgba):
      k = max(1, self.coarsen)
      if k > 1:  # block average, keeping the outer edge so the extent is unchanged
        ny, nx = (h.shape[0] - 1) // k * k + 1, (h.shape[1] - 1) // k * k + 1
        h = h[:ny, :nx][::k, ::k] * 0.5 + 0.5 * np.pad(h[:ny, :nx], ((0, k), (0, k)), mode="edge")[
          k // 2::k, k // 2::k][: (ny - 1) // k + 1, : (nx - 1) // k + 1]
      lo, hi = float(h.min()), float(h.max())
      span = max(hi - lo, 1e-4)
      hf = spec.add_hfield(name=f"poultry_{len(geoms)}", size=[L / 2, half_y, span, 0.05],
                           nrow=h.shape[0], ncol=h.shape[1],
                           userdata=((h - lo) / span).flatten().astype(np.float32).tolist())
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hf.name, pos=[cx, cy, lo])
      g.priority = 2  # above the G1 foot geoms (priority 1), as the training soft floor
      g.solref = tuple(solref); g.solimp = tuple(solimp); g.friction = (friction, 0.005, 0.0001)
      geoms.append(TerrainGeometry(geom=g, hfield=hf, color=rgba))

    litter = np.load(os.path.join(self.asset_dir, "litter.npy"))
    if self.flat:
      litter = np.zeros_like(litter); litter[0, 0] = 1e-3  # hfield needs a non-zero range
    solref, solimp = p["friable_solref"], p["friable_solimp"]
    if self.contact == "training":  # SoftFlatTerrainCfg at difficulty 1 (softest trained floor)
      solref, solimp = (0.05, 2.0), (0.6, 0.9, 0.02, 0.5, 2.0)
    elif self.contact in ("built_up", "fresh_deep"):  # floor presets (poultry_floor.PRESETS)
      if _POULTRY_FLOOR_DIR not in sys.path:
        sys.path.insert(0, _POULTRY_FLOOR_DIR)
      import poultry_floor as pf
      solref, solimp = pf.PRESETS[self.contact]["friable_solref"], pf.PRESETS[self.contact]["friable_solimp"]
    elif self.contact == "rigid":  # hard floor (protocol baseline)
      solref, solimp = (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)
    hfield(litter, L / 2, W / 2, W / 2, solref, solimp, p["friable_friction"], (0.24, 0.18, 0.12, 1))
    cake = np.load(os.path.join(self.asset_dir, "cake_strip.npy"))
    ys = -W / 2 + np.arange(litter.shape[0]) * res
    rows = np.where(np.abs(ys - p["drinker_y"]) <= p["cake_width"] / 2 + 0.1)[0]
    cy = (ys[rows[0]] + ys[rows[-1]]) / 2 + W / 2
    hfield(cake, L / 2, cy, (rows[-1] - rows[0]) * res / 2, p["cake_solref"], p["cake_solimp"], p["cake_friction"], (0.14, 0.11, 0.08, 1))
    for (fx, fy) in (meta["feeders"] if self.pans else []):
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p["pan_radius"], p["pan_height"] / 2, 0],
                        pos=[fx - p["x0"], fy + W / 2, p["pan_height"] / 2 - 0.01])
      geoms.append(TerrainGeometry(geom=g, color=(0.8, 0.12, 0.1, 1)))
    # Spawn origin = floor point (0, 0) in tile coordinates.
    return TerrainOutput(origin=np.array([-p["x0"], W / 2, 0.0]), geometries=geoms)
