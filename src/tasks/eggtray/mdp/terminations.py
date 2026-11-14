"""Terminations: an egg out of its cup, or the tray tipped past recovery."""

from __future__ import annotations

import math

import torch

from src.tasks.eggtray.mdp.state import state


def egg_out_of_cup(env) -> torch.Tensor:
  return state(env).sample()["lost"].any(-1)


def tray_tipped(env, limit_deg: float = 40.0) -> torch.Tensor:
  return state(env).sample()["tilt"] > math.radians(limit_deg)
