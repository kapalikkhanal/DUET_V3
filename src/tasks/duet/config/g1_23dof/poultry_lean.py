"""PoultrySurface plus a commanded forward lean, for picking eggs off the floor.

The PoultrySurface task itself is untouched. This variant adds one command and
one policy input (72 values per frame instead of 71, so 6 x 72 = 432 inputs):
the torso pitch to hold while standing. See ``mdp/lean.py`` for why a lean and
not a deeper squat.

It is fine-tuned from a PoultrySurface checkpoint. ``scripts/poultry/add_lean_input.py``
widens that checkpoint's networks by the new input with zero weights, so the
policy starts out behaving exactly as before and only has to learn the lean.
"""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import src.tasks.duet.mdp as mdp
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import lean

from .poultry_surface import unitree_g1_23dof_duet_poultry_surface_env_cfg

# Largest lean, rad (72 deg). With the pelvis at 0.45-0.50 m this puts the
# shoulders low and ahead of the knees, so a straight arm reaches the floor.
LEAN_MAX = 1.25
# The largest lean drawn opens up over the first 2,000 iterations of the
# fine-tune, which starts from PoultrySurface iteration 26,500. (Iteration
# counts continue from the checkpoint.)
LEAN_STAGES = ((26_500, 0.35), (28_500, LEAN_MAX))
# Pelvis heights commanded together with a lean, m.
STOOP_HEIGHT = (0.42, 0.62)


def unitree_g1_23dof_duet_poultry_lean_env_cfg(play=False):
  cfg = unitree_g1_23dof_duet_poultry_surface_env_cfg(play=play)
  torso = lambda: SceneEntityCfg("robot", body_names=("torso_link",))  # noqa: E731
  feet = lambda: SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))  # noqa: E731

  # The command, after the terms it reads (twist_held, base_height).
  cfg.commands[lean.LEAN] = lean.LeanCommandCfg(
    resampling_time_range=(4.0, 10.0),
    max_lean_stages=((0, LEAN_MAX),) if play else tuple((i * ITER, v) for i, v in LEAN_STAGES),
    stoop_height_range=STOOP_HEIGHT,
  )
  # The new input: LAST in both groups, so an older network only needs extra
  # input columns appended.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["lean_command"] = ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": lean.LEAN}
    )

  # "Upright" becomes "at the commanded lean". Same weights and parameters.
  old = cfg.rewards["body_orientation_l2"]
  cfg.rewards["body_orientation_l2"] = RewardTermCfg(
    func=lean.orientation_lean_l2, weight=old.weight, params=dict(old.params)
  )
  old = cfg.rewards["stand_upright"]
  cfg.rewards["stand_upright"] = RewardTermCfg(
    func=lean.stand_upright_lean, weight=old.weight, params=dict(old.params)
  )
  old = cfg.rewards["idle_position_anchor"]
  cfg.rewards["idle_position_anchor"] = RewardTermCfg(
    func=lean.idle_position_anchor_lean, weight=old.weight, params=dict(old.params)
  )
  # A 70 deg lean is not a fall.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=lean.fell_over_lean,
    params={"limit_angle": math.radians(60.0), "asset_cfg": feet()},
  )

  cfg.metrics.update({
    "lean_fraction": MetricsTermCfg(func=lean.lean_fraction),
    "lean_command_rad": MetricsTermCfg(func=lean.lean_command_rad),
    "lean_error_deg": MetricsTermCfg(func=lean.lean_error_deg, params={"asset_cfg": torso()}),
  })
  if not play:
    cfg.curriculum["lean_max"] = CurriculumTermCfg(func=lean.lean_max_rad)
  return cfg
