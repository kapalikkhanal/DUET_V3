from mjlab.tasks.registry import register_mjlab_task

from src.tasks.eggtray.rl import EggTrayOnPolicyRunner

from .env_cfgs import CARRY_12, TRAY_12, unitree_g1_23dof_eggtray_env_cfg
from .rl_cfg import eggtray_ppo_runner_cfg

# Flat ground: debugging and Phase A verification.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-EggTray-Flat",
  env_cfg=unitree_g1_23dof_eggtray_env_cfg(terrain="flat"),
  play_env_cfg=unitree_g1_23dof_eggtray_env_cfg(play=True, terrain="flat"),
  rl_cfg=eggtray_ppo_runner_cfg(),
  runner_cls=EggTrayOnPolicyRunner,
)

# The generated broiler-house floor (built_up litter) with the loose-litter
# drag layer: training and evaluation.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-EggTray-Poultry",
  env_cfg=unitree_g1_23dof_eggtray_env_cfg(terrain="poultry"),
  play_env_cfg=unitree_g1_23dof_eggtray_env_cfg(play=True, terrain="poultry"),
  rl_cfg=eggtray_ppo_runner_cfg(),
  runner_cls=EggTrayOnPolicyRunner,
)

# Twelve eggs (4 x 3 tray, same width): evaluation of a six-egg policy.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-EggTray-Poultry-12Egg",
  env_cfg=unitree_g1_23dof_eggtray_env_cfg(terrain="poultry", tray=TRAY_12, carry=CARRY_12),
  play_env_cfg=unitree_g1_23dof_eggtray_env_cfg(
    play=True, terrain="poultry", tray=TRAY_12, carry=CARRY_12
  ),
  rl_cfg=eggtray_ppo_runner_cfg(),
  runner_cls=EggTrayOnPolicyRunner,
)
