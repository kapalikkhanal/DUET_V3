"""Reset: arms at the carry pose, tray in the hands, one egg per cup."""

from __future__ import annotations

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

from src.tasks.eggtray.carry import CarryPoseCfg, compute_carry_geometry
from src.tasks.eggtray.tray_model import TrayCfg


def reset_carry_state(
  env,
  env_ids: torch.Tensor | None,
  tray: TrayCfg,
  carry: CarryPoseCfg,
  egg_xy_jitter: float = 0.002,
  egg_drop: float = 0.001,
  robot_name: str = "robot",
  tray_name: str = "tray",
  egg_prefix: str = "egg_",
) -> None:
  """Must run after the robot's root and joint reset events.

  The robot's root pose is read from ``qpos`` (written by the earlier reset
  events this call), not from the derived ``xpos``, which is stale until the
  next forward pass. The tray pose relative to the pelvis comes from the same
  CPU forward kinematics that set the weld ``relpose``, so the grasp starts
  with zero constraint violation.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  geo = compute_carry_geometry(carry, tray)
  robot = env.scene[robot_name]
  arm_ids, _ = robot.find_joints(list(geo.arm_joint_names), preserve_order=True)
  n = len(env_ids)
  q_arm = torch.tensor(geo.q_nominal, device=env.device, dtype=torch.float32).expand(n, -1)
  robot.write_joint_state_to_sim(q_arm, torch.zeros_like(q_arm), joint_ids=arm_ids, env_ids=env_ids)

  root = robot.data.data.qpos[env_ids][:, robot.data.indexing.free_joint_q_adr]
  root_pos, root_quat = root[:, :3], root[:, 3:7]
  t_pos = torch.tensor(geo.tray_pos, device=env.device, dtype=torch.float32).expand(n, -1)
  t_quat = torch.tensor(geo.tray_quat, device=env.device, dtype=torch.float32).expand(n, -1)
  tray_pos = root_pos + quat_apply(root_quat, t_pos)
  tray_quat = quat_mul(root_quat, t_quat)
  zeros = torch.zeros(n, 6, device=env.device)
  env.scene[tray_name].write_root_state_to_sim(torch.cat([tray_pos, tray_quat, zeros], -1), env_ids)

  rest = np.concatenate([tray.compartment_xy(), np.full((tray.num_eggs, 1), tray.egg_rest_z() + egg_drop)], 1)
  rest = torch.tensor(rest, device=env.device, dtype=torch.float32)
  for i in range(tray.num_eggs):
    local = rest[i].expand(n, -1).clone()
    local[:, :2] += (torch.rand(n, 2, device=env.device) * 2 - 1) * egg_xy_jitter
    yaw = (torch.rand(n, device=env.device) * 2 - 1) * np.pi
    egg_quat = quat_mul(tray_quat, quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw))
    egg_pos = tray_pos + quat_apply(tray_quat, local)
    env.scene[f"{egg_prefix}{i}"].write_root_state_to_sim(torch.cat([egg_pos, egg_quat, zeros], -1), env_ids)
