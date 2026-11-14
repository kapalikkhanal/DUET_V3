"""Per-step tray and egg state shared by rewards, terminations and metrics.

A passive, zero-dimension action term (the ``PoultrySurfaceAction`` pattern):
it owns per-episode latches (which eggs are lost) and history (last tray
acceleration, for jerk), and caches one ``sample()`` per control step.

Everything here is simulator ground truth. Rewards, terminations, the critic
and the evaluation may read it; the deployed actor must not (its observations
are in ``observations.py`` and use only joint encoders, the robot IMU and a
tray-mounted IMU).

Accelerations. ``acc`` is the tray's gravity-compensated linear acceleration
from the finite difference of its velocity over one 20 ms control step: the
band a 50 Hz tray IMU with anti-alias filtering would report. ``acc_imu`` is
the instantaneous MuJoCo accelerometer at the last physics substep, minus
gravity. Both are reported; rewards use ``acc``.

Egg contact force. Estimated every physics substep from each egg's momentum
change, F = m (dv/dt - g), using ``qvel`` directly: the net contact force the
solver applied to the egg over that 5 ms step (tray and, rarely, a neighbour
egg). The per-control-step peak is reported. MuJoCo Warp's contact sensor
(netforce, per-substep history) was not used: on resting eggs it read a 4 N
median against the true ~0.6 N (checked on the CPU with mj_contactForce),
independent of contact stiffness and solver iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse

from src.tasks.eggtray.carry import CarryPoseCfg, compute_carry_geometry
from src.tasks.eggtray.tray_model import TrayCfg

GRAVITY = 9.81


@dataclass(kw_only=True)
class EggTrayStateCfg(ActionTermCfg):
  tray: TrayCfg
  carry: CarryPoseCfg
  tray_entity: str = "tray"
  egg_prefix: str = "egg_"
  lost_xy: float = 0.035
  """An egg is lost once its centre is this far (horizontally, tray frame) from
  its cup: past the cup's top edge (cup half-width 25.5 mm)."""
  lost_z: float = 0.035
  """... or lifted this far above its resting height (cup depth 30 mm)."""

  def build(self, env):
    return EggTrayState(self, env)


class EggTrayState(ActionTerm):
  cfg: EggTrayStateCfg

  def __init__(self, cfg: EggTrayStateCfg, env):
    super().__init__(cfg=cfg, env=env)
    self.tray = env.scene[cfg.tray_entity]
    self.eggs = [env.scene[f"{cfg.egg_prefix}{i}"] for i in range(cfg.tray.num_eggs)]
    geo = compute_carry_geometry(cfg.carry, cfg.tray)
    robot = env.scene[cfg.entity_name]
    self.hand_ids, _ = robot.find_bodies(geo.hands, preserve_order=True)
    rel = torch.tensor(np.stack([geo.weld_relpose[h] for h in geo.hands]), device=self.device, dtype=torch.float32)
    anc = torch.tensor(np.stack([geo.weld_anchor[h] for h in geo.hands]), device=self.device, dtype=torch.float32)
    self.weld_relpos, self.weld_relquat, self.weld_anchor = rel[:, :3], rel[:, 3:], anc
    xy = torch.tensor(cfg.tray.compartment_xy(), device=self.device, dtype=torch.float32)
    z = torch.full((len(xy), 1), cfg.tray.egg_rest_z(), device=self.device)
    self.egg_rest = torch.cat([xy, z], -1)  # [N, 3] tray frame
    n, e = self.num_envs, cfg.tray.num_eggs
    self.lost = torch.zeros(n, e, dtype=torch.bool, device=self.device)
    self.newly_lost = torch.zeros_like(self.lost)
    self.prev_vel = torch.zeros(n, 3, device=self.device)
    self.prev_acc = torch.zeros(n, 3, device=self.device)
    self.last_vel = torch.zeros(n, 3, device=self.device)
    self.last_acc = torch.zeros(n, 3, device=self.device)
    self.fresh = torch.ones(n, dtype=torch.bool, device=self.device)
    self.peak_force = torch.zeros(n, e, device=self.device)
    self.step_force = torch.zeros(n, e, device=self.device)
    v_adr = [egg.data.indexing.free_joint_v_adr[:3] for egg in self.eggs]
    self._egg_v_adr = torch.stack([torch.as_tensor(a, device=self.device) for a in v_adr]).long()
    self._egg_body = torch.tensor(
      [int(egg.data.indexing.root_body_id) for egg in self.eggs], device=self.device, dtype=torch.long
    )
    self._prev_egg_v = torch.zeros(n, e, 3, device=self.device)
    self._egg_v_fresh = torch.ones(n, dtype=torch.bool, device=self.device)
    self._g = torch.tensor([0.0, 0.0, GRAVITY], device=self.device)
    self._raw = torch.zeros(n, 0, device=self.device)
    self._key = None
    self._s: dict[str, torch.Tensor] = {}

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    # Start of a control step: the last sample becomes the previous one.
    self.prev_vel.copy_(self.last_vel)
    self.prev_acc.copy_(self.last_acc)
    self.step_force.zero_()

  def apply_actions(self) -> None:
    # Called before every physics substep: qvel holds the state after the
    # previous one, so consecutive calls bracket exactly one physics step.
    sim = self._env.sim
    v = sim.data.qvel[:, self._egg_v_adr]  # [B, N, 3], world-frame COM velocity
    mass = sim.model.body_mass
    mass = (mass[:, self._egg_body] if mass.dim() == 2 else mass[self._egg_body].expand(v.shape[0], -1))
    f = mass[..., None] * ((v - self._prev_egg_v) / self._env.physics_dt + self._g)
    f = torch.where(self._egg_v_fresh[:, None, None], torch.zeros_like(f), f)
    self.step_force.copy_(torch.maximum(self.step_force, f.norm(dim=-1)))
    self._prev_egg_v.copy_(v)
    self._egg_v_fresh.zero_()

  def reset(self, env_ids=None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    for buf in (self.lost, self.newly_lost, self.prev_vel, self.prev_acc,
                self.last_vel, self.last_acc, self.peak_force):
      buf[ids] = 0
    self.fresh[ids] = True
    self._egg_v_fresh[ids] = True
    self.step_force[ids] = 0
    self._key = None

  def egg_relative(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Egg position offset from its resting point and velocity, tray frame, [B, N, 3]."""
    td = self.tray.data
    pos, quat = td.root_link_pos_w, td.root_link_quat_w
    vel, ang = td.root_link_lin_vel_w, td.root_link_ang_vel_w
    egg_pos = torch.stack([e.data.root_link_pos_w for e in self.eggs], 1)
    egg_vel = torch.stack([e.data.root_link_lin_vel_w for e in self.eggs], 1)
    r = egg_pos - pos[:, None]
    q = quat[:, None].expand(-1, r.shape[1], -1)
    rel_pos = quat_apply_inverse(q, r) - self.egg_rest
    v_point = vel[:, None] + torch.cross(ang[:, None].expand_as(r), r, dim=-1)
    return rel_pos, quat_apply_inverse(q, egg_vel - v_point)

  def grasp_strain(self) -> torch.Tensor:
    """Weld stretch per grasping hand (m), [B, H]: grip point on the hand vs on the tray."""
    td = self.tray.data
    pos, quat = td.root_link_pos_w, td.root_link_quat_w
    robot = self._env.scene[self.cfg.entity_name].data
    hp = robot.body_link_pos_w[:, self.hand_ids]
    hq = robot.body_link_quat_w[:, self.hand_ids]
    B, H = hp.shape[:2]
    in_hand = (self.weld_relpos + quat_apply(self.weld_relquat, self.weld_anchor)).expand(B, -1, -1)
    grip_hand = hp + quat_apply(hq, in_hand)
    grip_tray = pos[:, None] + quat_apply(quat[:, None].expand(-1, H, -1), self.weld_anchor.expand(B, -1, -1))
    return (grip_hand - grip_tray).norm(dim=-1)

  def sample(self) -> dict[str, torch.Tensor]:
    key = self._env.common_step_counter
    if key == self._key:
      return self._s
    env = self._env
    td = self.tray.data
    pos, quat = td.root_link_pos_w, td.root_link_quat_w
    vel, ang = td.root_link_lin_vel_w, td.root_link_ang_vel_w
    R = matrix_from_quat(quat)
    dt = env.step_dt
    fresh = self.fresh
    acc = torch.where(fresh[:, None], torch.zeros_like(vel), (vel - self.prev_vel) / dt)
    jerk = torch.where(fresh[:, None], torch.zeros_like(acc), (acc - self.prev_acc) / dt)
    self.last_vel.copy_(vel)
    self.last_acc.copy_(acc)
    self.fresh.zero_()

    f_imu = env.scene["tray/tray_acc"].data  # specific force, tray frame
    acc_imu = quat_apply(quat, f_imu) - torch.tensor([0.0, 0.0, GRAVITY], device=self.device)

    rel_pos, rel_vel = self.egg_relative()
    out = (rel_pos[..., :2].norm(dim=-1) > self.cfg.lost_xy) | (rel_pos[..., 2] > self.cfg.lost_z)
    self.newly_lost.copy_(out & ~self.lost)
    self.lost |= out

    force = self.step_force.clone()  # [B, N] peak over this control step's substeps
    self.peak_force.copy_(torch.maximum(self.peak_force, force))

    strain = self.grasp_strain()

    up = R[:, :, 2]
    self._s = dict(
      pos=pos, quat=quat, R=R, vel=vel, ang_vel=ang,
      ang_vel_b=quat_apply_inverse(quat, ang),
      acc=acc, jerk=jerk, acc_imu=acc_imu,
      tilt=torch.acos(up[:, 2].clamp(-1.0, 1.0)),
      roll_pitch=torch.stack([torch.atan2(R[:, 2, 1], R[:, 2, 2]), torch.asin((-R[:, 2, 0]).clamp(-1, 1))], -1),
      egg_rel_pos=rel_pos, egg_rel_vel=rel_vel, egg_force=force,
      lost=self.lost, newly_lost=self.newly_lost, grasp_strain=strain,
    )
    self._key = key
    return self._s


def state(env) -> EggTrayState:
  return env.action_manager.get_term("eggtray_state")
