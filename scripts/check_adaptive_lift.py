"""Short headless GPU trials at the mature Adaptive_Lift curriculum settings."""
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from src.tasks.duet.config.g1_23dof.adaptive_lift import TASK
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import adaptive_lift as lift
from src.tasks.duet.mdp.lift_terrain import LAYOUT, MATERIALS, SPLIT, material_at, split_offset

NUM_ENVS = 64


def check_compiled_beds(env, surface):
  """Ray-cast every litter cell and split-level strip in the compiled scene against the analytic map."""
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  terrain = surface.terrain_cfg
  kinds = surface.col_kind.cpu()
  xy = []
  for row in range(terrain.num_rows):
    for col in range(terrain.num_cols):
      if kinds[col] < 0:
        continue
      if kinds[col] == SPLIT:
        width = float(surface.split_w[col])
        for k in range(int(round(terrain.size[1] / width))):
          xy.append(((row - terrain.num_rows / 2 + .3) * terrain.size[0],
                     (col - terrain.num_cols / 2) * terrain.size[1] + (k + .5) * width))
        continue
      for i, j in np.ndindex(LAYOUT.shape[1:]):
        xy.append(((row - terrain.num_rows / 2 + (i + .5) / LAYOUT.shape[1]) * terrain.size[0],
                   (col - terrain.num_cols / 2 + (j + .5) / LAYOUT.shape[2]) * terrain.size[1]))
  xy = torch.tensor(xy, dtype=torch.float32)
  material, valid = material_at(xy, terrain, surface.layout.cpu(), kinds)
  assert valid.all()
  raised = split_offset(xy, terrain, surface.split_h.cpu(), surface.split_w.cpu())
  terrain_body = model.body("terrain").id
  geom = np.empty(1, dtype=np.int32)
  for (x, y), m, r in zip(xy.tolist(), material.tolist(), raised.tolist()):
    distance = mujoco.mj_ray(model, data, np.array([x, y, 1.]), np.array([0., 0., -1.]), None, 1, -1, geom)
    if model.geom_bodyid[geom[0]] != terrain_body:
      continue  # the single robot model standing at the origin
    assert abs(1. - distance - MATERIALS[m][0] - r) < 1e-4, f"Bed mismatch at {x:.2f}, {y:.2f}"
  steps = sorted({round(v, 4) for v in raised.tolist() if v > 0})
  print(f"Compiled beds match the analytic map at {len(xy)} cells; split steps {steps[0]*100:.1f}-"
        f"{steps[-1]*100:.1f} cm.", flush=True)


@torch.inference_mode()
def main():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 19
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.common_step_counter = 15000 * ITER
    surface = env.action_manager.get_term("poultry_surface")
    command = env.command_manager.get_term("clearance")
    pull = env.action_manager.get_term("pull_force")
    pull_bodies = pull._entity.data.indexing.body_ids[pull.body_ids]
    check_compiled_beds(env, surface)
    for seed in (19, 37):
      obs, _ = env.reset(seed=seed)
      assert obs["actor"].shape == (NUM_ENVS, 444) and obs["critic"].shape == (NUM_ENVS, 297)
      assert env.action_manager.total_action_dim == 13
      assert env.reward_manager.get_term_cfg("poultry_swing_height").weight == -.16
      assert env.reward_manager.get_term_cfg("touchdown_tilt").weight == -2.
      assert abs(surface.cfg.litter_depth_range[1] - .08) < 1e-9
      assert surface.loose_depth.min() >= .02 and surface.loose_depth.max() <= .08
      kinds = surface.col_kind[env.scene.terrain.terrain_types].unique().tolist()
      assert kinds == [-1, 0, 1, 2, 3, 4], kinds
      assert pull.scale == 1.
      pull.gap.uniform_(0., 1.)  # start pulls inside this short trial
      peak_pull, released_steps = 0., 0
      assert surface.endpoints.shape == (2, 14, 3)
      transitions = swings = 0
      for t in range(128):
        if t == 8:
          # Make a resample due on the next step, including in moving environments.
          command.time_left.zero_()
        previous = command.command.clone()
        action = torch.randn(NUM_ENVS, 13, device=env.device) * .35
        obs, reward, terminated, timeout, _ = env.step(action)
        alive = ~(terminated | timeout)
        delta = (command.command - previous).abs()
        moving = env.command_manager.get_command("twist").abs().sum(-1) > .1
        transitions += int(((delta.amax(-1) > 1e-5) & alive & moving).sum())
        assert (delta[alive] <= command.cfg.rate_mps * env.step_dt + 1e-7).all()
        assert (command.command >= .04).all() and (command.command <= .11).all()
        torch.testing.assert_close(obs["actor"][:, -2:], lift.clearance_commands(env))
        for data in (obs["actor"], obs["critic"], reward, env.sim.data.qvel,
                     surface.last_force, surface.last_apex, lift.leg_power_w(env)):
          assert torch.isfinite(data).all(), f"Non-finite trial at seed {seed}, step {t}"
        assert (surface.last_force.norm(dim=-1) <= surface.cfg.max_drag_force + 1e-4).all()
        applied = env.sim.data.xfrc_applied[:, pull_bodies, :3].sum(1)
        torch.testing.assert_close(applied, pull.force)
        assert (pull.force.norm(dim=-1) <= pull.cfg.max_force + 1e-4).all()
        peak_pull = max(peak_pull, float(pull.force.norm(dim=-1).max()))
        released_steps += int(pull.released.sum())
        # Metrics ran before automatic resets. Read their cached landing mask without
        # calling sample() again on a just-respawned robot.
        swings += int(surface._sample["tracked_landing"].sum())
      assert transitions > 0 and swings > 0 and peak_pull > 40 and released_steps > 0
      print(f"Adaptive_Lift seed {seed}: {NUM_ENVS * 128} env-steps, {transitions} moving command "
            f"transition samples, {swings} completed moving swings, peak pull {peak_pull:.0f} N, "
            f"{released_steps} released env-steps, all tile kinds.", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
