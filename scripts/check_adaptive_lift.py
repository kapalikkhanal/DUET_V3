"""Short headless GPU trials at the mature Adaptive_Lift curriculum settings."""
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from src.tasks.duet.config.g1_23dof.adaptive_lift import TASK
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import adaptive_lift as lift


@torch.inference_mode()
def main():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 32
  cfg.seed = 19
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.common_step_counter = 15000 * ITER
    surface = env.action_manager.get_term("poultry_surface")
    command = env.command_manager.get_term("clearance")
    for seed in (19, 37):
      obs, _ = env.reset(seed=seed)
      assert obs["actor"].shape == (32, 444) and obs["critic"].shape == (32, 286)
      assert env.action_manager.total_action_dim == 13
      assert env.reward_manager.get_term_cfg("poultry_swing_height").weight == -.16
      assert env.scene.terrain.terrain_types.unique().tolist() == [0, 1, 2, 3]
      assert surface.endpoints.shape == (2, 14, 3)
      transitions = 0
      swings = 0
      for t in range(128):
        if t == 8:
          # Make a resample due on the next step, including in moving environments.
          command.time_left.zero_()
        previous = command.command.clone()
        action = torch.randn(32, 13, device=env.device) * .35
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
        # Metrics ran before automatic resets. Read their cached landing mask without
        # calling sample() again on a just-respawned robot.
        swings += int(surface._sample["tracked_landing"].sum())
      assert transitions > 0 and swings > 0
      print(f"Adaptive_Lift seed {seed}: 4096 env-steps, {transitions} moving command "
            f"transition samples, {swings} completed moving swings, all four tiles.", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
