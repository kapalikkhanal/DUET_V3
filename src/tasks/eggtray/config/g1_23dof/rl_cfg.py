"""PPO for the 10-D arm residual (rsl_rl via mjlab), sized for short runs."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

EGGTRAY_MAX_ITERATIONS = 500
EGGTRAY_SAVE_INTERVAL = 25


def eggtray_ppo_runner_cfg(
  max_iterations: int = EGGTRAY_MAX_ITERATIONS,
  experiment_name: str = "EggTray_G1_23dof",
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      # 0.5 raw units = 0.025 rad of exploration on top of the carry pose.
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.003,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=EGGTRAY_SAVE_INTERVAL,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    logger="tensorboard",
    upload_model=False,
  )
