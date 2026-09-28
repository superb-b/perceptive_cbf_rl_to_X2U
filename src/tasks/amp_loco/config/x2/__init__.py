from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import x2_amp_flat_env_cfg

# 第一阶段先复用 G1 PPO hyperparameters。
from src.tasks.amp_loco.config.g1.rl_cfg import (
    g1_amp_ppo_runner_cfg,
)


register_mjlab_task(
    task_id="X2-AMP-Flat",
    env_cfg=x2_amp_flat_env_cfg(),
    play_env_cfg=x2_amp_flat_env_cfg(play=True),
    rl_cfg=g1_amp_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
