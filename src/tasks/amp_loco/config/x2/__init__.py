"""Register X2 AMP locomotion and G1-style state dodge tasks."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg

from .env_cfgs import x2_amp_flat_env_cfg
from .dodge_env_cfgs import (
    X2_DODGE_MOTION_DIR,
    x2_amp_dodge_state_flat_env_cfg,
)


def x2_amp_dodge_ppo_runner_cfg():
    """G1 Dodge's explicit AMP blend/obs settings with the X2 motion dataset."""
    cfg = g1_amp_ppo_runner_cfg()
    cfg.experiment_name = "x2_amp_dodge"
    cfg.save_interval = 1000
    cfg.amp_motion_files = X2_DODGE_MOTION_DIR
    cfg.amp_reward_coef = 0.5
    cfg.amp_task_reward_lerp = 0.5
    cfg.obs_groups = {
        "actor": ("actor", "ball_state"),
        "critic": ("critic", "ball_state"),
    }
    print(f"[X2 DODGE] AMP expert motion_dir = {cfg.amp_motion_files}")
    return cfg


register_mjlab_task(
    task_id="X2-AMP-Flat",
    env_cfg=x2_amp_flat_env_cfg(),
    play_env_cfg=x2_amp_flat_env_cfg(play=True),
    rl_cfg=g1_amp_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="X2-AMP-Dodge-State-Flat",
    env_cfg=x2_amp_dodge_state_flat_env_cfg(),
    play_env_cfg=x2_amp_dodge_state_flat_env_cfg(play=True),
    rl_cfg=x2_amp_dodge_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)


# BEGIN X2 LEAP GOTO SKILL TASK
from .skill_env_cfgs import x2_amp_leap_goto_flat_env_cfg
from .skill_rl_cfg import (
    X2SkillOnPolicyRunner,
    x2_amp_leap_goto_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task as _register_x2_skill_task

_register_x2_skill_task(
    task_id="X2-AMP-Leap-GoTo-Flat",
    env_cfg=x2_amp_leap_goto_flat_env_cfg(),
    play_env_cfg=x2_amp_leap_goto_flat_env_cfg(play=True),
    rl_cfg=x2_amp_leap_goto_ppo_runner_cfg(),
    runner_cls=X2SkillOnPolicyRunner,
)
# END X2 LEAP GOTO SKILL TASK

# BEGIN X2 TAKEOFF TASK
from .takeoff_cfgs import x2_amp_takeoff_env_cfg, x2_amp_takeoff_rl_cfg
from .skill_rl_cfg import X2SkillOnPolicyRunner as _X2TakeoffRunner
from mjlab.tasks.registry import register_mjlab_task as _register_x2_takeoff

_register_x2_takeoff(
    task_id="X2-AMP-Takeoff-Flat",
    env_cfg=x2_amp_takeoff_env_cfg(),
    play_env_cfg=x2_amp_takeoff_env_cfg(play=True),
    rl_cfg=x2_amp_takeoff_rl_cfg(),
    runner_cls=_X2TakeoffRunner,
)
# END X2 TAKEOFF TASK
