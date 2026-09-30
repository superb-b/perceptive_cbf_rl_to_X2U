"""X2 skill runner with strict policy/value warm-start from locomotion.

Set X2_SKILL_INIT_CHECKPOINT for a NEW training run only. Do not also pass the
train.py pretrained/resume flags. Play uses the normal inherited runner.load.
The discriminator, AMP normalizer, optimizer and iteration are initialized fresh
for the Leap dataset; actor/critic normalizers and the action head are retained.
"""

import os
from pathlib import Path

import torch

from src.tasks.amp_loco.rl import AMPOnPolicyRunner
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
from .skill_env_cfgs import X2_SKILL_MOTION_DIR


def x2_amp_leap_goto_ppo_runner_cfg():
    cfg = g1_amp_ppo_runner_cfg()
    cfg.experiment_name = "x2_amp_leap_goto"
    cfg.save_interval = 100
    cfg.amp_motion_files = str(Path(__file__).resolve().parents[4] / "assets/motions/x2/amp_skill_70_30.json")
    cfg.amp_reward_coef = 0.5
    cfg.amp_task_reward_lerp = 0.5
    # No ball-state group: preserve locomotion input sizes for strict loading.
    cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",)}
    print(f"[X2 SKILL] AMP expert motion_dir = {cfg.amp_motion_files}")
    return cfg


def _validate_state(module, state, label):
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {label}_state_dict must be a dictionary")
    current = module.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    mismatch = []
    for key in current.keys() & state.keys():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            mismatch.append((key, "not a tensor"))
        elif tensor.shape != current[key].shape:
            mismatch.append((key, tuple(tensor.shape), tuple(current[key].shape)))
    if missing or unexpected or mismatch:
        raise RuntimeError(
            f"[X2 SKILL INIT] {label} mismatch: missing={missing}, "
            f"unexpected={unexpected}, shape={mismatch}. "
            "Use the original X2 locomotion checkpoint and matching robot/obs config; "
            "do not use a Dodge checkpoint with six extra ball inputs."
        )


class X2SkillOnPolicyRunner(AMPOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        init_path = os.environ.get("X2_SKILL_INIT_CHECKPOINT", "").strip()
        if init_path:
            path = Path(init_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"X2_SKILL_INIT_CHECKPOINT does not exist: {path}")
        super().__init__(*args, **kwargs)
        if not init_path:
            print("[X2 SKILL INIT] no warm-start requested; play/resume uses runner.load")
            return
        # The user explicitly supplies their own trusted training checkpoint.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("actor_state_dict", "critic_state_dict"):
            if key not in checkpoint:
                raise KeyError(f"Checkpoint is missing {key}")
        # Validate BOTH modules before changing either. Keep the full action head.
        _validate_state(self.alg.actor, checkpoint["actor_state_dict"], "actor")
        _validate_state(self.alg.critic, checkpoint["critic_state_dict"], "critic")
        self.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        print(f"[X2 SKILL INIT] checkpoint: {path}")
        print("[X2 SKILL INIT] actor + critic strict load OK; action head + obs normalizers kept")
        print("[X2 SKILL INIT] optimizer/discriminator/AMP normalizer/iteration remain fresh")
