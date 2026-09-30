"""Experimental ground-start takeoff stage; shared MDP remains unchanged."""
from pathlib import Path

from src.tasks.amp_loco.mdp.events import reset_to_default_stand
from .skill_env_cfgs import x2_amp_leap_goto_flat_env_cfg
from .skill_rl_cfg import x2_amp_leap_goto_ppo_runner_cfg


def x2_amp_takeoff_env_cfg(play=False):
    # Inherit the LIVE successful configuration, not a copy of an old upload.
    cfg = x2_amp_leap_goto_flat_env_cfg(play=play)
    if "ball" in cfg.scene.entities:
        raise ValueError("Takeoff stage must have no ball")
    event = cfg.events["reset_from_motion"]
    event.func = reset_to_default_stand
    event.params = {}
    # Retain loader startup for inherited bookkeeping, but no RSI fall grace.
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.episode_length_s = 20.0
    command = cfg.commands["twist"]
    command.radius = (0.8, 1.5)
    command.rel_standing_envs = 0.0
    command.rel_inplace_throw_envs = 0.0
    # Keep velocity caps, gain, arrival tolerance, dwell and heading unchanged.
    flight = cfg.rewards["leap_flight"]
    if flight.func.__name__ != "leap_flight_time":
        raise ValueError("Unexpected leap_flight reward; inspect before tuning")
    flight.weight = 4.0
    # All physical costs, fall thresholds, observation/action layouts preserved.
    print("[X2 TAKEOFF] default-standing resets; delayed-reset ratio=0; no ball")
    print(f"[X2 TAKEOFF] radius={command.radius}; arrive_radius={command.arrive_radius}")
    print(f"[X2 TAKEOFF] leap_flight weight={flight.weight}; params={flight.params}")
    return cfg


def x2_amp_takeoff_rl_cfg():
    cfg = x2_amp_leap_goto_ppo_runner_cfg()
    cfg.experiment_name = "x2_amp_takeoff"
    cfg.amp_motion_files = str(
        Path(__file__).resolve().parents[4] / "assets/motions/x2/amp_takeoff_30_70.json"
    )
    print(f"[X2 TAKEOFF] AMP experts={cfg.amp_motion_files}")
    return cfg
