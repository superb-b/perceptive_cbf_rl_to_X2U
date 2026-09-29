"""X2 state-based dodge, ported from the supplied G1 MimicKit configuration.

Install at src/tasks/amp_loco/config/x2/dodge_env_cfgs.py.

Build directly on .env_cfgs.x2_amp_flat_env_cfg(play=play). The supplied X2
configuration identifies pelvis as the root and torso_link as the anchor.

This module requires the same project mdp/goal_command implementation as G1.
Only Python syntax has been checked outside the user's simulator. In particular,
verify that shared command/reward/event implementations have no G1 body-name
assumptions before training. Throw heights below are the supplied G1 defaults;
calibrate them against X2 geometry. This file does not configure the AMP runner,
sampling probabilities, or checkpoint transfer.
"""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.objects import get_ball_cfg
import src.tasks.amp_loco.mdp as mdp
from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommandCfg

from .env_cfgs import x2_amp_flat_env_cfg

# Shared with the Dodge runner in __init__.py; recursive WalkandRun/Dodge/Leap.
X2_DODGE_MOTION_DIR = str(
    Path(__file__).resolve().parents[4] / "assets/motions/x2/amp_dodge"
)


def x2_amp_dodge_state_flat_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the G1-style state dodge task on the supplied X2 flat base.

    Preserve X2 robot/action scales, joint ordering, actor/critic/AMP observations,
    physical regularizers, fall terminations, and robot reset settings. Use the
    X2 locomotion base rather than a G1 environment with its robot swapped out.
    """
    cfg = x2_amp_flat_env_cfg(play=play)
    for name in ("init_motion_loader", "reset_from_motion"):
        if name not in cfg.events:
            raise ValueError(f"X2 base is missing the RSI event {name!r}.")

    for name in ("init_motion_loader", "reset_from_motion"):
        cfg.events[name].params["motion_dir"] = X2_DODGE_MOTION_DIR

    print(f"[X2 DODGE] RSI motion_dir = {X2_DODGE_MOTION_DIR}")

    cfg.scene.entities = {**cfg.scene.entities, "ball": get_ball_cfg()}
    sensors = tuple(cfg.scene.sensors or ())
    if not any(sensor.name == "ball_robot_contact" for sensor in sensors):
        sensors += (
            ContactSensorCfg(
                name="ball_robot_contact",
                primary=ContactMatch(mode="body", pattern="ball", entity="ball"),
                secondary=ContactMatch(
                    mode="subtree", pattern="pelvis", entity="robot"
                ),
                fields=("found",),
                reduce="netforce",
                num_slots=1,
            ),
        )
    cfg.scene.sensors = sensors

    # G1 mixture: 20% ball-free stand, 10% zero-command with throws, remaining
    # 70% home-return with throws. Play: all home-return with throws.
    # Setting rel_standing_envs=1 would suppress every throw because of
    # skip_standing_envs=True below.
    twist = DodgeGoToGoalCommandCfg(
        entity_name="robot",
        resampling_time_range=(6.0, 10.0),
        radius=(0.5, 2.0),
        rel_standing_envs=0.0 if play else 0.2,
        rel_inplace_throw_envs=0.0 if play else 0.1,
        kp=1.5,
        kp_yaw=1.0,
        max_lin_vel_x=1.3,
        max_lin_vel_y=1.5,
        max_ang_vel_z=0.5,
        arrive_radius=0.25,
        dwell_time_range=(1.0, 3.0),
        simple_heading=False,
        debug_vis=True,
        ball_name="ball",
        back_offset=0.0,
        cbf_enabled=True,
        cbf_alpha=1.5,
        safe_radius=0.9,
        sense_radius=4.0,
        z_active=0.25,
    )
    twist.home_goal = True
    # Keep G1's threat/CBF bookkeeping, but do not filter the velocity command.
    # The safe rewards require command._dodge_threat maintained by this class.
    twist.cbf_filter_command = False
    cfg.commands["twist"] = twist

    # This is a flat home-return task. The inherited locomotion curricula may
    # expect UniformVelocityCommand.ranges (including through default params),
    # so do not carry them across the command-type replacement.
    cfg.curriculum = {}

    # G1 timed throws: a recovery interval instead of immediate ground rethrows.
    interval = (1.0, 4.0)
    cfg.events["reset_dodge_state"] = EventTermCfg(
        func=mdp.reset_dodge_state,
        mode="reset",
        params={
            "ball_name": "ball",
            "park_offset": (0.0, 3.0, 0.1),
            "throw_interval_range": interval,
        },
    )
    cfg.events["randomize_ball_size"] = EventTermCfg(
        func=dr.geom_size,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_collision",)),
            "operation": "abs",
            "ranges": (0.075, 0.125),
            "axes": [0],
        },
    )
    cfg.events["throw_ball_on_dwell"] = EventTermCfg(
        func=mdp.throw_ball_on_dwell,
        mode="step",
        params={
            "ball_name": "ball",
            "robot_name": "robot",
            "command_name": "twist",
            "dwell_time_s": 0.5,
            "command_threshold": 0.1,
            "speed_threshold": 0.3,
            "throw_interval_range": interval,
            "dist_range": (2.0, 3.0),
            "height_range": (1.5, 2.3),
            "angle_deg": 25.0,
            "flight_time_range": (0.58, 0.63),
            "lead_target": True,
            "aim_noise_scale": 0.1,
            "high_throw_fraction": 0.5,
            "high_launch_height_range": (0.4, 0.9),
            "high_target_z_range": (0.9, 1.1) if play else (0.9, 1.3),
            "skip_standing_envs": True,
        },
    )

    cfg.observations["ball_state"] = ObservationGroupCfg(
        terms={
            "ball_state": ObservationTermCfg(
                func=mdp.dodge_ball_state_b,
                params={"robot_name": "robot", "ball_name": "ball"},
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=0,
    )

    # Remove navigation/forced-flight objectives, as in G1's final state task.
    for name in (
        "goal_distance", "goal_progress", "goal_reached", "stop_at_goal",
        "leap_flight", "dodge_cbf", "dodge_sidestep",
    ):
        cfg.rewards.pop(name, None)
    for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity"):
        if name in cfg.rewards:
            cfg.rewards[name].weight = 0.5
    if "track_anchor_linear_velocity" in cfg.rewards:
        cfg.rewards["track_anchor_linear_velocity"].params["mask_inplace"] = True

    cfg.rewards["goal_distance"] = RewardTermCfg(
        func=mdp.goal_distance_reward,
        weight=0.3,
        params={"command_name": "twist", "std": 1.5},
    )
    cfg.rewards["mimickit_dodge"] = RewardTermCfg(
        func=mdp.mimickit_dodge_reward,
        weight=1.0,
        params={"robot_name": "robot", "ball_name": "ball"},
    )
    cfg.rewards["dodge_stillness_when_safe"] = RewardTermCfg(
        func=mdp.dodge_stillness_when_safe,
        weight=0.5,
        params={"command_name": "twist", "robot_name": "robot", "vel_scale": 2.0},
    )
    cfg.rewards["dodge_action_rate_when_safe"] = RewardTermCfg(
        func=mdp.dodge_action_rate_when_safe,
        weight=-0.05,
        params={"command_name": "twist"},
    )
    # The supplied G1 file DOES enable link-CBF despite its stale "no CBF" docstring.
    # Keep its effective default; the zero-weight sidestep term is omitted.
    cfg.rewards["dodge_link_cbf"] = RewardTermCfg(
        func=mdp.dodge_link_cbf_reward,
        weight=0.27,
        params={
            "robot_name": "robot", "ball_name": "ball",
            "alpha": 1.0, "margin": 0.05, "constraint_clip": 2.0,
        },
    )

    # A hit ends the episode immediately, but is excluded from the -200 fall
    # penalty. Preserve X2's own height/orientation limits, not G1 thresholds.
    cfg.terminations["ball_hit"] = TerminationTermCfg(
        func=mdp.ball_contact,
        params={
            "sensor_name": "ball_robot_contact",
            "delta_v_threshold": 1.5,
            "hit_dist": 1.0,
            "hit_z_min": 0.3,
        },
    )
    cfg.rewards["is_terminated"] = RewardTermCfg(
        func=mdp.is_terminated_except,
        weight=-200.0,
        params={"exclude_terms": ("ball_hit",)},
    )
    return cfg