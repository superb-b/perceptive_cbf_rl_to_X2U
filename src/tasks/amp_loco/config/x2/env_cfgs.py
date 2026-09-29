"""X2 AMP Locomotion environment configurations."""

import os

from src.assets.robots import (
    X2_ACTION_SCALE,
    get_x2_robot_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
import src.tasks.amp_loco.mdp as amp_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg


def x2_amp_rough_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create X2 rough-terrain AMP locomotion configuration."""

    cfg = make_amp_env_cfg()

    print("\n[X2 DEBUG] root-height configs")

    print(
        "track_root_height:",
        cfg.rewards["track_root_height"].params,
    )

    print(
        "bad_base_height:",
        cfg.terminations["bad_base_height"].params,
    )

    print()

    # ------------------------------------------------------------------
    # X2 sensor mapping
    # ------------------------------------------------------------------

    # X2 MJCF:
    #   robot/body-angular-velocity
    #   robot/body-linear-vel
    #
    # Base AMP config expects G1 names:
    #   robot/imu_ang_vel
    #   robot/imu_lin_vel

    cfg.observations["actor"].terms["base_ang_vel"].params[
        "sensor_name"
    ] = "robot/body-angular-velocity"

    cfg.observations["critic"].terms["base_ang_vel"].params[
        "sensor_name"
    ] = "robot/body-angular-velocity"

    cfg.observations["critic"].terms["base_lin_vel"].params[
        "sensor_name"
    ] = "robot/body-linear-vel"

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    cfg.sim.mujoco.ccd_iterations = 128
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = 48

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------

    cfg.scene.entities = {
        "robot": get_x2_robot_cfg(),
    }

    # ------------------------------------------------------------------
    # Terrain scan
    # ------------------------------------------------------------------

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "pelvis"

    # ------------------------------------------------------------------
    # X2 body names
    # ------------------------------------------------------------------

    body_names = (
        "pelvis",

        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",

        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",

        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",

        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    foot_body_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    anchor_name = "torso_link"
    root_name = "pelvis"

    # ------------------------------------------------------------------
    # Contact sensors
    # ------------------------------------------------------------------

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (
        cfg.scene.sensors or ()
    ) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    # ------------------------------------------------------------------
    # Terrain curriculum
    # ------------------------------------------------------------------

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    joint_pos_action = cfg.actions["joint_pos"]

    assert isinstance(
        joint_pos_action,
        JointPositionActionCfg,
    )

    joint_pos_action.scale = X2_ACTION_SCALE

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    cfg.viewer.body_name = "torso_link"

    # ------------------------------------------------------------------
    # Velocity command visualization
    # ------------------------------------------------------------------

    twist_cmd = cfg.commands["twist"]

    assert isinstance(
        twist_cmd,
        UniformVelocityCommandCfg,
    )

    twist_cmd.rel_standing_envs = 0.25
    twist_cmd.viz.z_offset = 1.15

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    # X2 currently has unnamed geoms, so do not override
    # foot_friction geom_names yet.

    cfg.events["base_com"].params["asset_cfg"].body_names = (
        "torso_link",
    )

    # ------------------------------------------------------------------
    # Motion initialization
    # ------------------------------------------------------------------

    _motion_base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "assets",
        "motions",
        "x2",
        "amp_dodge",
    )

    _motion_dir = os.path.abspath(_motion_base)

    cfg.events["init_motion_loader"].params[
        "motion_dir"
    ] = _motion_dir

    # No separate recovery clips for now.
    cfg.events["init_motion_loader"].params[
        "recovery_dir"
    ] = None

    # Reset event MUST use exactly the same directory.
    cfg.events["reset_from_motion"].params[
        "motion_dir"
    ] = _motion_dir

    cfg.events["init_motion_loader"].params[
        "delay_reset_env_ratio"
    ] = 0.4

    cfg.events["init_motion_loader"].params[
        "max_delay_steps"
    ] = 250

    # Temporary sanity check.
    print(
        f"[X2 AMP] init motion_dir  = "
        f"{cfg.events['init_motion_loader'].params['motion_dir']}"
    )
    print(
        f"[X2 AMP] reset motion_dir = "
        f"{cfg.events['reset_from_motion'].params['motion_dir']}"
    )

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    cfg.rewards["foot_slip"].params[
        "asset_cfg"
    ] = SceneEntityCfg(
        "robot",
        site_names=(
            "left_foot",
            "right_foot",
        ),
    )

    cfg.rewards[
        "track_anchor_linear_velocity"
    ].params["anchor_cfg"].body_names = (
        anchor_name,
    )

    cfg.rewards["stand_default_pose"] = RewardTermCfg(
        func=amp_mdp.stand_default_pose_l2,
        weight=-1.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(".*",),
            ),
            "lin_vel_threshold": 0.10,
            "ang_vel_threshold": 0.10,
        },
    )

    cfg.rewards["low_speed_arm_pose"] = RewardTermCfg(
        func=amp_mdp.low_speed_arm_default_pose_l2,
        weight=-0.5,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    r".*shoulder.*",
                    r".*elbow.*",
                    r".*wrist.*",
                ),
            ),
            "speed_scale": 0.6,
            "yaw_scale": 0.3,
        },
    )


    cfg.rewards[
        "track_anchor_angular_velocity"
    ].params["anchor_cfg"].body_names = (
        anchor_name,
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-0.1,
        params={
            "sensor_name": self_collision_cfg.name,
            "force_threshold": 10.0,
        },
    )

    cfg.rewards[
        "body_ang_vel_xy_l2"
    ].params["body_cfg"].body_names = (
        root_name,
    )

    # ------------------------------------------------------------------
    # Critic observations
    # ------------------------------------------------------------------

    cfg.observations[
        "critic"
    ].terms[
        "body_pos_b"
    ].params[
        "anchor_cfg"
    ].body_names = (
        anchor_name,
    )

    cfg.observations[
        "critic"
    ].terms[
        "body_pos_b"
    ].params[
        "body_cfg"
    ].body_names = body_names

    cfg.observations[
        "critic"
    ].terms[
        "body_ori_b"
    ].params[
        "anchor_cfg"
    ].body_names = (
        anchor_name,
    )

    cfg.observations[
        "critic"
    ].terms[
        "body_ori_b"
    ].params[
        "body_cfg"
    ].body_names = body_names

    # ------------------------------------------------------------------
    # AMP observations
    # ------------------------------------------------------------------

    for term_name in (
        "body_pos_b",
        "body_ori_b",
        "body_lin_vel_b",
        "body_ang_vel_b",
    ):
        term = cfg.observations["amp"].terms[
            term_name
        ]

        term.params[
            "anchor_cfg"
        ].body_names = (
            anchor_name,
        )

        term.params[
            "body_cfg"
        ].body_names = body_names

    # ------------------------------------------------------------------
    # Play mode
    # ------------------------------------------------------------------

    if play:
        cfg.episode_length_s = int(1e9)

        cfg.observations[
            "actor"
        ].enable_corruption = False

        cfg.events.pop(
            "push_robot",
            None,
        )

        cfg.curriculum = {}

        cfg.events[
            "randomize_terrain"
        ] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        cfg.events[
            "init_motion_loader"
        ].params[
            "delay_reset_env_ratio"
        ] = 1.0

    return cfg


def x2_amp_flat_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create X2 flat-terrain AMP locomotion configuration."""

    cfg = x2_amp_rough_env_cfg(
        play=play,
    )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    cfg.sim.njmax = 640
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.nconmax = None

    # ------------------------------------------------------------------
    # Flat terrain
    # ------------------------------------------------------------------

    assert cfg.scene.terrain is not None

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ------------------------------------------------------------------
    # Remove terrain raycast on flat terrain
    # ------------------------------------------------------------------

    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name != "terrain_scan"
    )


    # ------------------------------------------------------------------
    # Play command ranges
    # ------------------------------------------------------------------

    if play:
        twist_cmd = cfg.commands["twist"]

        assert isinstance(
            twist_cmd,
            UniformVelocityCommandCfg,
        )

        twist_cmd.ranges.lin_vel_x = (
            -1.5,
            3.0,
        )

        twist_cmd.ranges.lin_vel_y = (
            -1.0,
            1.0,
        )

        twist_cmd.ranges.ang_vel_z = (
            -3.14 / 2,
            3.14 / 2,
        )

    return cfg