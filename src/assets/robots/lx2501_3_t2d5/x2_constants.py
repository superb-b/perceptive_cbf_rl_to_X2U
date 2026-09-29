"""X2 (lx2501_3_t2d5) robot constants for mjlab.

First-pass PAC-MAN / perceptive_cbf_rl integration.

Sources used for this configuration:
- X2 active_joint_name / nominal_configuration
- X2 robot_model.yaml screenshots
- X2 trajectory_generator.yaml:
    Kp = 40.0
    Kd = 10.0

Important:
- The X2 physical model has 31 actuated joints.
- The dodge policy may later exclude the 2 head joints and use 29 actions.
- Joint mapping should always be done by joint name, NOT MuJoCo joint index.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


# =============================================================================
# XML
# =============================================================================

X2_XML: Path = (
    SRC_PATH
    / "assets"
    / "robots"
    / "lx2501_3_t2d5"
    / "xmls"
    / "x2_mjlab.xml"
)

assert X2_XML.exists(), f"X2 XML does not exist: {X2_XML}"


def get_spec() -> mujoco.MjSpec:
    """Load the X2 MuJoCo specification."""
    return mujoco.MjSpec.from_file(str(X2_XML))


# =============================================================================
# Joint names
#
# IMPORTANT:
# This follows the X2 controller active_joint_name order:
#
#   LEG -> WAIST -> HEAD -> ARM
#
# Do NOT assume this is the same as the MuJoCo joint-id order.
# =============================================================================

X2_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

X2_WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)

X2_HEAD_JOINT_NAMES = (
    "head_yaw_joint",
    "head_pitch_joint",
)

X2_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

X2_JOINT_NAMES = (
    X2_LEG_JOINT_NAMES
    + X2_WAIST_JOINT_NAMES
    + X2_HEAD_JOINT_NAMES
    + X2_ARM_JOINT_NAMES
)

assert len(X2_JOINT_NAMES) == 31


# 29-DoF body-only ordering.
#
# This is useful for the dodge policy if the head is held at its nominal
# position rather than controlled by RL.
X2_BODY_JOINT_NAMES = (
    X2_LEG_JOINT_NAMES
    + X2_WAIST_JOINT_NAMES
    + X2_ARM_JOINT_NAMES
)

assert len(X2_BODY_JOINT_NAMES) == 29


# =============================================================================
# Nominal configuration
# =============================================================================

X2_NOMINAL_JOINT_POS = {
    # -------------------------------------------------------------------------
    # Left leg
    # -------------------------------------------------------------------------
    "left_hip_pitch_joint": -0.24,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.45,
    "left_ankle_pitch_joint": -0.21,
    "left_ankle_roll_joint": 0.0,

    # -------------------------------------------------------------------------
    # Right leg
    # -------------------------------------------------------------------------
    "right_hip_pitch_joint": -0.24,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.45,
    "right_ankle_pitch_joint": -0.21,
    "right_ankle_roll_joint": 0.0,

    # -------------------------------------------------------------------------
    # Waist
    # -------------------------------------------------------------------------
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,

    # -------------------------------------------------------------------------
    # Head
    # -------------------------------------------------------------------------
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,

    # -------------------------------------------------------------------------
    # Left arm
    # -------------------------------------------------------------------------
    "left_shoulder_pitch_joint": 0.196,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,

    # -------------------------------------------------------------------------
    # Right arm
    # -------------------------------------------------------------------------
    "right_shoulder_pitch_joint": 0.196,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
}


# =============================================================================
# Real robot limits
#
# Values below are taken from the supplied X2 robot_model.yaml screenshots.
#
# Position limits should still be kept in the MJCF as the authoritative
# mechanical joint limits.
# =============================================================================

X2_MAX_EFFORT = {
    # Legs ---------------------------------------------------------------
    "left_hip_pitch_joint": 112.5,
    "left_hip_roll_joint": 112.5,
    "left_hip_yaw_joint": 112.5,
    "left_knee_joint": 112.5,
    "left_ankle_pitch_joint": 41.0,
    "left_ankle_roll_joint": 26.5,

    "right_hip_pitch_joint": 112.5,
    "right_hip_roll_joint": 112.5,
    "right_hip_yaw_joint": 112.5,
    "right_knee_joint": 112.5,
    "right_ankle_pitch_joint": 41.0,
    "right_ankle_roll_joint": 26.5,

    # Waist --------------------------------------------------------------
    "waist_yaw_joint": 112.5,
    "waist_pitch_joint": 53.0,
    "waist_roll_joint": 53.0,

    # Head ---------------------------------------------------------------
    "head_yaw_joint": 2.6,
    "head_pitch_joint": 0.6,

    # Left arm -----------------------------------------------------------
    "left_shoulder_pitch_joint": 41.0,
    "left_shoulder_roll_joint": 41.0,
    "left_shoulder_yaw_joint": 26.5,
    "left_elbow_joint": 26.5,
    "left_wrist_yaw_joint": 26.5,
    "left_wrist_pitch_joint": 4.8,
    "left_wrist_roll_joint": 4.8,

    # Right arm ----------------------------------------------------------
    "right_shoulder_pitch_joint": 41.0,
    "right_shoulder_roll_joint": 41.0,
    "right_shoulder_yaw_joint": 26.5,
    "right_elbow_joint": 26.5,
    "right_wrist_yaw_joint": 26.5,
    "right_wrist_pitch_joint": 4.8,
    "right_wrist_roll_joint": 4.8,
}


X2_MAX_VELOCITY = {
    # Legs ---------------------------------------------------------------
    "left_hip_pitch_joint": 12.356931,
    "left_hip_roll_joint": 12.356931,
    "left_hip_yaw_joint": 12.356931,
    "left_knee_joint": 12.356931,
    "left_ankle_pitch_joint": 14.556046,
    "left_ankle_roll_joint": 18.744836,

    "right_hip_pitch_joint": 12.356931,
    "right_hip_roll_joint": 12.356931,
    "right_hip_yaw_joint": 12.356931,
    "right_knee_joint": 12.356931,
    "right_ankle_pitch_joint": 14.556046,
    "right_ankle_roll_joint": 18.744836,

    # Waist --------------------------------------------------------------
    "waist_yaw_joint": 12.356931,
    "waist_pitch_joint": 18.744836,
    "waist_roll_joint": 18.744836,

    # Head ---------------------------------------------------------------
    "head_yaw_joint": 6.019,
    "head_pitch_joint": 6.28,

    # Arms ---------------------------------------------------------------
    "left_shoulder_pitch_joint": 14.556046,
    "left_shoulder_roll_joint": 14.556046,
    "left_shoulder_yaw_joint": 18.744836,
    "left_elbow_joint": 18.744836,
    "left_wrist_yaw_joint": 18.744836,
    "left_wrist_pitch_joint": 4.15,
    "left_wrist_roll_joint": 4.15,

    "right_shoulder_pitch_joint": 14.556046,
    "right_shoulder_roll_joint": 14.556046,
    "right_shoulder_yaw_joint": 18.744836,
    "right_elbow_joint": 18.744836,
    "right_wrist_yaw_joint": 18.744836,
    "right_wrist_pitch_joint": 4.15,
    "right_wrist_roll_joint": 4.15,
}


# =============================================================================
# PD gains
#
# From trajectory_generator.yaml supplied earlier.
# =============================================================================

X2_KP = 40.0
X2_KD = 10.0


# =============================================================================
# Actuators
#
# Grouped by real-robot effort limit.
#
# Keeping these as separate groups makes it easy to replace the simplified
# BuiltinPositionActuatorCfg with a more accurate X2 motor model later.
# =============================================================================

# 112.5 Nm ---------------------------------------------------------------

X2_STRONG_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint",
        ".*_knee_joint",
        "waist_yaw_joint",
    ),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=112.5,
)


# 53 Nm ------------------------------------------------------------------

X2_WAIST_PR_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_pitch_joint",
        "waist_roll_joint",
    ),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=53.0,
)


# 41 Nm ------------------------------------------------------------------

X2_MEDIUM_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_pitch_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
    ),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=41.0,
)


# 26.5 Nm ----------------------------------------------------------------

X2_SMALL_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_yaw_joint",
    ),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=26.5,
)


# 4.8 Nm -----------------------------------------------------------------

X2_WRIST_PR_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=4.8,
)


# Head -------------------------------------------------------------------

X2_HEAD_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("head_yaw_joint",),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=2.6,
)

X2_HEAD_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("head_pitch_joint",),
    stiffness=X2_KP,
    damping=X2_KD,
    effort_limit=0.6,
)


# =============================================================================
# Articulation
# =============================================================================

X2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        X2_STRONG_ACTUATOR,
        X2_WAIST_PR_ACTUATOR,
        X2_MEDIUM_ACTUATOR,
        X2_SMALL_ACTUATOR,
        X2_WRIST_PR_ACTUATOR,
        X2_HEAD_YAW_ACTUATOR,
        X2_HEAD_PITCH_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


# =============================================================================
# Initial state
# =============================================================================
#
# IMPORTANT:
# The base Z value has NOT yet been verified from the X2 model.
#
# 0.80 is only a temporary bootstrapping value.
# Before training, determine the correct pelvis/root height for the nominal
# configuration.
# =============================================================================

X2_INITIAL_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.80),
    joint_pos=X2_NOMINAL_JOINT_POS,
    joint_vel={".*": 0.0},
)


# =============================================================================
# Collision
# =============================================================================
#
# Do NOT copy the G1 foot collision regex blindly.
#
# Until the X2 geom names have been inspected, use a conservative generic
# collision configuration. Adjust this after checking x2.xml.
# =============================================================================

""" X2_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),
)
"""

# =============================================================================
# Robot configuration
# =============================================================================

X2_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=X2_INITIAL_STATE,
    articulation=X2_ARTICULATION,
)


def get_x2_robot_cfg() -> EntityCfg:
    """Return the X2 robot configuration."""
    return X2_CFG


# =============================================================================
# Action scale
#
# PAC-MAN / G1 uses approximately:
#
#     scale = 0.25 * effort_limit / stiffness
#
# We reproduce that rule here as a FIRST-PASS X2 action scale.
#
# This should be validated before real-robot deployment.
# =============================================================================


def _action_scale(joint_name: str) -> float:
    return 0.25 * X2_MAX_EFFORT[joint_name] / X2_KP


X2_ACTION_SCALE = {
    joint_name: _action_scale(joint_name)
    for joint_name in X2_JOINT_NAMES
}


# 29-DoF body-only action scale for the dodge policy.
X2_BODY_ACTION_SCALE = {
    joint_name: X2_ACTION_SCALE[joint_name]
    for joint_name in X2_BODY_JOINT_NAMES
}


# =============================================================================
# Sanity checks
# =============================================================================

assert len(X2_NOMINAL_JOINT_POS) == 31
assert len(X2_MAX_EFFORT) == 31
assert len(X2_MAX_VELOCITY) == 31
assert len(X2_ACTION_SCALE) == 31
assert len(X2_BODY_ACTION_SCALE) == 29


if __name__ == "__main__":
    print("X2 XML:", X2_XML)
    print("X2 joints:", len(X2_JOINT_NAMES))
    print("X2 body joints:", len(X2_BODY_JOINT_NAMES))

    print("\n=== X2 action scale ===")

    for joint_name in X2_JOINT_NAMES:
        print(
            f"{joint_name:32s} "
            f"effort={X2_MAX_EFFORT[joint_name]:7.2f} "
            f"vel={X2_MAX_VELOCITY[joint_name]:9.4f} "
            f"scale={X2_ACTION_SCALE[joint_name]:8.5f}"
        )