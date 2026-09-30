from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

# 改成你原 CSV converter 的真实模块名
from csv_to_x2_npz import (
    X2_JOINT_NAMES,
    retarget_g1_to_x2,
    build_x2_motion,
    qpos_address,
    geom_world_bottom_z,
    render_motion
)


def align_leap_root_z_constant(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    x2_joint_pos: np.ndarray,
    reference_frame: int = 0,
    ground_z: float = 0.0,
):
    """
    Determine one constant root-Z correction from a grounded frame.

    IMPORTANT:
    For leap/jump motions we must NOT align feet to ground frame-by-frame,
    otherwise the airborne trajectory gets destroyed.
    """

    data = mujoco.MjData(model)

    left_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "left_ankle_roll_link",
    )
    right_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "right_ankle_roll_link",
    )

    if left_body_id < 0 or right_body_id < 0:
        raise RuntimeError("Could not find X2 ankle roll bodies.")

    foot_body_ids = {
        left_body_id,
        right_body_id,
    }

    foot_geom_ids = []

    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])

        if body_id not in foot_body_ids:
            continue

        if (
            model.geom_contype[gid] == 0
            and model.geom_conaffinity[gid] == 0
        ):
            continue

        foot_geom_ids.append(gid)

    if not foot_geom_ids:
        raise RuntimeError("No X2 foot collision geoms found.")

    frame = int(reference_frame)

    if frame < 0 or frame >= len(root_pos):
        raise ValueError(
            f"Invalid reference frame {frame}; "
            f"motion contains {len(root_pos)} frames."
        )

    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = root_pos[frame]
    data.qpos[3:7] = root_quat[frame]

    for i, name in enumerate(X2_JOINT_NAMES):
        data.qpos[qpos_address(model, name)] = (
            x2_joint_pos[frame, i]
        )

    mujoco.mj_forward(model, data)

    min_foot_z = np.inf

    for gid in foot_geom_ids:
        bottom_z = geom_world_bottom_z(
            model,
            data,
            gid,
        )
        min_foot_z = min(min_foot_z, bottom_z)

    z_offset = ground_z - min_foot_z

    corrected = root_pos.copy()
    corrected[:, 2] += z_offset

    print()
    print("=== LEAP ROOT-Z ===")
    print(f"reference frame : {frame}")
    print(f"foot bottom z   : {min_foot_z:+.4f}")
    print(f"constant offset : {z_offset:+.4f}")
    print(
        f"input root-z    : "
        f"{root_pos[:, 2].min():+.4f} .. "
        f"{root_pos[:, 2].max():+.4f}"
    )
    print(
        f"output root-z   : "
        f"{corrected[:, 2].min():+.4f} .. "
        f"{corrected[:, 2].max():+.4f}"
    )
    print()

    return corrected


def convert_one(
    input_file: Path,
    output_file: Path,
    model: mujoco.MjModel,
    root_body_index: int = 0,
    root_z_reference_frame: int = 0,
):
    print("=" * 80)
    print(f"[INPUT] {input_file}")

    g1 = np.load(
        input_file,
        allow_pickle=True,
    )

    required = {
        "fps",
        "joint_pos",
        "body_pos_w",
        "body_quat_w",
    }

    missing = required - set(g1.files)

    if missing:
        raise RuntimeError(
            f"{input_file} missing keys: {sorted(missing)}"
        )

    fps = float(
        np.asarray(g1["fps"]).reshape(-1)[0]
    )

    g1_joint_pos = np.asarray(
        g1["joint_pos"],
        dtype=np.float64,
    )

    body_pos_w = np.asarray(
        g1["body_pos_w"],
        dtype=np.float64,
    )

    body_quat_w = np.asarray(
        g1["body_quat_w"],
        dtype=np.float64,
    )

    print(f"[INFO] fps           : {fps}")
    print(f"[INFO] joint_pos     : {g1_joint_pos.shape}")
    print(f"[INFO] body_pos_w    : {body_pos_w.shape}")
    print(f"[INFO] body_quat_w   : {body_quat_w.shape}")

    if g1_joint_pos.shape[1] != 29:
        raise RuntimeError(
            f"Expected G1 29 DoF, got "
            f"{g1_joint_pos.shape}"
        )

    # ------------------------------------------------------------
    # G1 pelvis/root trajectory
    #
    # For AMP files generated from data.xpos[1:] / data.xquat[1:],
    # index 0 corresponds to the first robot body after world.
    # Verify this is pelvis for your G1 model.
    # ------------------------------------------------------------

    root_pos = body_pos_w[:, root_body_index].copy()

    root_quat = body_quat_w[:, root_body_index].copy()

    # MuJoCo xquat format is:
    #
    #   [w, x, y, z]
    #
    # which is already exactly what data.qpos[3:7] expects.

    root_quat /= np.maximum(
        np.linalg.norm(
            root_quat,
            axis=1,
            keepdims=True,
        ),
        1e-8,
    )

    print(
        f"[INFO] G1 root-z range: "
        f"{root_pos[:, 2].min():.4f} .. "
        f"{root_pos[:, 2].max():.4f}"
    )

    # ------------------------------------------------------------
    # G1 29 DoF -> X2 31 DoF.
    # Uses your already validated mapping.
    # ------------------------------------------------------------

    x2_joint_pos = retarget_g1_to_x2(
        model,
        g1_joint_pos,
        clamp=True,
    )

    print(
        f"[INFO] X2 joint_pos: "
        f"{x2_joint_pos.shape}"
    )

    # ------------------------------------------------------------
    # LEAP-specific root Z correction.
    #
    # One constant offset ONLY.
    # Preserve original vertical jump trajectory.
    # ------------------------------------------------------------

    root_pos = align_leap_root_z_constant(
        model,
        root_pos,
        root_quat,
        x2_joint_pos,
        reference_frame=root_z_reference_frame,
        ground_z=0.0,
    )

    # ------------------------------------------------------------
    # Rebuild ALL AMP body states using X2 FK.
    #
    # Ignore G1 joint_vel / body velocities.
    # ------------------------------------------------------------

    x2_motion = build_x2_motion(
        model,
        root_pos,
        root_quat,
        x2_joint_pos,
        fps,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        **x2_motion,
    )

    print(f"[OK] {output_file}")

    for key, value in x2_motion.items():
        print(
            f"  {key:20s} "
            f"{value.shape} "
            f"{value.dtype}"
        )
        
    return x2_motion, root_pos, root_quat


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path(
            "src/assets/robots/"
            "lx2501_3_t2d5/xmls/x2_mjlab.xml"
        ),
    )

    parser.add_argument(
        "--root-body-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--root-z-reference-frame",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--render",
        action="store_true",
    )

    args = parser.parse_args()

    print(f"[INFO] Loading X2 MJCF: {args.mjcf}")

    model = mujoco.MjModel.from_xml_path(
        str(args.mjcf)
    )

    x2_motion, root_pos, root_quat = convert_one(
        input_file=args.input_file,
        output_file=args.output_file,
        model=model,
        root_body_index=args.root_body_index,
        root_z_reference_frame=args.root_z_reference_frame,
    )
    if args.render:
        render_motion(
            model,
            x2_motion,
            root_pos,
            root_quat,
        )


if __name__ == "__main__":
    main()