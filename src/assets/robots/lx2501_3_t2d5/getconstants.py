import mujoco

m = mujoco.MjModel.from_xml_path("xmls/x2_mjlab.xml")

print("=== GEOM -> BODY ===")

for gid in range(m.ngeom):
    bid = m.geom_bodyid[gid]

    body_name = mujoco.mj_id2name(
        m,
        mujoco.mjtObj.mjOBJ_BODY,
        bid,
    )

    print(
        f"geom={gid:2d}",
        f"body={bid:2d}",
        f"{body_name:30s}",
        f"contype={m.geom_contype[gid]}",
        f"conaffinity={m.geom_conaffinity[gid]}",
    )