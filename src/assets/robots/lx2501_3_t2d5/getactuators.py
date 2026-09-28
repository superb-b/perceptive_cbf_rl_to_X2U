import mujoco

m = mujoco.MjModel.from_xml_path("xmls/x2.xml")

print("=== ACTUATOR PARAMETERS ===")

for i in range(m.nu):
    name = mujoco.mj_id2name(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, i
    )

    print(f"\n[{i}] {name}")

    print("ctrlrange:",
          m.actuator_ctrlrange[i])

    print("forcerange:",
          m.actuator_forcerange[i])

    print("gear:",
          m.actuator_gear[i])

    print("gainprm:",
          m.actuator_gainprm[i])

    print("biasprm:",
          m.actuator_biasprm[i])