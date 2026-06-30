"""Build a MuJoCo scene: Kuka iiwa14 + Allegro (right) hand + table + 2.9cm cube.

This is a *visualization* scene for the grasp demo. The Allegro hand is attached to
the iiwa flange; a table and a free-floating cube are added. The grasp pose itself is
applied by `view_grasp.py` from the harvested grasp states.

Assets are MuJoCo Menagerie (kuka_iiwa_14, wonik_allegro), vendored under ./assets/.
"""
import os
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
CUBE_SIZE = 0.029          # m, edge length (matches the IsaacLab in-hand cube)
TABLE_TOP_Z = 0.40         # m, table surface height
TABLE_CENTER = (0.55, 0.0)  # x,y of the table/grasp area in front of the arm

# Allegro finger joints, Menagerie naming, in the SAME order as the IsaacLab harvest
# (index/middle/ring/thumb, joint 0..3). IsaacLab "X_joint_J" -> Menagerie "{ff/mf/rf/th}j{J}".
FINGER_MAP = {"index": "ff", "middle": "mf", "ring": "rf", "thumb": "th"}
ALLEGRO_JOINTS_MJ = [f"rh_{FINGER_MAP[f]}j{j}" for j in range(4) for f in ("index", "middle", "ring", "thumb")]
# ^ order: ffj0,mfj0,rfj0,thj0, ffj1,... matches IsaacLab [X_joint_0 x4, X_joint_1 x4, ...]
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
PALM_BODY = "rh_palm"


def build_spec() -> mujoco.MjSpec:
    arm = mujoco.MjSpec.from_file(os.path.join(ASSETS, "kuka_iiwa_14", "iiwa14.xml"))
    hand = mujoco.MjSpec.from_file(os.path.join(ASSETS, "wonik_allegro", "right_hand.xml"))

    # attach the hand (base body 'palm') to the iiwa flange site, prefix 'rh_'
    site = arm.site("attachment_site")
    site.attach_body(hand.body("palm"), "rh_", "")

    # larger offscreen framebuffer so we can render high-res PNGs
    try:
        arm.visual.global_.offwidth = 1600
        arm.visual.global_.offheight = 1000
    except Exception:
        pass

    wb = arm.worldbody
    # lighting + floor
    wb.add_light(pos=[0.4, 0.0, 1.8], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.05],
                rgba=[0.3, 0.3, 0.35, 1.0])

    # table: static box, top surface at TABLE_TOP_Z
    th = TABLE_TOP_Z / 2.0
    table = wb.add_body(name="table", pos=[TABLE_CENTER[0], TABLE_CENTER[1], th])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.28, 0.40, th],
                   rgba=[0.55, 0.42, 0.30, 1.0])

    # cube: free body, starts resting on the table
    cube = wb.add_body(name="cube", pos=[TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z + CUBE_SIZE / 2])
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_SIZE / 2] * 3,
                  rgba=[0.85, 0.2, 0.2, 1.0], mass=0.05, condim=4, friction=[1.0, 0.05, 0.001])

    # a fixed demo camera looking at the grasp area
    cam = wb.add_camera(name="demo")
    eye = np.array([TABLE_CENTER[0] + 0.9, -0.9, TABLE_TOP_Z + 0.55])
    tgt = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z + 0.05])
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.column_stack([right, up, -fwd])          # camera axes (x,y,z) in world; cam looks along -z
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    cam.pos = eye.tolist()
    cam.quat = q.tolist()
    return arm


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    spec = build_spec()
    model = spec.compile()
    print(f"[grasp_scene] compiled OK: {model.nbody} bodies, {model.njnt} joints, {model.nq} qpos")
    have_arm = [j for j in ARM_JOINTS if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0]
    have_fing = [j for j in ALLEGRO_JOINTS_MJ if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0]
    print(f"[grasp_scene] arm joints found {len(have_arm)}/7, finger joints found {len(have_fing)}/16")
    print(f"[grasp_scene] palm body present: {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY) >= 0}")
    # export a standalone combined XML. to_xml writes mesh basenames with meshdir="assets",
    # so flatten both source models' meshes into assets/ (no name collisions: allegro=.stl, iiwa=.obj).
    import glob
    import shutil
    for src in (os.path.join(ASSETS, "wonik_allegro", "assets"),
                os.path.join(ASSETS, "kuka_iiwa_14", "assets")):
        for f in glob.glob(os.path.join(src, "*")):
            shutil.copy(f, ASSETS)
    out = os.path.join(HERE, "kuka_allegro_cube.xml")
    try:
        spec.compile()
        with open(out, "w") as f:
            f.write(spec.to_xml())
        # sanity: load it back standalone
        mujoco.MjModel.from_xml_path(out)
        print(f"[grasp_scene] wrote + verified standalone XML -> {out}")
    except Exception as e:
        print(f"[grasp_scene] to_xml skipped ({e})")
