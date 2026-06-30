"""Build a MuJoCo scene from the EXACT IsaacLab Kuka+Allegro geometry/kinematics.

Geometry + kinematics are extracted from IsaacLab's `kuka.usd` by `extract_kuka.py` into
`assets/isaaclab_kuka_allegro/` (per-link OBJ meshes + model.json). This reconstructs that
robot in MuJoCo (so the hand/fingertips match IsaacSim exactly), and adds a table + cube.
"""
import os
import json
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.join(HERE, "assets", "isaaclab_kuka_allegro")
CUBE_SIZE = 0.029
TABLE_TOP_Z = 0.40
TABLE_CENTER = (0.55, 0.0)

# hand links get a dark material (IsaacLab Allegro is black); iiwa links a neutral gray.
HAND_PREFIXES = ("palm", "index", "middle", "ring", "thumb", "allegro")


def build_spec() -> mujoco.MjSpec:
    model = json.load(open(os.path.join(RB, "model.json")))
    links = model["links"]
    by_name = {l["name"]: l for l in links}

    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    spec.meshdir = RB
    try:
        spec.visual.global_.offwidth = 1600
        spec.visual.global_.offheight = 1000
    except Exception:
        pass
    spec.add_material(name="handmat", rgba=[0.15, 0.15, 0.16, 1.0])
    spec.add_material(name="armmat", rgba=[0.62, 0.62, 0.64, 1.0])
    # meshes
    for l in links:
        if l["mesh"]:
            spec.add_mesh(name=l["name"], file=l["mesh"])

    wb = spec.worldbody
    for lp in ([0.4, 0.0, 1.8], [0.9, -0.7, 1.2], [-0.4, 0.6, 1.4]):
        wb.add_light(pos=lp, dir=[-lp[0], -lp[1], -1], diffuse=[0.6, 0.6, 0.6], specular=[0.2, 0.2, 0.2])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.05], rgba=[0.45, 0.45, 0.5, 1])

    # build body tree (parents before children)
    built = {}

    def ensure(name):
        if name in built:
            return built[name]
        l = by_name[name]
        par = l["parent"]
        parent_body = wb if par is None else ensure(par)
        b = parent_body.add_body(name=name, pos=l["pos"], quat=l["quat"])
        j = l["joint"]
        if j is not None:  # revolute -> hinge; fixed -> no joint (rigid)
            rng = [j["lo"] if j["lo"] is not None else -3.14,
                   j["hi"] if j["hi"] is not None else 3.14]
            b.add_joint(name=j["name"], type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=j["axis"], range=rng, pos=j["pos"])
        if l["mesh"]:
            mat = "handmat" if name.startswith(HAND_PREFIXES) else "armmat"
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, material=mat,
                       contype=0, conaffinity=0)
        built[name] = b
        return b

    for l in links:
        ensure(l["name"])

    # table + cube
    th = TABLE_TOP_Z / 2.0
    table = wb.add_body(name="table", pos=[TABLE_CENTER[0], TABLE_CENTER[1], th])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.28, 0.40, th], rgba=[0.55, 0.42, 0.30, 1])
    cube = wb.add_body(name="cube", pos=[TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z + CUBE_SIZE / 2])
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_SIZE / 2] * 3,
                  rgba=[0.85, 0.2, 0.2, 1], mass=0.05, condim=4, friction=[1.0, 0.05, 0.001])

    cam = wb.add_camera(name="demo")
    eye = np.array([TABLE_CENTER[0] + 0.9, -0.9, TABLE_TOP_Z + 0.55])
    tgt = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z + 0.05])
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.column_stack([right, up, -fwd])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    cam.pos = eye.tolist(); cam.quat = q.tolist()
    return spec


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    spec = build_spec()
    m = spec.compile()
    print(f"[isaaclab scene] compiled: {m.nbody} bodies, {m.njnt} joints, {m.nq} qpos")
    out = os.path.join(HERE, "kuka_allegro_isaaclab.xml")
    open(out, "w").write(spec.to_xml())
    mujoco.MjModel.from_xml_path(out)  # verify standalone
    print(f"[isaaclab scene] wrote + verified {out}")
