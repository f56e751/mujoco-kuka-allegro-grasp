"""Extract IsaacLab Kuka+Allegro (kuka.usd) geometry + kinematics -> OBJ meshes + model.json.

Run INSIDE the isaac-sim container:
  ./isaaclab.sh -p /workspace/mujoco_grasp_demo/extract_kuka.py
"""
import argparse, json, os
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
a, _ = ap.parse_known_args(["--headless"]); app = AppLauncher(a).app

from pxr import Usd, UsdGeom, UsdPhysics, Gf
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

OUT = "/workspace/mujoco_grasp_demo/assets/isaaclab_kuka_allegro"
os.makedirs(OUT, exist_ok=True)
url = f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd"
stage = Usd.Stage.Open(url)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
print(f"opened={stage is not None} metersPerUnit={mpu}", flush=True)
T = Usd.TimeCode.Default()


def w(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(T)


def quat_wxyz(m):
    q = m.ExtractRotationQuat().GetNormalized()
    im = q.GetImaginary()
    return [q.GetReal(), im[0], im[1], im[2]]


# ---- rigid bodies ----
bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
bpaths = {p.GetPath().pathString for p in bodies}
print(f"rigid bodies: {len(bodies)}", flush=True)

# ---- joints: build parent(child) + joint info ----
parent = {}      # child_path -> parent_path
jinfo = {}       # child_path -> {axis,pos,quat,lo,hi,name} (None for fixed)
for p in stage.Traverse():
    if not p.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(p)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    if not b1:
        continue
    c = b1[0].pathString
    par = b0[0].pathString if b0 else None
    parent[c] = par
    if p.IsA(UsdPhysics.RevoluteJoint):
        rj = UsdPhysics.RevoluteJoint(p)
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[rj.GetAxisAttr().Get()]
        lr1 = j.GetLocalRot1Attr().Get()
        lp1 = j.GetLocalPos1Attr().Get()
        R = Gf.Matrix4d().SetRotate(Gf.Quatd(lr1.GetReal(), Gf.Vec3d(*lr1.GetImaginary())))
        axw = R.TransformDir(Gf.Vec3d(*ax))
        import math
        lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
        d2r = math.pi / 180.0
        jinfo[c] = {"name": p.GetName(), "axis": [axw[0], axw[1], axw[2]],
                    "pos": [lp1[0] * mpu, lp1[1] * mpu, lp1[2] * mpu],
                    "lo": (lo * d2r if lo is not None else None),
                    "hi": (hi * d2r if hi is not None else None)}
    else:
        jinfo[c] = None  # fixed

# ---- per-body: local transform (rel parent), mesh export ----
links = []
for p in bodies:
    path = p.GetPath().pathString
    name = p.GetName()
    par = parent.get(path, None)
    par_w = w(stage.GetPrimAtPath(par)) if (par and par in bpaths) else Gf.Matrix4d(1.0)
    loc = w(p) * par_w.GetInverse()
    pos = loc.ExtractTranslation()
    rec = {"name": name, "path": path,
           "parent": (stage.GetPrimAtPath(par).GetName() if (par and par in bpaths) else None),
           "pos": [pos[0] * mpu, pos[1] * mpu, pos[2] * mpu],
           "quat": quat_wxyz(loc), "mesh": None, "joint": jinfo.get(path, None)}
    # find mesh prim(s) under this body (instance proxies)
    rng = Usd.PrimRange(p, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    body_w_inv = w(p).GetInverse()
    verts, faces = [], []
    for mp in rng:
        if not mp.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(mp)
        pts = mesh.GetPointsAttr().Get()
        cnts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not cnts:
            continue
        m2b = w(mp) * body_w_inv
        base = len(verts)
        for v in pts:
            pv = m2b.Transform(Gf.Vec3d(v[0], v[1], v[2]))
            verts.append((pv[0] * mpu, pv[1] * mpu, pv[2] * mpu))
        k = 0
        for c in cnts:
            f = [base + idx[k + i] + 1 for i in range(c)]
            for i in range(1, c - 1):  # fan triangulation
                faces.append((f[0], f[i], f[i + 1]))
            k += c
    if verts:
        fn = f"{name}.obj"
        with open(os.path.join(OUT, fn), "w") as fo:
            for v in verts:
                fo.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for f in faces:
                fo.write(f"f {f[0]} {f[1]} {f[2]}\n")
        rec["mesh"] = fn
    links.append(rec)

json.dump({"metersPerUnit": mpu, "links": links},
          open(os.path.join(OUT, "model.json"), "w"), indent=1)
print(f"wrote {len([l for l in links if l['mesh']])} meshes + model.json to {OUT}", flush=True)
for l in links:
    print(f"  {l['name']:20s} parent={str(l['parent']):16s} mesh={l['mesh']} joint={'fixed' if l['joint'] is None else l['joint']['name'] if l['joint'] else None}", flush=True)
app.close()
