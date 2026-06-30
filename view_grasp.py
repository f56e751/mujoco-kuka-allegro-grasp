"""Show the Kuka+Allegro holding the cube in MuJoCo, posed from a harvested grasp.

The grasp finger/arm angles come from `allegro_pinch_states.npz` (states harvested from the
pretrained IsaacLab Lift policy). The cube is placed between the thumb & index fingertips
(frame-agnostic) so it reads as "grasped" regardless of small model-convention differences,
and the table is moved just under the grasp so the scene stays coherent.

Usage:
  .venv/bin/python view_grasp.py                 # render a PNG (headless)
  .venv/bin/python view_grasp.py --idx 7         # pick a different harvested grasp
  .venv/bin/python view_grasp.py --view          # interactive viewer (needs a display)
"""
import os
import argparse

os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen GL

import numpy as np
import mujoco

from grasp_scene import build_model, FINGER_MAP

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST = os.environ.get("GRASP_NPZ", os.path.join(HERE, "data", "grasp_states.npz"))


def _qadr(m, jname):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return int(m.jnt_qposadr[jid]) if jid >= 0 else None


def _cube_qadr(m):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    jadr = int(m.body_jntadr[bid])           # cube's (single, free) joint
    return int(m.jnt_qposadr[jadr])


def _ik_palm(m, d, target, body="rh_palm", iters=400, lam=0.2):
    """Damped least-squares position IK on the 7 arm joints -> put the palm at `target`.

    (The harvested arm angles are for IsaacLab's iiwa7; this scene uses Menagerie's iiwa14,
    a different arm, so we re-solve the arm to hold the hand over the table rather than reuse
    those angles. Finger angles DO transfer and are applied as-is.)
    """
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    arm = [(_qadr(m, f"joint{k}"), int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}")]))
           for k in range(1, 8)]
    qadr = np.array([a for a, _ in arm]); dofs = np.array([dd for _, dd in arm])
    # a reach-forward starting pose (elbow up) so IK converges to a hand-over-table solution
    start = [0.0, 0.6, 0.0, -1.4, 0.0, 1.2, 0.0]
    for q, s in zip(qadr, start):
        d.qpos[q] = s
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target - d.xpos[bid]
        if np.linalg.norm(err) < 1e-3:
            break
        mujoco.mj_jacBody(m, d, jacp, None, bid)
        J = jacp[:, dofs]                                   # 3x7
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(3), err)
        d.qpos[qadr] += np.clip(dq, -0.2, 0.2)
        lo = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}") for k in range(1, 8)]]
        d.qpos[qadr] = np.clip(d.qpos[qadr], lo[:, 0], lo[:, 1])


def apply_grasp(m, d, idx):
    npz = np.load(HARVEST)
    names = [str(x) for x in npz["joint_names"]]
    val = dict(zip(names, npz["joint_pos"][idx]))

    # fingers: IsaacLab {finger}_joint_J -> Menagerie rh_{ff/mf/rf/th}j{J}  (transfers directly)
    for finger, mj in FINGER_MAP.items():
        for J in range(4):
            a = _qadr(m, f"rh_{mj}j{J}")
            if a is not None and f"{finger}_joint_{J}" in val:
                d.qpos[a] = val[f"{finger}_joint_{J}"]

    # arm: re-solve so the hand sits over the table area (iiwa14 != IsaacLab iiwa7)
    table_x, table_y = 0.55, 0.0
    _ik_palm(m, d, np.array([table_x, table_y, 0.62]))
    mujoco.mj_forward(m, d)

    # place the cube between the thumb & index fingertips (frame-agnostic "in the grasp")
    tip = {f: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"rh_{mj}_tip")].copy()
           for f, mj in FINGER_MAP.items()}
    center = 0.5 * (tip["thumb"] + tip["index"])
    ca = _cube_qadr(m)
    d.qpos[ca:ca + 3] = center
    d.qpos[ca + 3:ca + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0

    # slide the (static) table to sit ~12 cm under the held cube so the scene reads coherently
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    half_h = float(m.geom_size[m.body_geomadr[tb]][2])
    m.body_pos[tb] = [center[0], center[1], center[2] - 0.12 - half_h]
    mujoco.mj_forward(m, d)
    return center


def apply_grasp_isaaclab(m, d, idx):
    """Pose the EXACT IsaacLab model. Harvested joint names match the extracted joints
    1:1 (iiwa7_joint_*, {finger}_joint_*), so apply directly -- arm too (real iiwa7).
    Cube placed between the index & thumb fingertips (biotac tip bodies)."""
    npz = np.load(HARVEST)
    val = dict(zip([str(x) for x in npz["joint_names"]], npz["joint_pos"][idx]))
    for jn, v in val.items():
        a = _qadr(m, jn)
        if a is not None:
            d.qpos[a] = v
    mujoco.mj_forward(m, d)
    bp = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
    center = 0.5 * (bp("index_biotac_tip") + bp("thumb_biotac_tip"))
    ca = _cube_qadr(m)
    d.qpos[ca:ca + 3] = center
    d.qpos[ca + 3:ca + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    half_h = float(m.geom_size[m.body_geomadr[tb]][2])
    m.body_pos[tb] = [center[0], center[1], center[2] - 0.12 - half_h]
    mujoco.mj_forward(m, d)
    return center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="harvested grasp index")
    ap.add_argument("--source", choices=["isaaclab", "menagerie"], default="isaaclab",
                    help="isaaclab = exact extracted kuka.usd geometry (default); menagerie = approximate")
    ap.add_argument("--view", action="store_true", help="open interactive viewer (needs display)")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "grasp.png"))
    ap.add_argument("--azimuth", type=float, default=None, help="camera azimuth deg (default: auto, faces the grasp)")
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--dist", type=float, default=0.34, help="camera distance (use ~1.0 to see the whole arm)")
    args = ap.parse_args()

    if args.source == "isaaclab":
        from grasp_scene_isaaclab import build_model as _build
        m = _build(); d = mujoco.MjData(m)
        c = apply_grasp_isaaclab(m, d, args.idx)
        palm_name = "palm_link"
    else:
        m = build_model(); d = mujoco.MjData(m)
        c = apply_grasp(m, d, args.idx)
        palm_name = "rh_palm"
    print(f"[view_grasp] source={args.source} grasp idx {args.idx}: cube held at {c.round(3)}")

    if args.view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(m, d) as v:
            while v.is_running():
                mujoco.mj_forward(m, d)
                v.sync()
        return

    # auto-aim: view the grasp in SIDE PROFILE (perpendicular to the palm->fingertips reach axis),
    # so the fingers wrapping the cube are visible for EVERY grasp regardless of hand orientation.
    cam = mujoco.MjvCamera()
    cam.lookat[:] = c
    cam.distance = args.dist
    cam.elevation = args.elevation
    if args.azimuth is None:
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        tips = (["index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip"]
                if args.source == "isaaclab" else ["rh_ff_tip", "rh_mf_tip", "rh_rf_tip", "rh_th_tip"])
        palm = d.xpos[bid(palm_name)]
        fcen = np.mean([d.xpos[bid(t)] for t in tips], axis=0)
        fwd = fcen - palm                                   # palm -> fingertips (reach axis)
        side = np.cross(fwd, [0, 0, 1.0])                   # horizontal, perpendicular to reach
        if np.linalg.norm(side) < 1e-6:
            cam.azimuth = 130.0
        else:
            side /= np.linalg.norm(side)
            cam.azimuth = float(np.degrees(np.arctan2(side[1], side[0])))
    else:
        cam.azimuth = args.azimuth
    r = mujoco.Renderer(m, height=900, width=1400)
    r.update_scene(d, camera=cam)
    from PIL import Image
    Image.fromarray(r.render()).save(args.out)
    print(f"[view_grasp] saved {args.out} (azimuth {cam.azimuth:.0f}, dist {args.dist})")


if __name__ == "__main__":
    main()
