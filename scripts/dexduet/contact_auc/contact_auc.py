# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Contact-AUC scorer of the V2D study (reproduce/contact_auc.sh runs it through contact_auc_surface.py).

It is DexDuet's Slip-AUC tool, dexmachina/eval/slip_auc.py (tool version 2, github.com/HaoranZhangumich/DexDuet), ported
to the V2D recordings: the parts tuple and the frame shift are parameters (configure()), and V2D loaders and a per-episode
V2D driver (v2d_main) are added. Lines changed for V2D carry "# V2D EDIT"; every scoring function is unchanged. The
original tool's main() (DexMachina recordings) is kept as it was; reproduce/ runs only v2d_main.

Slip-AUC: does the robot hand hold the object where the human held it, without its contacts sliding? (offline, CPU, numpy)

Scores a recording made by dexmachina/eval/slip_record.py (object state and both hands' keypoints for every evaluation lane
and step) against the human demonstration. Only recorded trajectories are used, no simulator: like ADD-AUC it can score
rollouts from any simulator once they are recorded in the same format.

Definition (defaults; every constant is a parameter of the original tool's main(), which this file's
command line does not expose):
  * Entries: (hand h, part p, step t, lane) where the HUMAN hand h holds part p over the window [t, t+w] (w = 0.1 s):
    some MANO vertex of hand h is within 5 mm of p at t and at t+w, and the MANO vertices touching p at t move < 1 cm (mean)
    in p's own frame by t+w. Contacts belong to the nearest part.
  * Robot slip of an entry: the robot hand's keypoint links within 1.5 cm of part p at t are its contacts; slip = their mean
    displacement in p's own frame from t to t+w (cm). No robot contact on p, a lost object (mean ADD point error > 5 cm) or a
    broken state counts as maximal slip.
  * Score per entry = AUC of 1[slip < th] over th = 0.5, 1.0, ..., 5.0 cm (the ADD-AUC construction). Slip-AUC = mean over all
    entries; 0-1, higher = better, a rigid grip that touches the held part scores 1.0.
  * Reported next to it: coverage (share of entries where the robot touches the part and the object is not lost), slip score
    when touching, lost share, per hand / part, controls (hand-only, object-only, human reference), lanes, a moving-block
    bootstrap CI over steps, and the sensitivity to every constant.
  * It measures whether the grip is stable, not whether it is human-like: one contact that does not slide gets full credit.

On the DexMachina suite (the original tool's main(), not used here), the human data is the per-sequence MANO export of
dexmachina/hand_proc/export_mano_vertices.py, and
`python -m dexmachina.eval.slip_auc <recording dir> --mano <export dir>` writes <recording dir>/slip_auc.json and
slip_auc_frames.npz. On V2D, reproduce/contact_auc.sh runs contact_auc_surface.py, which calls v2d_main() with the SOMA
hands of reproduce/tools/human_soma.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SIDES = ('left', 'right')
PARTS = ('bottom', 'top')  # V2D EDIT: parameter, set by configure()
SHIFT = 1  # V2D EDIT: frame shift, +1 DexMachina (demo[i+1]), 0 V2D (frame k)
DEFAULTS = dict(window_s=0.1, human_touch_m=0.005, human_slip_cm=1.0, robot_touch_m=0.015, th_lo_cm=0.5, th_hi_cm=5.0,
                th_n=10, lost_m=0.05)
MOVING_CM = 1.0                     # sensitivity variant 'moving part only': part moves > 1 cm (mean point) in the world per window
BOOT_N, BOOT_BLOCK, BOOT_SEED = 5000, 20, 0
SURFACE_SPACING_M = 0.001           # dense surface samples for the human-object distance (~1 mm apart)
TOOL_VERSION = 2


# ------------------------------------------------------------------------------------------------------------------------------
# geometry and statistics


def rotmats(quat):
    """wxyz (..., 4) -> (..., 3, 3)."""
    q = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat(np.roll(q.reshape(-1, 4), -1, axis=1)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def part_poses(states):
    """states (..., 8) [pos, wxyz, arti] -> {part: (R (..., 3, 3), t (..., 3))}; top = bottom * Rz(-arti)."""
    st = np.asarray(states, dtype=np.float64)
    rb = rotmats(st[..., 3:7])
    if PARTS == ('object',):  # V2D EDIT: one rigid part
        return {'object': (rb, st[..., :3])}
    a = st[..., 7]
    rz = Rotation.from_rotvec(np.outer(-a.reshape(-1), [0, 0, 1])).as_matrix().reshape(*a.shape, 3, 3)
    return {'bottom': (rb, st[..., :3]), 'top': (rb @ rz, st[..., :3])}


def state_ok(states):
    st = np.asarray(states)
    return np.isfinite(st).all(-1) & (np.linalg.norm(np.nan_to_num(st[..., 3:7]), axis=-1) > 1e-6)


def auc_scores(err_cm, ths):
    """Per-entry AUC of 1[err < th] over the thresholds (x in [0, 1]); its mean is the AUC of the pooled accuracy curve."""
    x = np.linspace(0, 1, len(ths))
    ind = (np.asarray(err_cm)[..., None] < np.asarray(ths)).astype(np.float64)
    return np.trapz(ind, x=x, axis=-1)


def add_point_error(states, demo, pts):
    """Mean per-part point error (m) of states (S, L, 8) vs demo (S, 8) aligned per step -> (2, S, L); broken states -> inf."""
    ok = state_ok(states)
    st = np.where(ok[..., None], states, np.array([0, 0, 0, 1, 0, 0, 0, 0.]))
    rp, dp = part_poses(st), part_poses(demo)
    out = np.zeros((len(PARTS),) + st.shape[:2])  # V2D EDIT: number of parts
    for k, part in enumerate(PARTS):
        R, t = rp[part]
        Rd, td = dp[part]
        world_d = np.einsum('sij,pj->spi', Rd, pts[part]) + td[:, None]
        for l in range(st.shape[1]):
            world = np.einsum('sij,pj->spi', R[:, l], pts[part]) + t[:, l, None]
            out[k, :, l] = np.linalg.norm(world - world_d, axis=-1).mean(-1)
    out[:, ~ok] = np.inf
    return out


def surface_trees(mesh_files):
    """KD-tree per part over the mesh vertices + dense surface samples (~1 mm apart), in the part frame; also the
    discretisation error (distance from fresh surface samples to the cloud, 99th percentile) for the record."""
    import trimesh
    trees, disc = {}, {}
    for part in PARTS:
        m = trimesh.load(mesh_files[part], force='mesh')
        n = int(min(2_000_000, max(50_000, m.area / SURFACE_SPACING_M ** 2)))
        pts = np.concatenate([np.asarray(m.vertices), trimesh.sample.sample_surface(m, n, seed=0)[0]])
        trees[part] = cKDTree(pts)
        probe = trimesh.sample.sample_surface(m, 20000, seed=1)[0]
        disc[part] = float(np.percentile(trees[part].query(probe)[0], 99))
    return trees, disc


def boot_index(n_items, block=BOOT_BLOCK, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    if n_items == 0:
        return np.zeros((n, 0), int)
    b = min(block, n_items)
    nb = int(math.ceil(n_items / b))
    starts = rng.integers(0, n_items - b + 1, size=(n, nb))
    return (starts[:, :, None] + np.arange(b)).reshape(n, -1)[:, :n_items]


def ci(samples):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    return [None, None] if len(s) == 0 else [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def official_add(rec):
    """ADD-AUC recomputed from the recording, compute_add.py convention (row i vs demo frame i, incl. the final step)."""
    pts = {'bottom': rec['add_points_bottom'], 'top': rec['add_points_top']}
    rows = rec['obj_state'] if rec['final_state'].size == 0 else np.concatenate([rec['obj_state'], rec['final_state'][None]])
    rows = rows.astype(np.float64)
    err = add_point_error(rows, rec['demo'][:len(rows)], pts)
    ths = np.arange(0.01, 0.1, 0.01)
    return float(np.mean([np.trapz([np.mean(e < th) for th in ths], x=np.linspace(0, 1, len(ths))) for e in err]))


# ------------------------------------------------------------------------------------------------------------------------------
# Slip-AUC


def to_part_frames(points, states, trees):
    """points (..., K, 3) world, states (..., 8) -> {part: (local (..., K, 3), dist (..., K))}: the points in each part's own frame
    and their distance to that part's surface. Broken states (non-finite / zero quaternion) are replaced by identity; callers
    mask them."""
    st = np.asarray(states, np.float64)
    ok = state_ok(st)
    st = np.where(ok[..., None], st, np.array([0, 0, 0, 1, 0, 0, 0, 0.]))
    pts = np.nan_to_num(np.asarray(points, np.float64), nan=1e3, posinf=1e3, neginf=-1e3)
    poses = part_poses(st)
    out = {}
    for part in PARTS:
        Rm, t = poses[part]
        local = np.einsum('...ji,...kj->...ki', Rm, pts - t[..., None, :])
        dist = trees[part].query(local.reshape(-1, 3), workers=-1)[0].reshape(local.shape[:-1])
        out[part] = (local, dist)
    return out


def contacts(frames, part, touch_m):
    """frames = to_part_frames(...) output; -> (..., K) bool: the point is within touch_m of `part` AND `part` is its nearest
    part (a point on the ketchup cap is also < 1 cm from the bottle; it belongs to the cap)."""
    d = frames[part][1]  # V2D EDIT: nearest part among PARTS (vacuous with one part)
    near = d < touch_m
    for other in PARTS:
        if other != part:
            near = near & (d <= frames[other][1])
    return near


def window_slip(local, near_all, w):
    """local (S, ..., K, 3), near_all (S, ..., K) contact mask -> slip (S-w, ...) cm: mean displacement in the part frame from
    t to t+w of the points in contact at t; inf where no point touches at t."""
    near = near_all[:-w]
    disp = np.linalg.norm(local[w:] - local[:-w], axis=-1) * 100.0
    n = near.sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        slip = (disp * near).sum(-1) / n
    return np.where(n > 0, slip, np.inf), n


def part_world_motion(demo_rows, pts, w):
    """(S-w,) per part: mean ADD-point displacement of the part in the world over the window (cm), demo pose."""
    poses = part_poses(demo_rows)
    out = {}
    for part in PARTS:
        Rm, t = poses[part]
        world = np.einsum('sij,pj->spi', Rm, pts[part]) + t[:, None]
        out[part] = np.linalg.norm(world[w:] - world[:-w], axis=-1).mean(-1) * 100.0
    return out


class Clip:
    """Human side of one clip (demo object track + MANO meshes), precomputed once: per hand and part the MANO vertices in the
    part frame and their surface distance, per demo frame."""

    def __init__(self, demo, human_verts, pts, trees):
        self.demo, self.pts, self.trees = demo, pts, trees
        self.h = {s: to_part_frames(np.asarray(human_verts[s], np.float64), demo, trees) for s in SIDES}

    def gate(self, S, w, touch_m, slip_cm):
        """(side, part) -> dict(touch, slip_cm, hold) over replay windows t = 0..S-w-1 (demo frames t+1 -> t+1+w)."""
        f0 = np.arange(SHIFT, S + SHIFT - w)  # V2D EDIT: frame shift
        out = {}
        for s in SIDES:
            for p in PARTS:
                local = self.h[s][p][0]
                con = contacts(self.h[s], p, touch_m)
                c0, c1 = con[f0], con[f0 + w]
                disp = np.linalg.norm(local[f0 + w] - local[f0], axis=-1) * 100.0
                n = c0.sum(-1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    hs = np.where(n > 0, (disp * c0).sum(-1) / n, np.inf)
                touch = c0.any(-1) & c1.any(-1)
                out[s, p] = dict(touch=touch, slip=hs, hold=touch & (hs < slip_cm))
        return out


def score(gate, slip, lost_pair, ths, key='hold', moving=None):
    """gate (side, part) -> dict; slip (side, part) -> (S-w, L) cm (inf = no contact / broken); lost_pair (S-w, L).
    Returns summary + per-step sums (num, cnt) for the bootstrap + per-entry score arrays."""
    Sm, L = lost_pair.shape
    num, cnt = np.zeros(Sm), np.zeros(Sm)
    res = dict(per_hand={}, per_part={}, per_hand_part={})
    acc = {k: [0.0, 0] for k in ('touched', 'touched_score')}
    lost_n = 0
    sc_all = {}
    tot_h = {s: [0.0, 0] for s in SIDES}
    tot_p = {p: [0.0, 0] for p in PARTS}
    for s in SIDES:
        res['per_hand_part'][s] = {}
        for p in PARTS:
            m = gate[s, p][key] if moving is None else gate[s, p][key] & moving[p]
            x = slip[s, p]
            sc = np.where(lost_pair, 0.0, auc_scores(x, ths))
            sc_all[s, p] = sc
            v = sc[m]
            res['per_hand_part'][s][p] = float(v.mean()) if v.size else None
            num += np.where(m, sc.sum(1), 0.0)
            cnt += m * L
            tot_h[s][0] += float(v.sum()); tot_h[s][1] += v.size
            tot_p[p][0] += float(v.sum()); tot_p[p][1] += v.size
            m2 = np.broadcast_to(m[:, None], sc.shape)
            touched = np.isfinite(x) & ~lost_pair
            acc['touched'][0] += float((touched & m2).sum())
            acc['touched'][1] += int(m2.sum())
            ts = sc[m2 & touched]
            acc['touched_score'][0] += float(ts.sum())
            acc['touched_score'][1] += int(ts.size)
            lost_n += int((lost_pair & m2).sum())
    for s in SIDES:
        res['per_hand'][s] = tot_h[s][0] / tot_h[s][1] if tot_h[s][1] else None
    for p in PARTS:
        res['per_part'][p] = tot_p[p][0] / tot_p[p][1] if tot_p[p][1] else None
    E = int(cnt.sum())
    res['slip_auc'] = float(num.sum() / E) if E else None
    res['entries'] = E
    res['entries_per_lane'] = E // L if L else 0
    res['coverage'] = acc['touched'][0] / acc['touched'][1] if acc['touched'][1] else None
    res['slip_score_when_touching'] = acc['touched_score'][0] / acc['touched_score'][1] if acc['touched_score'][1] else None
    res['lost_share'] = lost_n / E if E else None
    return res, num, cnt, sc_all


def lost_pairs(states, demo_rows, pts, lost_m, w):
    if lost_m is None:
        ok = state_ok(states)
        bad = ~ok
    else:
        bad = add_point_error(states, demo_rows, pts).mean(0) > lost_m          # broken states give inf -> lost
    return bad[w:] | bad[:-w], bad


class Run:
    """Robot side of one recording, precomputed once: hand points in each part frame of the ROBOT object (and of the DEMO object,
    for the hand-only control)."""

    def __init__(self, clip, states, hand_points, demo_rows):
        self.clip, self.states, self.demo_rows = clip, states, demo_rows
        S, L = states.shape[:2]
        self.S, self.L = S, L
        self.kfinite = {s: np.isfinite(hand_points[s]).all((-1, -2)) for s in SIDES}
        self.robot = {s: to_part_frames(hand_points[s], states, clip.trees) for s in SIDES}
        demo_l = np.broadcast_to(demo_rows[:, None], states.shape)
        self.hand_only = {s: to_part_frames(hand_points[s], demo_l, clip.trees) for s in SIDES}

    def slips(self, which, w, touch_m):
        src = self.robot if which == 'robot' else self.hand_only
        out = {}
        for s in SIDES:
            kf = self.kfinite[s]
            bad = ~(kf[w:] & kf[:-w])
            for p in PARTS:
                sl, _n = window_slip(src[s][p][0], contacts(src[s], p, touch_m), w)
                out[s, p] = np.where(bad, np.inf, sl)
        return out

    def evaluate(self, window_s=0.1, human_touch_m=0.005, human_slip_cm=1.0, robot_touch_m=0.015, th_lo_cm=0.5, th_hi_cm=5.0,
                 th_n=10, lost_m=0.05, dt=1 / 60, which='robot', key='hold', moving_only=False, with_frames=False):
        w = int(round(window_s / dt))
        gate = self.clip.gate(self.S, w, human_touch_m, human_slip_cm)
        slip = self.slips(which, w, robot_touch_m)
        if which == 'hand_only':
            lost_pair = np.zeros((self.S - w, self.L), bool)
        else:
            lost_pair, _ = lost_pairs(self.states, self.demo_rows, self.clip.pts, lost_m, w)
        ths = np.linspace(th_lo_cm, th_hi_cm, th_n)
        moving = part_world_motion(self.demo_rows, self.clip.pts, w) if moving_only else None
        if moving is not None:
            moving = {p: v > MOVING_CM for p, v in moving.items()}
        res, num, cnt, sc = score(gate, slip, lost_pair, ths, key=key, moving=moving)
        allslip = np.concatenate([np.where(lost_pair, np.inf, slip[s, p])[gate[s, p][key]].ravel() for s in SIDES for p in PARTS])
        fin = allslip[np.isfinite(allslip)]
        res.update(window_steps=w, thresholds_cm=[float(x) for x in ths],
                   median_slip_cm_when_touching=float(np.median(fin)) if fin.size else None)
        if with_frames:
            return res, dict(num=num, cnt=cnt, gate=gate, slip=slip, lost_pair=lost_pair, sc=sc)
        return res


def human_reference(clip, human_joints, S, dt, **kw):
    """The human's own hand points (MANO joints incl. fingertips) through the robot formula, demo object, one lane."""
    fr = np.arange(SHIFT, S + SHIFT)  # V2D EDIT: frame shift
    hp = {s: np.asarray(human_joints[s], np.float64)[fr][:, None] for s in SIDES}
    run = Run(clip, clip.demo[fr][:, None], hp, clip.demo[fr])
    return run.evaluate(dt=dt, **kw)['slip_auc']


def object_only(clip, human_joints, states, dt, **kw):
    """Control: the human's hand points (MANO joints) against the ROBOT object (lost rule on): how much of the score the robot
    object alone explains."""
    S, L = states.shape[:2]
    fr = np.arange(SHIFT, S + SHIFT)  # V2D EDIT: frame shift
    hp = {s: np.broadcast_to(np.asarray(human_joints[s], np.float64)[fr][:, None], (S, L, 21, 3)) for s in SIDES}
    run = Run(clip, states, hp, clip.demo[fr])
    return run.evaluate(dt=dt, **kw)['slip_auc']


def load_recording(rdir):
    rdir = Path(rdir)
    meta = json.loads((rdir / 'record.json').read_text())
    z = np.load(rdir / 'recording.npz', allow_pickle=False)
    return meta, {k: z[k] for k in z.files}


def human_data(meta, rec, mano_dir):
    """MANO vertices (T, 778, 3) and joints (T, 21, 3) per hand for the recording's clip, from the per-sequence export."""
    obj, f0, f1, subject, use = meta['clip'].split('-')
    path = Path(mano_dir) / subject / f'{obj}_use_{use[1:]}.npz'
    if not path.is_file():
        sys.exit(f'{path} not found: export it with dexmachina/hand_proc/export_mano_vertices.py')
    z = np.load(path)
    f0, f1 = int(f0), int(f1)
    if z['verts_left'].shape[0] < f1:
        sys.exit(f'{path} has {z["verts_left"].shape[0]} frames, the clip needs {f1}')
    joints = {s: z[f'joints_{s}'][f0:f1].astype(np.float64) for s in SIDES}
    verts = {s: z[f'verts_{s}'][f0:f1].astype(np.float64) for s in SIDES}
    if len(rec['demo']) != f1 - f0:
        sys.exit(f"clip {meta['clip']} has {f1 - f0} frames, the recording's demo {len(rec['demo'])}")
    # when the run's demo data carries the human hand joints, they must equal the export (same clip, frames and world frame)
    check = {}
    for s in SIDES:
        hj = rec[f'demo_hand_joints_{s}']
        check[f'mano_vs_demo_joints_{s}_m'] = float(np.abs(joints[s] - hj).max()) if hj.size else None
    if any(v is not None and v > 1e-5 for v in check.values()):
        sys.exit(f'MANO export does not match the demo hand joints: {check}')
    return str(path), joints, verts, check


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rdir', help='recording directory (recording.npz + record.json from slip_record.py)')
    ap.add_argument('--mano', required=True, help='directory written by export_mano_vertices.py (<subject>/<object>_use_<NN>.npz)')
    ap.add_argument('--out', default='slip_auc')
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', type=type(v), default=v)
    ap.add_argument('--no_sensitivity', action='store_true')
    a = ap.parse_args()
    rdir = Path(a.rdir)
    meta, rec = load_recording(rdir)
    mano_file, joints, verts, mano_check = human_data(meta, rec, a.mano)
    pts = {'bottom': rec['add_points_bottom'].astype(np.float64), 'top': rec['add_points_top'].astype(np.float64)}
    trees, disc = surface_trees(meta['mesh_files'])
    demo = rec['demo'].astype(np.float64)
    dt = float(rec['dt'])
    states = rec['obj_state'].astype(np.float64)
    S, L = states.shape[:2]
    fr = np.arange(SHIFT, S + SHIFT)  # V2D EDIT: frame shift
    kpts = {s: rec[f'kpts_{s}'].astype(np.float64) for s in SIDES}
    clip = Clip(demo, verts, pts, trees)
    run = Run(clip, states, kpts, demo[fr])
    kw = {k: getattr(a, k) for k in DEFAULTS}
    res, fz = run.evaluate(dt=dt, with_frames=True, **kw)
    idx = boot_index(len(fz['num']))
    with np.errstate(invalid='ignore', divide='ignore'):
        res['ci95'] = ci(fz['num'][idx].sum(1) / fz['cnt'][idx].sum(1))
    res['ci95_method'] = f'moving-block bootstrap over steps, block {BOOT_BLOCK} steps, {BOOT_N} resamples, seed {BOOT_SEED}'
    lane = np.array([sum(fz['sc'][s, p][:, l][fz['gate'][s, p]['hold']].sum() for s in SIDES for p in PARTS) for l in range(L)])
    lane = lane / (fz['cnt'].sum() / L)
    res['lanes'] = dict(lane0=float(lane[0]), min=float(lane.min()), max=float(lane.max()), std=float(lane.std()), n=int(L))
    res['controls'] = dict(
        hand_only=run.evaluate(dt=dt, which='hand_only', **kw)['slip_auc'],
        object_only=object_only(clip, joints, states, dt, **kw),
        human_reference=human_reference(clip, joints, S, dt, **kw),
        rigid_grip_touching=1.0,
        notes='hand_only: robot hand points vs the DEMO object pose (no lost rule); object_only: human MANO joints vs the ROBOT '
              'object (lost rule on); human_reference: MANO joints vs the demo object through the robot formula; a rigid grip '
              'that touches the held part scores 1.0 by construction')
    sens = {}
    if not a.no_sensitivity:
        variants = {'window_0.05s': dict(window_s=0.05), 'window_0.2s': dict(window_s=0.2),
                    'human_touch_3mm': dict(human_touch_m=0.003), 'human_touch_10mm': dict(human_touch_m=0.01),
                    'human_slip_gate_0.5cm': dict(human_slip_cm=0.5), 'human_slip_gate_2cm': dict(human_slip_cm=2.0),
                    'human_slip_gate_off': dict(human_slip_cm=math.inf),
                    'robot_touch_1cm': dict(robot_touch_m=0.01), 'robot_touch_2cm': dict(robot_touch_m=0.02),
                    'thresholds_0.25-2.5cm': dict(th_lo_cm=0.25, th_hi_cm=2.5), 'thresholds_1-10cm': dict(th_lo_cm=1.0, th_hi_cm=10.0),
                    'lost_10cm': dict(lost_m=0.10), 'lost_off': dict(lost_m=None)}
        for name, v in variants.items():
            sens[name] = run.evaluate(dt=dt, **{**kw, **v})['slip_auc']
        sens['moving_part_only'] = run.evaluate(dt=dt, moving_only=True, **kw)['slip_auc']
    res['sensitivity'] = sens
    add_rec = official_add(rec)
    out = dict(tool='dexmachina/eval/slip_auc.py', tool_version=TOOL_VERSION, recording=str(rdir / 'recording.npz'),
               task=meta.get('task', ''), clip=meta['clip'], mano_file=mano_file, human_data_check=mano_check,
               settings=dict(**kw, window_steps=res['window_steps'], robot_points='all keypoint links per hand',
                             human_gate_points='MANO vertices (778 per hand)', human_reference_points='MANO joints (21, incl. fingertips)',
                             object_points='ADD-AUC points (official compute_add.py: 500 mesh vertices per part, seed 42), lost rule only',
                             surface_cloud_discretisation_p99_mm={k: v * 1000 for k, v in disc.items()}),
               checkpoint=meta['inputs']['checkpoint'], checkpoint_sha256=meta['inputs'].get('checkpoint_sha256'),
               first_observation=meta['settings']['first_observation'], add_auc_recorded=meta['add_auc'],
               add_auc_recomputed=add_rec, **res)
    (rdir / f'{a.out}.json').write_text(json.dumps(out, indent=1) + '\n')
    np.savez_compressed(
        rdir / f'{a.out}_frames.npz', num=fz['num'], cnt=fz['cnt'], lost_pair=fz['lost_pair'],
        **{f'hold_{s}_{p}': fz['gate'][s, p]['hold'] for s in SIDES for p in PARTS},
        **{f'human_slip_{s}_{p}': fz['gate'][s, p]['slip'].astype(np.float32) for s in SIDES for p in PARTS},
        **{f'slip_{s}_{p}': fz['slip'][s, p].astype(np.float32) for s in SIDES for p in PARTS})
    c = res['controls']
    print(f"{meta.get('task') or meta['clip']} Slip-AUC {res['slip_auc']:.4f} CI {np.round(res['ci95'], 4).tolist()} | coverage "
          f"{res['coverage']:.3f} slip score when touching {res['slip_score_when_touching']:.3f} lost {res['lost_share']:.3f} | "
          f"hand-only {c['hand_only']:.3f} object-only {c['object_only']:.3f} human {c['human_reference']:.3f} | "
          f"ADD-AUC {add_rec:.4f} (recorded {meta['add_auc']:.4f})")



# ==============================================================================================================================
# V2D drivers (new code, not part of slip_auc.py): configure(), main()'s computations as a function, the V2D loaders, the
# mesh check, the synthetic check, and the per-episode V2D driver.

import hashlib as _hashlib  # noqa: E402
import os as _os  # noqa: E402  # V2D EDIT
import time as _time  # noqa: E402

OUT_ = Path(_os.environ['DD_CAUC_OUT'])  # V2D EDIT: where score/epNN.json is written
HUMAN_ = Path(_os.environ['DD_CAUC_HUMAN'])  # human hand side per episode (reproduce/tools/human_soma.py)
RUNS_ = json.loads(_os.environ['DD_CAUC_RUNS'])  # V2D EDIT: {method: runs root with r<k>/ep<NN>/try<n>/}
SOMA_DIR_ = Path(_os.environ['HUMAN_MOTION_DATA_DIR']) / 'whole_body/soma'  # prepared Tier-1 inputs (object meshes)
TRIALS_ = tuple(int(x) for x in _os.environ.get('DD_CAUC_TRIALS', '1,2,3,4').split(','))  # V2D EDIT
SETS_ = tuple(_os.environ['DD_CAUC_ARMS'].split(','))  # V2D EDIT: methods scored together (shared reference)
V2D_DT = 1 / 60  # nominal: window_s 0.1 -> w = 6 frames (variants 0.05 s / 0.2 s -> 3 / 12 frames)
VARIANTS_ = {'window_0.05s': dict(window_s=0.05), 'window_0.2s': dict(window_s=0.2),
             'human_touch_3mm': dict(human_touch_m=0.003), 'human_touch_10mm': dict(human_touch_m=0.01),
             'human_slip_gate_0.5cm': dict(human_slip_cm=0.5), 'human_slip_gate_2cm': dict(human_slip_cm=2.0),
             'human_slip_gate_off': dict(human_slip_cm=math.inf),
             'robot_touch_1cm': dict(robot_touch_m=0.01), 'robot_touch_2cm': dict(robot_touch_m=0.02),
             'thresholds_0.25-2.5cm': dict(th_lo_cm=0.25, th_hi_cm=2.5), 'thresholds_1-10cm': dict(th_lo_cm=1.0, th_hi_cm=10.0),
             'lost_10cm': dict(lost_m=0.10), 'lost_off': dict(lost_m=None)}  # = slip_auc.py main() l.408-414


def configure(parts, shift):
    global PARTS, SHIFT
    PARTS, SHIFT = tuple(parts), int(shift)


def sha256_file(p):
    h = _hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def evaluate_all(clip, run, states, joints, dt, kw, sensitivity=True):
    """slip_auc.py main() l.389-418 on a built Clip / Run (same calls, same order); adds per-lane values and per-hand entry
    counts, and skips the CI / lanes when there are no entries (main() never met that case)."""
    S, L = states.shape[:2]
    res, fz = run.evaluate(dt=dt, with_frames=True, **kw)
    res['entries_by_hand'] = {s: int(sum(fz['gate'][s, p]['hold'].sum() for p in PARTS)) for s in SIDES}
    if res['entries']:
        idx = boot_index(len(fz['num']))
        with np.errstate(invalid='ignore', divide='ignore'):
            res['ci95'] = ci(fz['num'][idx].sum(1) / fz['cnt'][idx].sum(1))
        lane = np.array([sum(fz['sc'][s, p][:, l][fz['gate'][s, p]['hold']].sum() for s in SIDES for p in PARTS) for l in range(L)])
        lane = lane / (fz['cnt'].sum() / L)
        res['lanes'] = dict(lane0=float(lane[0]), min=float(lane.min()), max=float(lane.max()), std=float(lane.std()), n=int(L),
                            values=[float(x) for x in lane])
    else:
        res['ci95'], res['lanes'] = [None, None], dict(n=int(L), values=[None] * L)
    res['ci95_method'] = f'moving-block bootstrap over steps, block {BOOT_BLOCK} steps, {BOOT_N} resamples, seed {BOOT_SEED}'
    res['controls'] = dict(
        hand_only=run.evaluate(dt=dt, which='hand_only', **kw)['slip_auc'],
        object_only=object_only(clip, joints, states, dt, **kw),
        human_reference=human_reference(clip, joints, S, dt, **kw),
        rigid_grip_touching=1.0)
    sens = {}
    if sensitivity:
        for name, v in VARIANTS_.items():
            sens[name] = run.evaluate(dt=dt, **{**kw, **v})['slip_auc']
        sens['moving_part_only'] = run.evaluate(dt=dt, moving_only=True, **kw)['slip_auc']
    res['sensitivity'] = sens
    return res


# ------------------------------------------------------------------------------------------------------------------------------
# V2D inputs

def to_state(pose_xyzw):
    """(..., 7) [pos, xyzw] -> (..., 8) [pos, wxyz, 0] (the port's state; articulation 0)."""
    p = np.asarray(pose_xyzw, np.float64)
    out = np.zeros(p.shape[:-1] + (8,))
    out[..., :3] = p[..., :3]
    out[..., 3] = p[..., 6]
    out[..., 4:7] = p[..., 3:6]
    return out


def run_dir(st, tr, ep):
    base = Path(RUNS_[st]) / f'r{tr}/ep{ep:02d}'  # V2D EDIT: the first try with status ok
    for d in sorted(base.glob('try*')):  # V2D EDIT
        rj = d / 'record.json'  # V2D EDIT
        if rj.is_file() and json.loads(rj.read_text()).get('status') == 'ok':  # V2D EDIT
            return d  # V2D EDIT
    raise FileNotFoundError(f'no ok run under {base}')  # V2D EDIT


def load_runs(st, ep, trials):
    recs = [np.load(run_dir(st, tr, ep) / 'recording.npz', allow_pickle=False) for tr in trials]
    recs = [{k: z[k] for k in z.files} for z in recs]
    states = np.stack([to_state(r['body_q'][:, int(r['object_body_ids'][0])].astype(np.float64)) for r in recs], axis=1)
    kpts = {s: np.stack([r[f'kpts_{s}'].astype(np.float64) for r in recs], axis=1) for s in SIDES}
    return recs, states, kpts


def obj_vertices(path):
    v = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                v.append([float(x) for x in line.split()[1:4]])
    return np.asarray(v, np.float64)


def point_set_dist(a, b):
    return max(float(cKDTree(b).query(a)[0].max()), float(cKDTree(a).query(b)[0].max()))


def check_mesh(recs, mesh_file, tree_mesh_vertices):
    obj_v = obj_vertices(mesh_file)
    out = dict(mesh_file=str(mesh_file), obj_vertex_count=int(len(obj_v)), problems=[])
    for i, r in enumerate(recs):
        if [str(x) for x in r['object_body_names']] != ['object'] or len(r['object_body_ids']) != 1:
            out['problems'].append(f'run {i}: object bodies {r["object_body_names"]}')
        if not np.all(r['object_scales'] == 1.0):
            out['problems'].append(f'run {i}: object scale {r["object_scales"]}')
        if not np.array_equal(r['mesh_vertices_o'], recs[0]['mesh_vertices_o']):
            out['problems'].append(f'run {i}: simulator mesh differs from run 0')
    out['sim_vs_obj_point_set_max_m'] = point_set_dist(recs[0]['mesh_vertices_o'], obj_v)
    out['sim_vs_trimesh_point_set_max_m'] = point_set_dist(recs[0]['mesh_vertices_o'], tree_mesh_vertices)
    out['sim_vertex_count'] = int(len(recs[0]['mesh_vertices_o']))
    if out['sim_vs_obj_point_set_max_m'] > 1e-6:
        out['problems'].append('simulator mesh != textured_mesh.obj vertices')
    out['ok'] = not out['problems']
    return out


# ------------------------------------------------------------------------------------------------------------------------------
# synthetic check: recordings with the robot object = reference and keypoints fixed in the object frame

def synthetic_check(clip, demo, mesh_file, tree, kw):
    import trimesh
    m = trimesh.load(mesh_file, force='mesh')
    p, fi = trimesh.sample.sample_surface(m, 4000, seed=7)
    cand = p + 0.005 * m.face_normals[fi]
    d = tree.query(cand)[0]
    near = cand[(d < 0.012) & (d > 0.001)][:22]
    c = np.asarray(m.vertices).mean(0)
    rmax = float(np.linalg.norm(np.asarray(m.vertices) - c, axis=1).max())
    u = np.random.default_rng(3).normal(size=(22, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    far = c + (rmax + 0.10) * u
    out = dict(near_points=int(len(near)), near_dist_m=[float(tree.query(near)[0].min()), float(tree.query(near)[0].max())],
               far_min_dist_m=float(tree.query(far)[0].min()))
    S = len(demo)
    states = demo[:, None]
    poses = part_poses(demo)['object']
    res = {}
    for name, q in (('near_5mm', near), ('far_10cm', far)):
        world = np.einsum('sij,kj->ski', poses[0], q) + poses[1][:, None]
        kp = {'left': world[:, None, :11], 'right': world[:, None, 11:22]}
        run = Run(clip, states, kp, demo[np.arange(SHIFT, S + SHIFT)])
        r = run.evaluate(dt=V2D_DT, **kw)
        res[name] = dict(contact_auc=r['slip_auc'], entries=r['entries'], coverage=r['coverage'])
    out.update(res)
    out['passed'] = bool(len(near) == 22 and out['far_min_dist_m'] >= 0.0999 and res['near_5mm']['contact_auc'] == 1.0
                         and res['far_10cm']['contact_auc'] == 0.0)
    return out


# ------------------------------------------------------------------------------------------------------------------------------
# per-episode V2D driver: python -B contact_auc.py v2d <episode>

def v2d_main(ep):
    t0 = _time.time()
    configure(('object',), 0)
    kw = dict(DEFAULTS)
    hj = json.loads((HUMAN_ / f'ep{ep:02d}.json').read_text())  # V2D EDIT
    if not hj.get('ok'):
        sys.exit(f'episode {ep}: human-side pre-checks failed: {hj}')
    seq = hj['sequence_id']
    mesh_file = SOMA_DIR_ / f'sequence_id={seq}/object/textured_mesh.obj'
    hz = np.load(HUMAN_ / f'ep{ep:02d}.npz')  # V2D EDIT
    verts = {s: hz[f'verts_{s}'].astype(np.float64) for s in SIDES}
    joints = {s: hz[f'joints_{s}'].astype(np.float64) for s in SIDES}
    out = dict(episode=ep, object=hj['object'], sequence_id=seq, tool='contact_auc.py', tool_sha256=sha256_file(__file__),
               settings=dict(**kw, window_steps=int(round(kw['window_s'] / V2D_DT)), parts=list(PARTS), frame_shift=SHIFT,
                             robot_points='11 Dex3 keypoints per hand (3 stock fingertip frames + 8 collision-link origins)',
                             human_gate_points='SOMA hand vertices (2847 per hand)', human_reference_points='SOMA hand joints (21)',
                             object_points='evaluator ADD points (500, default_rng(0)), lost rule only'))
    # reference / ADD points / mesh: identical across all 8 runs of the episode
    allrecs = {}
    for st in SETS_:
        allrecs[st] = load_runs(st, ep, TRIALS_)
    r0 = allrecs[SETS_[0]][0][0]  # V2D EDIT: reference from the first method (checked equal across methods below)
    demo = to_state(r0['reference_pose_w'][:, 0])
    pts = r0['object_vertices_o'][0].astype(np.float64)
    same = all(np.array_equal(r['reference_pose_w'], r0['reference_pose_w']) and np.array_equal(r['object_vertices_o'], r0['object_vertices_o'])
               and int(r['step_count']) == len(demo) for st in SETS_ for r in allrecs[st][0])
    if not same:
        sys.exit(f'episode {ep}: reference / ADD points differ between runs')
    if len(demo) != verts['left'].shape[0]:
        sys.exit(f'episode {ep}: T mismatch {len(demo)} vs human {verts["left"].shape}')
    trees, disc = surface_trees({'object': mesh_file})
    import trimesh
    tm_v = np.asarray(trimesh.load(mesh_file, force='mesh').vertices)
    out['mesh_check'] = check_mesh([r for st in SETS_ for r in allrecs[st][0]], mesh_file, tm_v)
    out['surface_cloud_discretisation_p99_mm'] = disc['object'] * 1000
    out['surface_cloud_points'] = int(trees['object'].n)
    if not out['mesh_check']['ok']:
        sys.exit(f'episode {ep}: mesh check failed {out["mesh_check"]}')
    clip = Clip(demo, verts, {'object': pts}, trees)
    out['synthetic_check'] = synthetic_check(clip, demo, mesh_file, trees['object'], kw)
    S = len(demo)
    fr = np.arange(SHIFT, S + SHIFT)
    out['results'] = {}
    for st in SETS_:
        recs, states, kpts = allrecs[st]
        run = Run(clip, states, kpts, demo[fr])
        res = evaluate_all(clip, run, states, joints, V2D_DT, kw)
        res['trials'] = list(TRIALS_)
        out['results'][f'{st}:all'] = res
    out['seconds'] = round(_time.time() - t0, 1)
    (OUT_ / 'score').mkdir(parents=True, exist_ok=True)  # V2D EDIT
    out['human_files_sha256'] = {n: sha256_file(HUMAN_ / n) for n in (f'ep{ep:02d}.json', f'ep{ep:02d}.npz')}  # V2D EDIT
    out['runs'] = {st: {str(tr): str(run_dir(st, tr, ep)) for tr in TRIALS_} for st in SETS_}  # V2D EDIT
    (OUT_ / f'score/ep{ep:02d}.json').write_text(json.dumps(out, indent=1, default=float))
    brief = {k: (v['entries'] if isinstance(v, dict) else v) for k, v in out['results'].items()}
    print(json.dumps(dict(episode=ep, mesh_check=out['mesh_check']['ok'], synthetic_check=out['synthetic_check']['passed'],
                          entries=brief, seconds=out['seconds'])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'v2d':
        sys.exit('usage: contact_auc.py v2d <episode>')
    v2d_main(int(sys.argv[2]))
