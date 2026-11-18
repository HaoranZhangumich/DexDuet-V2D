# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Human hand side of the V2D Contact-AUC (reproduce/contact_auc.sh), with its checks.

Runs in the robotic_grounding environment (py-soma-x 0.2.1), CPU only, with -B. For one Tier-1 episode it
  * checks soma_params.npz against the Tier-1 parquet and that every frame count agrees,
  * reconstructs the SOMA body with the official retargeter's reader, unchanged
    (read_soma.SOMA(identity_model_type="mhr", device=cpu).load_motion(params, normalize=False)),
  * selects each hand's vertices (strongest skinning joint is a <Side>Hand* joint) and its 21 joints,
  * measures the constant rigid map X from the raw SOMA world to the evaluator (G1) world,
  * checks the hand sides against the G1 reference palm links,
and writes <out>/ep<NN>.npz (hand vertices / joints in the G1 world, float64) + ep<NN>.json (checks). The checks are
stored under the keys P1 (parquet), P2 (frame counts), P3 (rigid map), P5 (hand vertices) and P6 (hand sides);
`ok` is true when all of them pass.

Usage: python -B human_soma.py <episode> --data <V2D_DATA> --tier1 <track_2/tier_1_multiview_caption>
           --robotic-grounding <video_to_data>/robotic_grounding/source/robotic_grounding --recording <recording.npz> --out <dir>
--recording is any recorded evaluation of this episode (reproduce/contact_auc.sh passes its first one): its reference
object pose fixes the map from the SOMA world to the evaluator world.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csc_matrix, csr_matrix  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import argparse  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument('episode', type=int)
_ap.add_argument('--data', required=True, type=Path, help='prepared Tier-1 inputs (V2D_DATA, holds whole_body/soma)')
_ap.add_argument('--tier1', required=True, type=Path, help='challenge folder track_2/tier_1_multiview_caption')
_ap.add_argument('--robotic-grounding', required=True, type=Path,
                 help='<video_to_data>/robotic_grounding/source/robotic_grounding (read_soma.py and the SOMA model)')
_ap.add_argument('--recording', required=True, type=Path, help='recording.npz of one recorded evaluation of this episode')
_ap.add_argument('--out', required=True, type=Path)
ARGS = _ap.parse_args()
OUT = ARGS.out
SOMA_DIR = ARGS.data / 'whole_body/soma'
MANIFEST = SOMA_DIR / 'track2_manifest.json'
T1 = ARGS.tier1 / 'data/chunk-000'
RG = ARGS.robotic_grounding
READ_SOMA = RG / 'robotic_grounding/retarget/read_soma.py'
SOMA_NPZ = RG / 'robotic_grounding/assets/body_models/soma/SOMA_neutral.npz'
# The official SOMA reader and body model (video_to_data @ 8c778f9) are checked; the manifest written by
# reproduce/tools/track2_to_soma_inputs.py for all 30 episodes in one call had the sha256 below (reported, not required).
PINS = {READ_SOMA: '00e99371f2532c317db846d88b54714271a7b3acac7e895bfd31b47de8bc8c87',
        SOMA_NPZ: '515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f'}
MANIFEST_SHA256 = '12fe4494fff6d5eea074e682a7466b4edf943883e2c3cae0d64781f4479a0d52'
D = np.diag([1.0, -1.0, -1.0])
SIDES = ('left', 'right')
SNAME = {'left': 'Left', 'right': 'Right'}
EXPECTED_HAND_VERTS = 2847


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def joints21(S):
    n = [f'{S}Hand', f'{S}HandThumb1', f'{S}HandThumb2', f'{S}HandThumb3', f'{S}HandThumbEnd']
    for f in ('Index', 'Middle', 'Ring', 'Pinky'):
        n += [f'{S}Hand{f}2', f'{S}Hand{f}3', f'{S}Hand{f}4', f'{S}Hand{f}End']
    return n


def pose_mats(pos, quat_xyzw):
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:, :3, 3] = pos
    return T


def main():
    ep = ARGS.episode
    t0 = time.time()
    chk = dict(episode=ep, problems=[])
    for p, h in PINS.items():
        got = sha256(p)
        if got != h:
            sys.exit(f'pinned file changed: {p} {got}')
    chk['manifest_sha256_matches_paper'] = sha256(MANIFEST) == MANIFEST_SHA256
    man = json.loads(MANIFEST.read_text())
    ent = [v for v in man.values() if int(v['episode_index']) == ep]
    assert len(ent) == 1, ent
    ent = ent[0]
    seq = ent['sequence_id']
    sdir = SOMA_DIR / f'sequence_id={seq}'
    chk.update(sequence_id=seq, object=ent['object'], manifest_frames=int(ent['frames']),
               invisible_object_frames=int(ent['invisible_object_frames']))

    # ---- P1: soma_params.npz == Tier-1 parquet
    params = sdir / 'soma_params.npz'
    z = np.load(params, allow_pickle=True)
    t1 = pd.read_parquet(T1 / f'episode_{ep:06d}.parquet')
    assert (t1['episode_index'] == ep).all()
    P1 = {}
    P1['pose'] = float(np.abs(np.stack(t1['observation.human.pose'].values).reshape(z['poses'].shape) - z['poses']).max())
    P1['translation'] = float(np.abs(np.stack(t1['observation.human.translation'].values) - z['transl']).max())
    P1['identity_coeffs'] = float(np.abs(np.stack(t1['observation.human.identity_coeffs'].values) - z['identity_coeffs']).max())
    P1['scale_params'] = float(np.abs(np.stack(t1['observation.human.scale_params'].values) - z['scale_params']).max())
    chk['P1_max_abs_diff'] = P1
    chk['P1'] = bool(max(P1.values()) <= 1e-6)
    # Tier-1 object pose vs poses.npy (the object pose T_SOMA is built from)
    poses = np.load(sdir / 'poses.npy')
    t1_obj = np.stack(t1['observation.object.pose'].values).astype(np.float64)
    vis = t1['observation.object.visible'].values.astype(bool)
    chk['tier1_object_pos_vs_poses_npy_max_m'] = float(np.abs(t1_obj[:, :3] - poses[:, :3, 3]).max())
    chk['invisible_frames_tier1'] = [int(i) for i in np.flatnonzero(~vis)]

    # ---- P2: frame counts
    g1 = pd.read_parquet(sdir / 'robot_name=g1/data.parquet').iloc[0]
    T_soma = int(z['poses'].shape[0])
    T_t1 = len(t1)
    T_g1 = len(g1['object_root_position'])
    rec = np.load(ARGS.recording)
    T_eval = int(rec['step_count'])
    ts = t1['timestamp'].values.astype(np.float64)
    dts = np.diff(ts)
    chk['P2_counts'] = dict(soma=T_soma, tier1=T_t1, manifest=chk['manifest_frames'], g1_reference=T_g1, evaluator_step_count=T_eval,
                            poses_npy=int(len(poses)), tier1_frame_index_ok=bool((t1['frame_index'].values == np.arange(T_t1)).all()),
                            timestamp_step_minmax=[float(dts.min()), float(dts.max())], g1_fps=float(g1['fps']))
    chk['P2'] = bool(len({T_soma, T_t1, chk['manifest_frames'], T_g1, T_eval, len(poses)}) == 1
                     and chk['P2_counts']['tier1_frame_index_ok'] and np.allclose(dts, 1 / 30, atol=1e-5) and float(g1['fps']) == 30.0)
    T = T_soma

    # ---- SOMA reconstruction with the official reader (unchanged), CPU
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    sys.path.insert(0, str(RG))
    from robotic_grounding.retarget import read_soma as rs
    if Path(rs.__file__).resolve() != READ_SOMA.resolve():
        sys.exit(f'read_soma imported from {rs.__file__}')
    soma = rs.SOMA(identity_model_type='mhr', device=torch.device('cpu'))
    chk['soma_data_root'] = str(soma.data_root)
    mot = soma.load_motion(params, normalize=False)
    V = np.asarray(mot['vertices'], np.float64)
    J = np.asarray(mot['joints'], np.float64)
    jn = list(mot['joint_names'])
    chk['soma_shapes'] = dict(vertices=list(V.shape), joints=list(J.shape), joint_names=len(jn))
    import soma as soma_pkg
    chk['py_soma_x_version'] = getattr(soma_pkg, '__version__', None)
    if J.shape[1] != len(jn):
        sys.exit(f'joint count {J.shape} != names {len(jn)}')

    # ---- hand vertex sets from the skinning matrix (P5)
    a = np.load(SOMA_NPZ, allow_pickle=True)
    names78 = [str(n) for n in a['joint_names']]
    shp = tuple(int(x) for x in a['skinning_weights_shape'])
    ip = a['skinning_weights_indptr']
    if len(ip) == shp[0] + 1:
        W = csr_matrix((a['skinning_weights_data'], a['skinning_weights_indices'], ip), shape=shp)
        layout = 'csr'
    elif len(ip) == shp[1] + 1:
        W = csc_matrix((a['skinning_weights_data'], a['skinning_weights_indices'], ip), shape=shp)
        layout = 'csc'
    else:
        sys.exit(f'skinning layout? shape {shp} indptr {len(ip)}')
    Wd = W.toarray()
    assert Wd.shape == (V.shape[1], len(names78)), (Wd.shape, V.shape)
    top = np.argmax(Wd, axis=1)
    hv = {s: np.flatnonzero(np.array([names78[j].startswith(f'{SNAME[s]}Hand') for j in top])) for s in SIDES}
    par = a['joint_parent_ids']
    parent_ok = all(names78[par[names78.index(f'{SNAME[s]}Hand{f}1')]] == f'{SNAME[s]}Hand'
                    for s in SIDES for f in ('Index', 'Middle', 'Ring', 'Pinky'))
    hand_joint_names = {s: sorted({names78[j] for j in top[hv[s]]}) for s in SIDES}
    chk['P5_detail'] = dict(skinning_layout=layout, skinning_shape=list(shp), hand_vertices={s: int(len(hv[s])) for s in SIDES},
                            disjoint=bool(len(np.intersect1d(hv['left'], hv['right'])) == 0), finger1_parent_is_hand=parent_ok,
                            hand_joints_owning_vertices={s: len(v) for s, v in hand_joint_names.items()},
                            rig_names_match_output=bool(names78[1:] == jn))
    chk['P5'] = bool(all(len(hv[s]) == EXPECTED_HAND_VERTS for s in SIDES) and chk['P5_detail']['disjoint'] and parent_ok
                     and names78[1:] == jn)
    j21 = {s: [jn.index(n) for n in joints21(SNAME[s])] for s in SIDES}

    # ---- X: raw SOMA world -> evaluator (G1) world (P3)
    ref = rec['reference_pose_w'][:, 0].astype(np.float64)        # (T, 7) pos + xyzw
    TG = pose_mats(ref[:, :3], ref[:, 3:7])
    TS = np.einsum('ij,tjk->tik', np.block([[D, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]), poses)
    X = TG @ np.linalg.inv(TS)
    X0 = X[0]
    dt_ = np.linalg.norm(X[:, :3, 3] - X0[:3, 3], axis=1)
    rel = np.einsum('tij,kj->tik', X[:, :3, :3], X0[:3, :3])
    ang = np.degrees(Rotation.from_matrix(rel).magnitude())
    chk['P3_detail'] = dict(max_dt_m=float(dt_.max()), max_angle_deg=float(ang.max()), argmax_dt=int(dt_.argmax()),
                            det_R_X0=float(np.linalg.det(X0[:3, :3])), X0=X0.tolist(),
                            orthonormal_err=float(np.abs(X0[:3, :3] @ X0[:3, :3].T - np.eye(3)).max()))
    chk['P3'] = bool(dt_.max() <= 1e-4 and ang.max() <= 0.01 and abs(np.linalg.det(X0[:3, :3]) - 1) < 1e-6)
    # also: the G1 reference parquet object track equals the evaluator reference (same world)
    g1pos = np.stack(g1['object_root_position']).astype(np.float64)
    chk['g1_parquet_vs_evaluator_ref_pos_max_m'] = float(np.abs(g1pos - ref[:, :3]).max())

    RX, tX = X0[:3, :3], X0[:3, 3]
    to_g1 = lambda P: np.einsum('ij,...j->...i', RX, P) + tX  # noqa: E731
    verts = {s: to_g1(V[:, hv[s]]) for s in SIDES}
    jts = {s: to_g1(J[:, j21[s]]) for s in SIDES}

    # ---- P6 side check against the G1 reference palm links (ee_pose_w)
    ee_names = [str(n) for n in g1['ee_link_names']]
    ee = np.stack([np.stack(list(f)) for f in g1['ee_pose_w']]).astype(np.float64)   # (T, 2, 7)
    P6 = {}
    for s in SIDES:
        k = ee_names.index(f'{s}_hand_palm_link')
        palm = ee[:, k, :3]
        own = np.median(np.linalg.norm(palm - jts[s][:, 0], axis=1))
        other = np.median(np.linalg.norm(palm - jts['right' if s == 'left' else 'left'][:, 0], axis=1))
        P6[s] = dict(median_to_same_side_m=float(own), median_to_other_side_m=float(other))
    chk['P6_detail'] = dict(source='g1 data.parquet ee_pose_w (ee_link_names %s); hand_frames_w is empty' % ee_names, **P6)
    chk['P6'] = bool(all(P6[s]['median_to_same_side_m'] < P6[s]['median_to_other_side_m'] for s in SIDES))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'ep{ep:02d}.npz', verts_left=verts['left'], verts_right=verts['right'],
             joints_left=jts['left'], joints_right=jts['right'], X=X0, hand_vertex_ids_left=hv['left'],
             hand_vertex_ids_right=hv['right'], joint21_names_left=np.array(joints21('Left')),
             joint21_names_right=np.array(joints21('Right')))
    chk['ok'] = bool(chk['P1'] and chk['P2'] and chk['P3'] and chk['P5'] and chk['P6'])
    chk['seconds'] = round(time.time() - t0, 1)
    chk['tool_sha256'] = sha256(__file__)
    (OUT / f'ep{ep:02d}.json').write_text(json.dumps(chk, indent=1))
    print(json.dumps({k: chk[k] for k in ('episode', 'object', 'P1', 'P2', 'P3', 'P5', 'P6', 'ok', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
