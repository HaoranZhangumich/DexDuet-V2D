# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Contact-AUC on the V2D challenge with the paper's surface rule (G1 + Dex3), for one episode.

Usage (reproduce/contact_auc.sh sets the environment):
    python -B contact_auc_surface.py <episode> --touch 0.005 --n 64 --out <dir>

Environment:
    DD_CAUC_OUT     output folder (the score goes to <out>/score/epNN.json)
    DD_CAUC_RUNS    JSON {"<method>": "<folder holding r<k>/epNN/try<n>/recording.npz>"}
    DD_CAUC_ARMS    the method name(s), comma separated
    DD_CAUC_TRIALS  recorded trials to score, e.g. 1,2,3,4
    DD_CAUC_HUMAN   folder with the human hand side epNN.npz / epNN.json (reproduce/tools/human_soma.py)
    HUMAN_MOTION_DATA_DIR  prepared Tier-1 inputs (object meshes)
    FC_OFFICIAL     NVIDIA's official FlashCHORD checkout (the G1 URDF and its collision meshes)

The scorer is contact_auc.py (the Contact-AUC code of the DexMachina suite, ported to the challenge's recordings); this
file replaces two of its inputs:
  * robot points: N points sampled area-uniformly (trimesh, seed 0) on the URDF collision mesh of each of the 8 links of
    each Dex3 hand (palm, thumb 0-2, middle 0-1, index 0-1), moved with the recorded link pose. A hand touches the object
    when one of its points is within --touch (5 mm) of the object surface;
  * human points of the controls (human reference, object-only): the 2,847 SOMA hand vertices.
Everything else (the human gate, windows, thresholds, the lost rule, sensitivity variants) is the scorer's own code.
Nearest-neighbour queries are bounded at 5 cm, which only speeds them up: every distance is used only through d < 1 cm.
"""
import argparse
import importlib.util
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

URDF = Path(os.environ['FC_OFFICIAL']) / 'src/flash_chord/assets/urdfs/g1/main_with_hand.urdf'
LINKS = ('hand_palm_link', 'hand_thumb_0_link', 'hand_thumb_1_link', 'hand_thumb_2_link', 'hand_middle_0_link',
         'hand_middle_1_link', 'hand_index_0_link', 'hand_index_1_link')


def link_samples(n):
    root = ET.parse(URDF).getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    out = {}
    for side in ('left', 'right'):
        for ln in LINKS:
            c = links[f'{side}_{ln}'].find('collision')
            assert c.find('origin') is None
            m = trimesh.load(URDF.parent / c.find('geometry/mesh').get('filename'), force='mesh')
            out[f'{side}_{ln}'] = trimesh.sample.sample_surface(m, n, seed=0)[0].astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('episode', type=int)
    ap.add_argument('--touch', type=float, required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / f'score/ep{a.episode:02d}.json').is_file():
        return
    (out / 'score').mkdir(parents=True, exist_ok=True)
    os.cpu_count = lambda: 1  # thread count only: cKDTree.query(workers=-1) runs one thread; same results
    if not hasattr(np, 'trapz'):  # NumPy 2 renamed trapz to trapezoid (same function); the scorer was written for 1.x
        np.trapz = np.trapezoid
    spec = importlib.util.spec_from_file_location('contact_auc', Path(__file__).resolve().parent / 'contact_auc.py')
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    C.OUT_ = out
    C.DEFAULTS['robot_touch_m'] = a.touch
    C.VARIANTS_['robot_touch_1cm'] = dict(robot_touch_m=a.touch * 2 / 3)   # same relative spread as 1 / 1.5 / 2 cm
    C.VARIANTS_['robot_touch_2cm'] = dict(robot_touch_m=a.touch * 4 / 3)
    samples = link_samples(a.n)
    orig_load = C.load_runs

    def load_runs(st, ep, trials):
        recs, states, _ = orig_load(st, ep, trials)
        kpts = {}
        for s in C.SIDES:
            per = []
            for r in recs:
                labels = [str(x) for x in r['body_labels']]
                bq = r['body_q'].astype(np.float64)
                pts = []
                for ln in LINKS:
                    q = bq[:, labels.index(f'g1/{s}_{ln}')]
                    R = Rotation.from_quat(q[:, 3:7]).as_matrix()            # body_q quaternion is xyzw
                    pts.append(np.einsum('sij,kj->ski', R, samples[f'{s}_{ln}']) + q[:, None, :3])
                per.append(np.concatenate(pts, 1))
            kpts[s] = np.stack(per, 1)
        return recs, states, kpts

    C.load_runs = load_runs
    hz = np.load(Path(os.environ['DD_CAUC_HUMAN']) / f'ep{a.episode:02d}.npz')
    verts = {s: hz[f'verts_{s}'].astype(np.float64) for s in C.SIDES}
    orig_ref = C.human_reference
    C.human_reference = lambda clip, _j, S, dt, **kw: orig_ref(clip, verts, S, dt, **kw)

    def object_only(clip, _j, states, dt, **kw):  # C.object_only with the point count not fixed at 21
        S, L = states.shape[:2]
        fr = np.arange(C.SHIFT, S + C.SHIFT)
        hp = {s: np.broadcast_to(verts[s][fr][:, None], (S, L) + verts[s].shape[1:]) for s in C.SIDES}
        return C.Run(clip, states, hp, clip.demo[fr]).evaluate(dt=dt, **kw)['slip_auc']

    C.object_only = object_only
    # Bounded nearest-neighbour search (speed only): a hand point farther than 5 cm from the surface gets distance inf
    # instead of its exact distance. Distances are used only through d < touch (every threshold <= 1 cm, asserted), so
    # every contact mask and score is unchanged. The discretisation check and the synthetic checks keep the exact tree.
    BOUND_M = 0.05
    touch = [C.DEFAULTS['human_touch_m'], C.DEFAULTS['robot_touch_m']] + [
        v[k] for v in C.VARIANTS_.values() for k in ('human_touch_m', 'robot_touch_m') if k in v]
    assert max(touch) * 2 < BOUND_M, touch

    class Bounded:
        def __init__(self, tree):
            self.tree, self.n = tree, tree.n

        def query(self, x, *args, **kw):
            return self.tree.query(x, *args, distance_upper_bound=BOUND_M, **kw)

    orig_clip_init = C.Clip.__init__

    def clip_init(self, demo, human_verts, pts, trees):
        orig_clip_init(self, demo, human_verts, pts, {k: Bounded(t) for k, t in trees.items()})

    C.Clip.__init__ = clip_init
    C.v2d_main(a.episode)


if __name__ == '__main__':
    main()
