#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""DexDuet sidecars, stage A (robotic_grounding environment, PY_V2D, CPU): SOMA forward kinematics of the human hands.

Per episode ``<data>/whole_body/soma/sequence_id=<id>/`` this writes ``<work>/<id>/soma_hand.npz`` with, in SOMA world:

* ``joints`` (T, 2, 8, 3) float64, sides (left, right) x HAND_JOINTS: the hand-graph nodes
  (Hand = MANO 0, Thumb1 = 13, Index2 = 1, Middle2 = 4, Pinky2 = 7, ThumbEnd = 16, IndexEnd = 17, MiddleEnd = 18);
* ``verts_<side>_<region>`` (T, n, 3) float32: SOMA mesh vertices whose dominant skinning joint belongs to the contact
  region (DexDuet ho_contact MANO_ANCHOR_JOINTS: palm {0} -> Hand + Index1/Middle1/Ring1/Pinky1; thumb {14, 15} ->
  Thumb2 + Thumb3; index {2, 3} -> Index3 + Index4; middle {5, 6} -> Middle3 + Middle4) and ``vidx_<side>_<region>``;
* ``object_pose`` (T, 4, 4): the object pose in SOMA world, ``diag(1, -1, -1, 1) @ poses.npy`` (OpenCV world);
* ``soma_sha256`` / ``poses_sha256`` of the inputs.

SOMA settings as read_soma.py / soma_to_g1.py: model ``mhr``, ``enable_procedural_transforms=False``, identity and scale
of frame 0, ``repose_to_bind_pose=False``, ``apply_correctives=False``. Read-only on the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HAND_JOINTS = ("Hand", "HandThumb1", "HandIndex2", "HandMiddle2", "HandPinky2", "HandThumbEnd", "HandIndexEnd", "HandMiddleEnd")
REGIONS = {
    "palm": ("Hand", "HandIndex1", "HandMiddle1", "HandRing1", "HandPinky1"),
    "thumb": ("HandThumb2", "HandThumb3"),
    "index": ("HandIndex3", "HandIndex4"),
    "middle": ("HandMiddle3", "HandMiddle4"),
}
SIDES = (("left", "Left"), ("right", "Right"))
CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", required=True, help="$HUMAN_MOTION_DATA_DIR (contains whole_body/soma)")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--episodes", default="all", help="'all' or comma-separated manifest episode indices")
    parser.add_argument(
        "--soma-assets",
        required=True,
        help="SOMA body model folder, robotic_grounding's assets/body_models/soma (reproduce/prepare_data.sh)",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    import torch
    from soma import SOMALayer

    torch.set_num_threads(args.threads)
    soma_root = Path(args.data_dir) / "whole_body" / "soma"
    manifest = json.loads((soma_root / "track2_manifest.json").read_text())
    by_index = {int(entry["episode_index"]): entry["sequence_id"] for entry in manifest.values()}
    indices = sorted(by_index) if args.episodes == "all" else [int(v) for v in args.episodes.split(",")]

    layer = SOMALayer(
        data_root=args.soma_assets, identity_model_type="mhr", device="cpu", enable_procedural_transforms=False
    )
    names = [str(n) for n in layer.rig_data["joint_names"]][1:]
    joint_index = {n: i for i, n in enumerate(names)}
    public = list(layer.public_joint_names)
    weights = layer.public_skinning_weights()
    if torch.is_tensor(weights) and weights.layout != torch.strided:
        weights = weights.to_dense()
    dominant = np.asarray(weights.cpu().numpy() if torch.is_tensor(weights) else weights).argmax(1)
    members = {}
    for side, prefix in SIDES:
        for region, joints in REGIONS.items():
            ids = [public.index(prefix + joint) for joint in joints]
            members[(side, region)] = np.where(np.isin(dominant, ids))[0]
            if len(members[(side, region)]) == 0:
                raise RuntimeError(f"no SOMA vertices for {side} {region}")
    print("region vertex counts", {f"{s}/{r}": len(v) for (s, r), v in members.items()}, flush=True)

    for episode in indices:
        sequence = by_index[episode]
        source = soma_root / f"sequence_id={sequence}"
        out_dir = Path(args.work_dir) / sequence
        out_path = out_dir / "soma_hand.npz"
        if out_path.exists():
            print(f"ep{episode:02d} {sequence}: exists, skipped", flush=True)
            continue
        start = time.time()
        z = np.load(source / "soma_params.npz", allow_pickle=True)
        if [str(x) for x in z["joint_names"]] != names:
            raise RuntimeError(f"{sequence}: SOMA joint order differs from the rig")
        frames = z["poses"].shape[0]
        with torch.no_grad():
            layer.prepare_identity(
                torch.from_numpy(np.asarray(z["identity_coeffs"][0], np.float32))[None],
                torch.from_numpy(np.asarray(z["scale_params"][0], np.float32))[None],
                repose_to_bind_pose=False,
            )
            out = layer.pose(
                torch.from_numpy(np.ascontiguousarray(z["poses"], dtype=np.float32).reshape(frames, 77, 3)),
                transl=torch.from_numpy(np.ascontiguousarray(z["transl"], dtype=np.float32)),
                apply_correctives=False,
                fk_only=False,
            )
        joints_all = out["joints"].numpy().astype(np.float64)
        vertices = out["vertices"].numpy()
        joints = np.zeros((frames, 2, len(HAND_JOINTS), 3), dtype=np.float64)
        payload = {}
        for s, (side, prefix) in enumerate(SIDES):
            for j, joint in enumerate(HAND_JOINTS):
                joints[:, s, j] = joints_all[:, joint_index[prefix + joint]]
            for region in REGIONS:
                idx = members[(side, region)]
                payload[f"verts_{side}_{region}"] = vertices[:, idx].astype(np.float32)
                payload[f"vidx_{side}_{region}"] = idx.astype(np.int64)
        poses = np.load(source / "poses.npy")
        if poses.shape != (frames, 4, 4):
            raise RuntimeError(f"{sequence}: poses.npy {poses.shape} vs {frames} SOMA frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "soma_hand.tmp.npz"
        np.savez(
            tmp,
            sequence_id=np.asarray(sequence),
            T=np.asarray(frames),
            sides=np.asarray([s for s, _ in SIDES]),
            hand_joints=np.asarray(HAND_JOINTS),
            regions=np.asarray(list(REGIONS)),
            joints=joints,
            object_pose=np.einsum("ij,tjk->tik", CV_TO_SOMA, poses),
            soma_sha256=np.asarray(sha256(source / "soma_params.npz")),
            poses_sha256=np.asarray(sha256(source / "poses.npy")),
            **payload,
        )
        os.replace(tmp, out_path)
        print(f"ep{episode:02d} {sequence}: T={frames} {time.time() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
