#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""DexDuet sidecars, stage B (FlashCHORD environment, PY, one GPU): human targets, object grids and their checks.

Inputs per episode: stage A's ``<work>/<id>/soma_hand.npz`` (scripts/dexduet/soma_hands.py), the Tier-1 parquet
``sequence_id=<id>/robot_name=g1/data.parquet`` and the object mesh ``sequence_id=<id>/object/textured_mesh.obj``.
Outputs under ``<targets>/<id>/``:

* ``dexduet_targets.npz``: ``hg_palm`` (T, 2, 2, 3) [normal, forward], ``hg_fingers`` (T, 2, 3, 3) (thumb, index,
  middle), ``hg_gaps`` (T, 2, 3) (= gap scale x human tip distance, pairs TI, TM, IM), ``palm_axis`` (2, 3),
  ``palm_axis_resultant`` (2), ``ho_dist`` (T, 2, 4), ``ho_spot`` (T, 2, 4, 3) (palm, thumb, index, middle), plus
  sequence_id, T, fps, source sha256s and the code commit. Vectors and spots are in the object (mesh) frame.
* ``object_grid.npz``: origin, spacing 0.003, margin 0.06, dims, ``dist`` (X, Y, Z), ``nearest`` (X, Y, Z, 3).

Rules: the frame chain parquet <- SOMA world is T = P_parquet(0) P_soma(0)^-1, checked over every frame
(blocks above 0.01 deg / 0.01 cm). Human vectors v(t) in the parquet frame are stored as R_obj,ref(t)^T v(t). Palm
normal n_h = unit((Index2 - Hand) x (Pinky2 - Hand)); forward unit(0.5 (Index2 + Middle2) - Hand); edges
unit(tip - base) for Thumb1 -> ThumbEnd, Index2 -> IndexEnd, Middle2 -> MiddleEnd. Palm axis (DexDuet
calibrate_palm_axis, float64): a = normalize(sum_t R_ref(t)^T n_h(t)), R_ref from ee_pose_w (wxyz) of the palm link;
blocks if the mean resultant is below 0.5. Contact: d_h = min over the region's vertices of the exact unsigned
distance to the mesh (Warp mesh_query_point_no_sign), s_h = the closest surface point of that vertex. Grid as DexDuet
build_grid (lo = bounds_min - margin, size = ceil((bounds_max + margin - lo) / spacing) + 1) with exact queries.
Blocking failures exit non-zero and write nothing for that episode; the check report goes to ``--check-dir``.
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

FINGERS = ("thumb", "index", "middle")
REGIONS = ("palm", "thumb", "index", "middle")
SIDES = ("left", "right")
GAP_PAIRS = ((0, 1), (0, 2), (1, 2))
MIN_RESULTANT = 0.5
MAX_CHAIN_DEG = 0.01
MAX_CHAIN_M = 1.0e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def calibrate_palm_axis(rotations: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, float]:
    """DexDuet rewards.calibrate_palm_axis in float64: a = normalize(mean_t R(t)^T n(t)), resultant |mean|."""
    local = np.einsum("tji,tj->ti", rotations, normals)
    mean = local.mean(axis=0)
    length = float(np.linalg.norm(mean))
    if not length > 0:
        raise ValueError("palm-axis calibration: the human palm normals cancel out")
    return mean / length, length


def hand_graph_targets(joints_p: np.ndarray, object_rot: np.ndarray, gap_scale: float):
    """joints_p (T, 2, 8, 3) in the parquet frame (HAND_JOINTS order) -> object-frame targets and parquet normals."""
    hand, thumb1, index2, middle2, pinky2, thumb_end, index_end, middle_end = (joints_p[:, :, j] for j in range(8))
    normal = unit(np.cross(index2 - hand, pinky2 - hand))
    forward = unit(0.5 * (index2 + middle2) - hand)
    palm = np.stack([normal, forward], axis=2)  # (T, 2, 2, 3)
    bases = np.stack([thumb1, index2, middle2], axis=2)
    tips = np.stack([thumb_end, index_end, middle_end], axis=2)
    fingers = unit(tips - bases)  # (T, 2, 3, 3)
    gaps = gap_scale * np.stack([np.linalg.norm(tips[:, :, a] - tips[:, :, b], axis=-1) for a, b in GAP_PAIRS], axis=-1)
    to_object = lambda v: np.einsum("tji,tsej->tsei", object_rot, v)  # noqa: E731  R^T v per frame
    return to_object(palm), to_object(fingers), gaps, normal


class MeshQuery:
    """Exact unsigned closest points on a triangle mesh with Warp's BVH (one device)."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        import warp as wp

        self.wp = wp
        self.device = device
        self.mesh = wp.Mesh(
            points=wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device),
        )

        @wp.kernel
        def closest(
            mesh: wp.uint64,
            points: wp.array(dtype=wp.vec3),
            max_dist: float,
            dist: wp.array(dtype=wp.float32),
            nearest: wp.array(dtype=wp.vec3),
        ):
            i = wp.tid()
            query = wp.mesh_query_point_no_sign(mesh, points[i], max_dist)
            if query.result:
                surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                nearest[i] = surface
                dist[i] = wp.length(points[i] - surface)
            else:
                nearest[i] = points[i]
                dist[i] = 1.0e6

        self.kernel = closest

    def __call__(self, points: np.ndarray, batch: int = 1 << 24) -> tuple[np.ndarray, np.ndarray]:
        wp = self.wp
        points = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        dist = np.empty(len(points), dtype=np.float32)
        nearest = np.empty((len(points), 3), dtype=np.float32)
        for start in range(0, len(points), batch):
            chunk = points[start : start + batch]
            d = wp.zeros(len(chunk), dtype=wp.float32, device=self.device)
            n = wp.zeros(len(chunk), dtype=wp.vec3, device=self.device)
            wp.launch(
                self.kernel,
                dim=len(chunk),
                inputs=[self.mesh.id, wp.array(chunk, dtype=wp.vec3, device=self.device), 1.0e3],
                outputs=[d, n],
                device=self.device,
            )
            dist[start : start + len(chunk)] = d.numpy()
            nearest[start : start + len(chunk)] = n.numpy()
        return dist, nearest


def grid_lookup(grid: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy transcription of DexDuet PartDistanceGrid.query (trilinear, align_corners=True, border clamp)."""
    dims = np.asarray(grid["dims"], dtype=np.int64)
    u = (np.asarray(points, dtype=np.float64) - np.asarray(grid["origin"], dtype=np.float64)) / float(grid["spacing"])
    u = np.clip(u, 0.0, dims - 1)
    i = np.minimum(np.floor(u).astype(np.int64), dims - 2)
    f = u - i
    dist = np.zeros(len(u))
    near = np.zeros((len(u), 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                index = (i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
                dist += w * grid["dist"][index]
                near += w[:, None] * grid["nearest"][index]
    return dist, near


def build_grid(query: MeshQuery, bounds: np.ndarray, spacing: float, margin: float) -> dict:
    lo = bounds[0] - margin
    size = np.ceil((bounds[1] + margin - lo) / spacing).astype(int) + 1
    axes = [lo[i] + spacing * np.arange(size[i]) for i in range(3)]
    dist = np.empty(tuple(size), dtype=np.float32)
    nearest = np.empty(tuple(size) + (3,), dtype=np.float32)
    for i in range(0, size[0], 32):  # x slabs keep host buffers small
        gx = axes[0][i : i + 32]
        pts = np.stack(np.meshgrid(gx, axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 3)
        d, n = query(pts)
        dist[i : i + len(gx)] = d.reshape(len(gx), size[1], size[2])
        nearest[i : i + len(gx)] = n.reshape(len(gx), size[1], size[2], 3)
    return dict(
        origin=lo.astype(np.float32),
        spacing=np.float32(spacing),
        margin=np.float32(margin),
        dims=size.astype(np.int64),
        dist=dist,
        nearest=nearest,
    )


def process(sequence: str, args, query_device: str) -> dict:
    import pyarrow.parquet as pq
    import trimesh

    data = Path(args.data_dir) / "whole_body" / "soma" / f"sequence_id={sequence}"
    soma_path = Path(args.work_dir) / sequence / "soma_hand.npz"
    parquet_path = data / "robot_name=g1" / "data.parquet"
    mesh_path = data / "object" / "textured_mesh.obj"
    report: dict = {"sequence_id": sequence, "blocking": []}
    soma = np.load(soma_path, allow_pickle=False)
    frames = int(soma["T"])
    row = pq.read_table(
        parquet_path,
        columns=["object_body_position", "object_body_wxyz", "ee_link_names", "ee_pose_w", "fps", "hand_contact_active"],
    ).to_pylist()[0]
    obj_p = np.asarray(row["object_body_position"], np.float64).reshape(-1, 3)
    obj_q = np.asarray(row["object_body_wxyz"], np.float64).reshape(-1, 4)
    ee = np.asarray(row["ee_pose_w"], np.float64)
    report["T"] = frames
    report["parquet_T"] = int(len(obj_p))
    if len(obj_p) != frames or ee.shape[0] != frames:
        report["blocking"].append(f"frame count: SOMA {frames}, parquet {len(obj_p)}, ee {ee.shape[0]}")
        return report
    fps = float(row["fps"])

    # Frame chain parquet <- SOMA world.
    p_parquet = np.tile(np.eye(4), (frames, 1, 1))
    p_parquet[:, :3, :3] = quat_wxyz_to_matrix(obj_q)
    p_parquet[:, :3, 3] = obj_p
    p_soma = np.asarray(soma["object_pose"], dtype=np.float64)
    chain = p_parquet[0] @ np.linalg.inv(p_soma[0])
    mapped = np.einsum("ij,tjk->tik", chain, p_soma)
    rot_err = np.einsum("tji,tjk->tik", mapped[:, :3, :3], p_parquet[:, :3, :3])
    angle = np.degrees(np.arccos(np.clip((np.trace(rot_err, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    trans = np.linalg.norm(mapped[:, :3, 3] - p_parquet[:, :3, 3], axis=-1)
    report["frame_chain"] = {
        "max_deg": float(angle.max()),
        "max_cm": float(trans.max() * 100.0),
        "det": float(np.linalg.det(chain[:3, :3])),
    }
    if angle.max() > MAX_CHAIN_DEG or trans.max() > MAX_CHAIN_M or abs(np.linalg.det(chain[:3, :3]) - 1.0) > 1e-6:
        report["blocking"].append(f"frame chain residual {angle.max():.5f} deg / {trans.max() * 100:.5f} cm")
        return report
    object_rot = p_parquet[:, :3, :3]

    # Hand graph.
    joints_p = np.einsum("ij,tskj->tski", chain[:3, :3], np.asarray(soma["joints"], dtype=np.float64)) + chain[:3, 3]
    hg_palm, hg_fingers, hg_gaps, normal_p = hand_graph_targets(joints_p, object_rot, args.gap_scale)
    links = [str(n) for n in row["ee_link_names"]]
    palm_axis = np.zeros((2, 3))
    resultant = np.zeros(2)
    report["palm_axis"] = {}
    for s, side in enumerate(SIDES):
        k = links.index(f"{side}_hand_palm_link")
        rotations = quat_wxyz_to_matrix(ee[:, k, 3:7])
        axis, length = calibrate_palm_axis(rotations, normal_p[:, s])
        axis32 = axis.astype(np.float32)
        palm_axis[s], resultant[s] = axis32, length
        angle_ref = np.degrees(np.arccos(np.clip(np.einsum("tij,j,ti->t", rotations, axis32.astype(np.float64), normal_p[:, s]), -1, 1)))
        report["palm_axis"][side] = {
            "axis": axis32.tolist(),
            "resultant": length,
            "ref_normal_angle_deg_median": float(np.median(angle_ref)),
            "ref_normal_angle_deg_p90": float(np.percentile(angle_ref, 90)),
        }
        if length < MIN_RESULTANT:
            report["blocking"].append(f"{side} palm-axis resultant {length:.4f} < {MIN_RESULTANT}")
    if report["blocking"]:
        return report

    # Contact targets and the object grid.
    mesh = trimesh.load(mesh_path, force="mesh")
    query = MeshQuery(mesh.vertices, mesh.faces, query_device)
    ho_dist = np.zeros((frames, 2, len(REGIONS)), dtype=np.float32)
    ho_spot = np.zeros((frames, 2, len(REGIONS), 3), dtype=np.float32)
    for s, side in enumerate(SIDES):
        for r, region in enumerate(REGIONS):
            verts = np.asarray(soma[f"verts_{side}_{region}"], dtype=np.float64)  # (T, n, 3) SOMA world
            verts_p = np.einsum("ij,tnj->tni", chain[:3, :3], verts) + chain[:3, 3]
            local = np.einsum("tji,tnj->tni", object_rot, verts_p - obj_p[:, None])  # R^T (v - p)
            d, near = query(local)
            d = d.reshape(frames, -1)
            near = near.reshape(frames, -1, 3)
            arg = d.argmin(axis=1)
            ho_dist[:, s, r] = d[np.arange(frames), arg]
            ho_spot[:, s, r] = near[np.arange(frames), arg]
    labels = np.asarray(row["hand_contact_active"], dtype=np.float64)
    report["contact"] = {}
    for s, side in enumerate(SIDES):
        any_touch = ho_dist[:, s].min(axis=1) < 0.01
        label = labels[s] > 0.5
        report["contact"][side] = {
            "label_share": float(label.mean()),
            "any_region_lt_1cm_share": float(any_touch.mean()),
            "agreement": float((any_touch == label).mean()),
            "region_lt_1cm_share": {region: float((ho_dist[:, s, r] < 0.01).mean()) for r, region in enumerate(REGIONS)},
            "region_median_cm": {region: float(np.median(ho_dist[:, s, r]) * 100) for r, region in enumerate(REGIONS)},
        }
    start = time.time()
    grid = build_grid(query, np.asarray(mesh.bounds, dtype=np.float64), args.spacing, args.margin)
    report["grid"] = {
        "dims": grid["dims"].tolist(),
        "nodes": int(np.prod(grid["dims"])),
        "mib": float(np.prod(grid["dims"]) * 16 / 2**20),
        "seconds": time.time() - start,
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extents_m": np.round(mesh.extents, 4).tolist(),
    }
    # Grid self-check: trilinear grid distance vs the exact distance at random points within 5 cm of the surface.
    rng = np.random.default_rng(0)
    surface = np.asarray(trimesh.sample.sample_surface(mesh, 20000, seed=0)[0], dtype=np.float64)
    direction = rng.normal(size=surface.shape)
    probe = surface + direction / np.linalg.norm(direction, axis=-1, keepdims=True) * rng.uniform(0.0, 0.05, (len(surface), 1))
    exact, _ = query(probe)
    interpolated = grid_lookup(grid, probe)[0]
    difference = np.abs(interpolated - exact)
    report["grid"]["trilinear_vs_exact_mm"] = {
        "median": float(np.median(difference) * 1000),
        "p99": float(np.percentile(difference, 99) * 1000),
        "max": float(difference.max() * 1000),
    }

    out_dir = Path(args.targets_dir) / sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "parquet": sha256(parquet_path),
        "mesh": sha256(mesh_path),
        "soma_params": str(soma["soma_sha256"]),
        "poses": str(soma["poses_sha256"]),
        "soma_hand": sha256(soma_path),
    }
    tmp = out_dir / "dexduet_targets.tmp.npz"
    np.savez(
        tmp,
        sequence_id=np.asarray(sequence),
        T=np.asarray(frames),
        fps=np.asarray(fps),
        sides=np.asarray(SIDES),
        fingers=np.asarray(FINGERS),
        regions=np.asarray(REGIONS),
        hg_palm=hg_palm.astype(np.float32),
        hg_fingers=hg_fingers.astype(np.float32),
        hg_gaps=hg_gaps.astype(np.float32),
        hg_gap_scale=np.asarray(args.gap_scale),
        palm_axis=palm_axis.astype(np.float32),
        palm_axis_resultant=resultant,
        ho_dist=ho_dist,
        ho_spot=ho_spot,
        frame_chain=chain,
        source_sha256=np.asarray(json.dumps(sources, sort_keys=True)),
        code_commit=np.asarray(args.code_commit),
    )
    os.replace(tmp, out_dir / "dexduet_targets.npz")
    tmp = out_dir / "object_grid.tmp.npz"
    np.savez(tmp, mesh_sha256=np.asarray(sources["mesh"]), **grid)
    os.replace(tmp, out_dir / "object_grid.npz")
    report["sources"] = sources
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--targets-dir", required=True)
    parser.add_argument("--check-dir", required=True)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--gap-scale", type=float, default=1.5)
    parser.add_argument("--spacing", type=float, default=0.003)
    parser.add_argument("--margin", type=float, default=0.06)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finalize", action="store_true", help="only write TARGETS.sha256 over the targets dir")
    args = parser.parse_args(argv)
    targets = Path(args.targets_dir)
    if args.finalize:
        lines = []
        for path in sorted(targets.rglob("*.npz")):
            if ".tmp." in path.name:
                continue
            lines.append(f"{sha256(path)}  {path.relative_to(targets).as_posix()}")
        (targets / "TARGETS.sha256").write_text("\n".join(lines) + "\n")
        print(f"TARGETS.sha256: {len(lines)} files")
        return 0

    import warp as wp

    wp.init()
    manifest = json.loads((Path(args.data_dir) / "whole_body" / "soma" / "track2_manifest.json").read_text())
    by_index = {int(entry["episode_index"]): entry["sequence_id"] for entry in manifest.values()}
    indices = sorted(by_index) if args.episodes == "all" else [int(v) for v in args.episodes.split(",")]
    check_dir = Path(args.check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for episode in indices:
        sequence = by_index[episode]
        start = time.time()
        report = process(sequence, args, args.device)
        report["episode"] = episode
        report["seconds"] = time.time() - start
        (check_dir / f"ep{episode:02d}_{sequence}.json").write_text(json.dumps(report, indent=1, sort_keys=True))
        status = "BLOCKED " + "; ".join(report["blocking"]) if report["blocking"] else "ok"
        failures += bool(report["blocking"])
        print(f"ep{episode:02d} {sequence}: {status} ({report['seconds']:.1f}s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
