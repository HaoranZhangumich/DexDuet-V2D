# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""DexDuet hand rewards for the G1 + Dex3 ReconBody objective (off by default).

Two terms ported from DexDuet (github.com/HaoranZhangumich/DexDuet):

* ``hand_graph`` (r_hg): ``dexmachina/envs/rewards.py`` ``hand_graph_palm_frame`` / ``compute_hand_graph_reward`` with
  the general finger rule and the Dex3 calibrated palm axis (``calibrate_palm_axis``).
* ``ho_contact`` (r_hoc): ``dexmachina/envs/ho_contact.py`` ``anchor_contacts``,
  ``HandObjectContact.robot_distances`` and ``HandObjectContact.score`` with one object part.

Per hand s in (left, right), every control step, at frame ``f = clamp(timestep, 0, T - 1)`` (the stock ReconBody frame):

hand graph (nodes: wrist = palm_link origin; bases = thumb_0 / index_0 / middle_0 origins; tips = thumb_2 + (0, -+o, 0)
left/right, index_1 / middle_1 + (o, 0, 0), o = ``hand_graph_tip_offset``)::

    n_r = unit(R_palm a_s)                     (a_s: palm axis calibrated offline per episode and side)
    f_r = unit(0.5 (b_index + b_middle) - w)
    e_f = unit(tip_f - base_f)                 f in (thumb, index, middle)
    targets: R_obj,sim(t) v_obj(f) for the human normal, forward and finger edges stored in the object frame
    palm_err   = mean_{normal, forward} (1 - clamp(cos, -1, 1))
    finger_err = sum_f w_f (1 - clamp(cos, -1, 1)),  w = (thumb_weight, 1, 1) / sum
    dir_err    = 0.5 (palm_err + finger_err)
    gap_err    = mean over (T,I), (T,M), (I,M) of | |tip_a - tip_b| - g* |   (g* = gap scale x human tip distance)
    r_s        = exp(-beta_dir dir_err - beta_gap gap_err)
    hand_graph = 0.5 (r_left + r_right)

hand-object contact (regions palm, thumb, index, middle; robot surface points sampled offline in their link frames;
object distance grid with trilinear, border-clamped lookups; the simulator contact overrides a touching region)::

    d_r, s_r   = min over the region's points of the grid distance, and the grid's nearest surface point there
    touching   -> d_r = 0, s_r = contact point (object frame) of the most distal touching link
    err        = |min(d_r, cap) - min(d_h, cap)| + |s_r - s_h| [d_h < cap]
    score      = exp(-max(err - tol, 0) / scale)
    weight     = 1 if d_r < touch or d_h < touch; near if d_r < cap or d_h < cap; else 0
    hand_s     = sum(weight score) / sum(weight), 1 if every weight is 0
    ho_contact = 0.5 (hand_left + hand_right)

Human targets (d_h, s_h, the object-frame hand-graph vectors, gap targets and the palm axes) are offline sidecars
(``scripts/dexduet/``); this module only loads and checks them. Everything is allocated at build time; ``evaluate``
launches fixed-size kernels only (no host reads, no allocation), so it is CUDA-graph capturable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

DEXDUET_SIDES = ("left", "right")
DEXDUET_FINGERS = ("thumb", "index", "middle")
DEXDUET_REGIONS = ("palm", "thumb", "index", "middle")
#: Robot links of each contact region, proximal to distal (DexDuet finger_spec CONTACT_DEPTH dex3_hand (2, 1)).
DEXDUET_REGION_LINKS = {
    "palm": ("hand_palm_link",),
    "thumb": ("hand_thumb_1_link", "hand_thumb_2_link"),
    "index": ("hand_index_1_link",),
    "middle": ("hand_middle_1_link",),
}
#: Hand-graph robot nodes per side: wrist, finger bases (thumb, index, middle), fingertip links (thumb, index, middle).
DEXDUET_GRAPH_LINKS = (
    "hand_palm_link",
    "hand_thumb_0_link",
    "hand_index_0_link",
    "hand_middle_0_link",
    "hand_thumb_2_link",
    "hand_index_1_link",
    "hand_middle_1_link",
)
#: Fingertip gap pairs, indices into DEXDUET_FINGERS (thumb with each other finger, then neighbouring fingers).
DEXDUET_GAP_PAIRS = ((0, 1), (0, 2), (1, 2))
DEXDUET_TARGETS_FILE = "dexduet_targets.npz"
DEXDUET_GRID_FILE = "object_grid.npz"
DEXDUET_POINTS_FILE = "dex3_region_points.npz"
DEXDUET_HASH_FILE = "TARGETS.sha256"
MIN_PALM_AXIS_RESULTANT = 0.5
_GRAPH_NODES = wp.constant(7)
_NUM_FINGERS = wp.constant(3)


def dexduet_tip_offsets(offset: float) -> np.ndarray:
    """(2, 3, 3) fingertip offsets in the tip-link frames, (left, right) x (thumb, index, middle).

    DexMachina's Dex3 URDFs (assets/dex3_hand/dex3_{left,right}_6dof.urdf): ``left_thumb_tip_joint`` xyz 0 -0.045 0,
    ``right_thumb_tip_joint`` 0 0.045 0, index/middle tips 0.045 0 0 on both sides.
    """
    tips = np.zeros((2, 3, 3), dtype=np.float32)
    tips[0, 0] = (0.0, -offset, 0.0)
    tips[1, 0] = (0.0, offset, 0.0)
    tips[:, 1] = (offset, 0.0, 0.0)
    tips[:, 2] = (offset, 0.0, 0.0)
    return tips


def dexduet_finger_weights(thumb_weight: float) -> np.ndarray:
    """DexDuet finger weights (thumb_weight, 1, 1), normalised to sum 1 (thumb 0.5 -> 0.2, 0.4, 0.4)."""
    weights = np.asarray((thumb_weight, 1.0, 1.0), dtype=np.float64)
    return (weights / weights.sum()).astype(np.float32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_hash_list(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        out[name.lstrip("*").strip()] = digest
    return out


def _scalar(value) -> object:
    array = np.asarray(value)
    return array.item() if array.shape == () else array


@dataclass(frozen=True)
class DexDuetSidecars:
    """Host copies of the offline DexDuet targets of one episode, checked against the loaded reference."""

    sequence_id: str
    num_frames: int
    hg_palm: np.ndarray | None  # (T, 2, 2, 3) object frame: human palm normal, forward
    hg_fingers: np.ndarray | None  # (T, 2, 3, 3) object frame: human finger edges (thumb, index, middle)
    hg_gaps: np.ndarray | None  # (T, 2, 3) gap targets [m] (already scaled)
    palm_axis: np.ndarray | None  # (2, 3) palm-link frame
    ho_dist: np.ndarray | None  # (T, 2, 4) human region distances [m]
    ho_spot: np.ndarray | None  # (T, 2, 4, 3) object frame
    grid: dict | None  # origin (3,), spacing, dims (3,), dist (X, Y, Z), nearest (X, Y, Z, 3)
    points: dict | None  # per side: points (P, 3), link names per point, region slices (4, 2)
    sha256: dict[str, str]


def load_dexduet_sidecars(
    directory: str | Path,
    sequence_id: str | None,
    num_frames: int,
    *,
    hand_graph: bool,
    ho_contact: bool,
    gap_scale: float,
    contact_cap: float,
) -> DexDuetSidecars:
    """Load and check ``<directory>/<sequence_id>/dexduet_targets.npz`` (+ grid and region points for r_hoc).

    Refuses on a missing directory or file, a sequence_id or frame-count mismatch, a gap scale that differs from the
    configured one, a palm axis that is not unit length, non-finite values, a grid margin below the contact cap, and on
    any file whose sha256 differs from ``<directory>/TARGETS.sha256``.
    """
    if not directory:
        raise ValueError("DexDuet rewards are enabled but objective.dexduet_targets_dir is empty")
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"DexDuet targets directory does not exist: {root}")
    if not sequence_id:
        raise ValueError("DexDuet rewards need the reference sequence_id to find the episode sidecars")
    hash_path = root / DEXDUET_HASH_FILE
    if not hash_path.is_file():
        raise FileNotFoundError(f"DexDuet targets need {hash_path} (sha256 of every sidecar)")
    recorded = _read_hash_list(hash_path)

    def checked(relative: str) -> Path:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"DexDuet sidecar missing: {path}")
        if relative not in recorded:
            raise ValueError(f"DexDuet sidecar {relative} is not listed in {hash_path}")
        digest = sha256_file(path)
        if digest != recorded[relative]:
            raise ValueError(f"DexDuet sidecar {relative} sha256 {digest} differs from {hash_path} ({recorded[relative]})")
        hashes[relative] = digest
        return path

    hashes: dict[str, str] = {}
    targets_path = checked(f"{sequence_id}/{DEXDUET_TARGETS_FILE}")
    with np.load(targets_path, allow_pickle=False) as z:
        file_sequence = str(_scalar(z["sequence_id"]))
        file_frames = int(_scalar(z["T"]))
        if file_sequence != sequence_id:
            raise ValueError(f"DexDuet targets are for {file_sequence!r}, the reference is {sequence_id!r}")
        if file_frames != num_frames:
            raise ValueError(f"DexDuet targets have T={file_frames}, the reference has {num_frames} frames")
        if tuple(str(s) for s in z["sides"]) != DEXDUET_SIDES:
            raise ValueError(f"DexDuet targets side order {tuple(z['sides'])} != {DEXDUET_SIDES}")
        arrays = {key: np.asarray(z[key]) for key in z.files}
    shapes = {
        "hg_palm": (num_frames, 2, 2, 3),
        "hg_fingers": (num_frames, 2, 3, 3),
        "hg_gaps": (num_frames, 2, 3),
        "palm_axis": (2, 3),
        "ho_dist": (num_frames, 2, len(DEXDUET_REGIONS)),
        "ho_spot": (num_frames, 2, len(DEXDUET_REGIONS), 3),
    }
    needed = (("hg_palm", "hg_fingers", "hg_gaps", "palm_axis") if hand_graph else ()) + (
        ("ho_dist", "ho_spot") if ho_contact else ()
    )
    for key in needed:
        if key not in arrays:
            raise ValueError(f"DexDuet targets {targets_path} lack {key}")
        if arrays[key].shape != shapes[key]:
            raise ValueError(f"DexDuet targets {key} has shape {arrays[key].shape}; expected {shapes[key]}")
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"DexDuet targets {key} contain non-finite values")
    if hand_graph:
        file_scale = float(_scalar(arrays["hg_gap_scale"]))
        if abs(file_scale - gap_scale) > 1.0e-9:
            raise ValueError(f"DexDuet gap targets were built with scale {file_scale}, the config says {gap_scale}")
        norms = np.linalg.norm(arrays["palm_axis"].astype(np.float64), axis=-1)
        if np.any(np.abs(norms - 1.0) > 1.0e-4):
            raise ValueError(f"DexDuet palm axes must be unit vectors, got norms {norms}")
        resultant = np.asarray(arrays["palm_axis_resultant"], dtype=np.float64)
        if np.any(resultant < MIN_PALM_AXIS_RESULTANT):
            raise ValueError(f"DexDuet palm-axis calibration resultant {resultant} < {MIN_PALM_AXIS_RESULTANT}")
        if tuple(str(f) for f in arrays["fingers"]) != DEXDUET_FINGERS:
            raise ValueError(f"DexDuet targets finger order {tuple(arrays['fingers'])} != {DEXDUET_FINGERS}")

    grid = points = None
    if ho_contact:
        if tuple(str(r) for r in arrays["regions"]) != DEXDUET_REGIONS:
            raise ValueError(f"DexDuet targets region order {tuple(arrays['regions'])} != {DEXDUET_REGIONS}")
        with np.load(checked(f"{sequence_id}/{DEXDUET_GRID_FILE}"), allow_pickle=False) as z:
            grid = {key: np.asarray(z[key]) for key in ("origin", "spacing", "dims", "dist", "nearest", "margin")}
        dims = tuple(int(v) for v in grid["dims"])
        if grid["dist"].shape != dims or grid["nearest"].shape != dims + (3,) or min(dims) < 2:
            raise ValueError(f"DexDuet grid arrays {grid['dist'].shape}/{grid['nearest'].shape} != dims {dims}")
        if float(grid["margin"]) < contact_cap:
            raise ValueError(f"DexDuet grid margin {float(grid['margin'])} m is smaller than the contact cap {contact_cap}")
        if not (np.all(np.isfinite(grid["dist"])) and np.all(np.isfinite(grid["nearest"]))):
            raise ValueError("DexDuet grid contains non-finite values")
        with np.load(checked(DEXDUET_POINTS_FILE), allow_pickle=False) as z:
            if tuple(str(r) for r in z["regions"]) != DEXDUET_REGIONS:
                raise ValueError(f"DexDuet region points order {tuple(z['regions'])} != {DEXDUET_REGIONS}")
            points = {}
            for side in DEXDUET_SIDES:
                side_points = np.asarray(z[f"points_{side}"], dtype=np.float32)
                links = tuple(str(name) for name in z[f"links_{side}"])
                point_link = np.asarray(z[f"point_link_{side}"], dtype=np.int64)
                slices = np.asarray(z[f"region_slices_{side}"], dtype=np.int64)
                if side_points.ndim != 2 or side_points.shape[1] != 3 or point_link.shape != (side_points.shape[0],):
                    raise ValueError(f"DexDuet {side} region points have inconsistent shapes")
                if slices.shape != (len(DEXDUET_REGIONS), 2) or slices[0, 0] != 0 or slices[-1, 1] != len(side_points):
                    raise ValueError(f"DexDuet {side} region slices {slices.tolist()} do not cover the points")
                if np.any(slices[1:, 0] != slices[:-1, 1]) or np.any(slices[:, 1] <= slices[:, 0]):
                    raise ValueError(f"DexDuet {side} region slices {slices.tolist()} must be contiguous and nonempty")
                points[side] = {"points": side_points, "links": links, "point_link": point_link, "slices": slices}
            if points["left"]["points"].shape != points["right"]["points"].shape:
                raise ValueError("DexDuet left and right hands need the same number of region points")
    return DexDuetSidecars(
        sequence_id=sequence_id,
        num_frames=num_frames,
        hg_palm=arrays.get("hg_palm") if hand_graph else None,
        hg_fingers=arrays.get("hg_fingers") if hand_graph else None,
        hg_gaps=arrays.get("hg_gaps") if hand_graph else None,
        palm_axis=arrays.get("palm_axis") if hand_graph else None,
        ho_dist=arrays.get("ho_dist") if ho_contact else None,
        ho_spot=arrays.get("ho_spot") if ho_contact else None,
        grid=grid,
        points=points,
        sha256=hashes,
    )


# --------------------------------------------------------------------------------------------------------------------
# Warp kernels
# --------------------------------------------------------------------------------------------------------------------


@wp.func
def _dd_unit(v: wp.vec3) -> wp.vec3:
    """DexDuet ``unit``: v / max(|v|, 1e-6)."""
    return v / wp.max(wp.length(v), 1.0e-6)


@wp.func
def _dd_one_minus_cos(robot: wp.vec3, target: wp.vec3) -> float:
    return 1.0 - wp.clamp(wp.dot(robot, target), -1.0, 1.0)


@wp.kernel
def dexduet_hand_graph(
    body_q: wp.array(dtype=wp.transform),
    timestep: wp.array(dtype=wp.int32),
    object_body_id: int,
    bodies_per_world: int,
    num_frames: int,
    node_body_ids: wp.array(dtype=wp.int32),
    tip_offsets: wp.array(dtype=wp.vec3),
    palm_axis: wp.array(dtype=wp.vec3),
    target_palm: wp.array(dtype=wp.vec3),
    target_fingers: wp.array(dtype=wp.vec3),
    target_gaps: wp.array(dtype=wp.float32),
    finger_weights: wp.array(dtype=wp.float32),
    beta_dir: float,
    beta_gap: float,
    reward: wp.array(dtype=wp.float32),
    palm_err_out: wp.array(dtype=wp.float32),
    finger_err_out: wp.array(dtype=wp.float32),
    gap_err_out: wp.array(dtype=wp.float32),
) -> None:
    """r_hg of one (world, side); thread = world * 2 + side."""
    tid = wp.tid()
    world = tid // 2
    side = tid - world * 2
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    base = world * bodies_per_world
    object_rotation = wp.normalize(wp.transform_get_rotation(body_q[base + object_body_id]))
    node = side * _GRAPH_NODES
    wrist_xf = body_q[base + node_body_ids[node]]
    wrist = wp.transform_get_translation(wrist_xf)
    base_thumb = wp.transform_get_translation(body_q[base + node_body_ids[node + 1]])
    base_index = wp.transform_get_translation(body_q[base + node_body_ids[node + 2]])
    base_middle = wp.transform_get_translation(body_q[base + node_body_ids[node + 3]])
    tip_thumb = wp.transform_point(body_q[base + node_body_ids[node + 4]], tip_offsets[side * _NUM_FINGERS])
    tip_index = wp.transform_point(body_q[base + node_body_ids[node + 5]], tip_offsets[side * _NUM_FINGERS + 1])
    tip_middle = wp.transform_point(body_q[base + node_body_ids[node + 6]], tip_offsets[side * _NUM_FINGERS + 2])

    normal = _dd_unit(wp.quat_rotate(wp.normalize(wp.transform_get_rotation(wrist_xf)), palm_axis[side]))
    forward = _dd_unit(0.5 * (base_index + base_middle) - wrist)
    palm_row = (frame * 2 + side) * 2
    palm_err = 0.5 * (
        _dd_one_minus_cos(normal, wp.quat_rotate(object_rotation, target_palm[palm_row]))
        + _dd_one_minus_cos(forward, wp.quat_rotate(object_rotation, target_palm[palm_row + 1]))
    )
    finger_row = (frame * 2 + side) * _NUM_FINGERS
    finger_err = (
        finger_weights[0]
        * _dd_one_minus_cos(_dd_unit(tip_thumb - base_thumb), wp.quat_rotate(object_rotation, target_fingers[finger_row]))
        + finger_weights[1]
        * _dd_one_minus_cos(
            _dd_unit(tip_index - base_index), wp.quat_rotate(object_rotation, target_fingers[finger_row + 1])
        )
        + finger_weights[2]
        * _dd_one_minus_cos(
            _dd_unit(tip_middle - base_middle), wp.quat_rotate(object_rotation, target_fingers[finger_row + 2])
        )
    )
    dir_err = 0.5 * (palm_err + finger_err)
    gap_err = (
        wp.abs(wp.length(tip_thumb - tip_index) - target_gaps[finger_row])
        + wp.abs(wp.length(tip_thumb - tip_middle) - target_gaps[finger_row + 1])
        + wp.abs(wp.length(tip_index - tip_middle) - target_gaps[finger_row + 2])
    ) / 3.0
    reward[tid] = wp.exp(-beta_dir * dir_err - beta_gap * gap_err)
    palm_err_out[tid] = palm_err
    finger_err_out[tid] = finger_err
    gap_err_out[tid] = gap_err


@wp.func
def _dd_grid_axis(value: float, origin: float, spacing: float, count: int) -> float:
    """Continuous grid index, clamped to the border (grid_sample align_corners=True, padding_mode='border')."""
    return wp.clamp((value - origin) / spacing, 0.0, float(count - 1))


@wp.func
def _dd_grid_cell(u: float, count: int) -> int:
    return wp.min(int(wp.floor(u)), count - 2)


@wp.func
def _dd_grid_distance(
    point: wp.vec3,
    grid_dist: wp.array(dtype=wp.float32),
    origin: wp.vec3,
    spacing: float,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    ux = _dd_grid_axis(point[0], origin[0], spacing, nx)
    uy = _dd_grid_axis(point[1], origin[1], spacing, ny)
    uz = _dd_grid_axis(point[2], origin[2], spacing, nz)
    i = _dd_grid_cell(ux, nx)
    j = _dd_grid_cell(uy, ny)
    k = _dd_grid_cell(uz, nz)
    fx = ux - float(i)
    fy = uy - float(j)
    fz = uz - float(k)
    c00 = grid_dist[(i * ny + j) * nz + k] * (1.0 - fx) + grid_dist[((i + 1) * ny + j) * nz + k] * fx
    c10 = grid_dist[(i * ny + j + 1) * nz + k] * (1.0 - fx) + grid_dist[((i + 1) * ny + j + 1) * nz + k] * fx
    c01 = grid_dist[(i * ny + j) * nz + k + 1] * (1.0 - fx) + grid_dist[((i + 1) * ny + j) * nz + k + 1] * fx
    c11 = grid_dist[(i * ny + j + 1) * nz + k + 1] * (1.0 - fx) + grid_dist[((i + 1) * ny + j + 1) * nz + k + 1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _dd_grid_nearest(
    point: wp.vec3,
    grid_near: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    spacing: float,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    ux = _dd_grid_axis(point[0], origin[0], spacing, nx)
    uy = _dd_grid_axis(point[1], origin[1], spacing, ny)
    uz = _dd_grid_axis(point[2], origin[2], spacing, nz)
    i = _dd_grid_cell(ux, nx)
    j = _dd_grid_cell(uy, ny)
    k = _dd_grid_cell(uz, nz)
    fx = ux - float(i)
    fy = uy - float(j)
    fz = uz - float(k)
    c00 = grid_near[(i * ny + j) * nz + k] * (1.0 - fx) + grid_near[((i + 1) * ny + j) * nz + k] * fx
    c10 = grid_near[(i * ny + j + 1) * nz + k] * (1.0 - fx) + grid_near[((i + 1) * ny + j + 1) * nz + k] * fx
    c01 = grid_near[(i * ny + j) * nz + k + 1] * (1.0 - fx) + grid_near[((i + 1) * ny + j) * nz + k + 1] * fx
    c11 = grid_near[(i * ny + j + 1) * nz + k + 1] * (1.0 - fx) + grid_near[((i + 1) * ny + j + 1) * nz + k + 1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def dexduet_ho_contact_regions(
    body_q: wp.array(dtype=wp.transform),
    object_body_id: int,
    bodies_per_world: int,
    points: wp.array(dtype=wp.vec3),
    point_body_ids: wp.array(dtype=wp.int32),
    points_per_side: int,
    region_start: wp.array(dtype=wp.int32),
    region_count: wp.array(dtype=wp.int32),
    num_regions: int,
    region_slots: wp.array(dtype=wp.int32),
    contact_active: wp.array(dtype=wp.int32),
    contact_pos_o: wp.array(dtype=wp.vec3),
    slots_per_world: int,
    grid_dist: wp.array(dtype=wp.float32),
    grid_near: wp.array(dtype=wp.vec3),
    grid_origin: wp.vec3,
    grid_spacing: float,
    nx: int,
    ny: int,
    nz: int,
    region_dist: wp.array(dtype=wp.float32),
    region_spot: wp.array(dtype=wp.vec3),
    region_grid_dist: wp.array(dtype=wp.float32),
    region_touching: wp.array(dtype=wp.int32),
) -> None:
    """Robot distance and spot of one (world, side, region); thread = (world * 2 + side) * num_regions + region."""
    tid = wp.tid()
    world = tid // (2 * num_regions)
    rest = tid - world * 2 * num_regions
    side = rest // num_regions
    region = rest - side * num_regions
    base = world * bodies_per_world
    object_inverse = wp.transform_inverse(body_q[base + object_body_id])
    entry = side * num_regions + region
    start = side * points_per_side + region_start[entry]
    best = float(1.0e30)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    best_point = wp.vec3(0.0)
    for offset in range(region_count[entry]):
        index = start + offset
        point_w = wp.transform_point(body_q[base + point_body_ids[index]], points[index])
        point_o = wp.transform_point(object_inverse, point_w)
        distance = _dd_grid_distance(point_o, grid_dist, grid_origin, grid_spacing, nx, ny, nz)
        if distance < best:
            best = distance
            best_point = point_o
    spot = _dd_grid_nearest(best_point, grid_near, grid_origin, grid_spacing, nx, ny, nz)
    region_grid_dist[tid] = best
    touching = int(0)
    for link in range(2):
        slot = region_slots[entry * 2 + link]
        if slot >= 0:
            flat = world * slots_per_world + slot
            if contact_active[flat] != 0:
                touching = 1
                spot = contact_pos_o[flat]
    distance = best
    if touching != 0:
        distance = 0.0
    region_dist[tid] = distance
    region_spot[tid] = spot
    region_touching[tid] = touching


@wp.kernel
def dexduet_ho_contact_score(
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    num_regions: int,
    region_dist: wp.array(dtype=wp.float32),
    region_spot: wp.array(dtype=wp.vec3),
    target_dist: wp.array(dtype=wp.float32),
    target_spot: wp.array(dtype=wp.vec3),
    cap: float,
    tol: float,
    scale: float,
    touch: float,
    near: float,
    hand_out: wp.array(dtype=wp.float32),
) -> None:
    """Hand score of one (world, side) from its region distances and the human targets; thread = world * 2 + side."""
    tid = wp.tid()
    world = tid // 2
    side = tid - world * 2
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    weight_sum = float(0.0)  # noqa: UP018
    score_sum = float(0.0)  # noqa: UP018
    for region in range(num_regions):
        robot = tid * num_regions + region
        human = (frame * 2 + side) * num_regions + region
        d_robot = region_dist[robot]
        d_human = target_dist[human]
        human_near = d_human < cap
        error = wp.abs(wp.clamp(d_robot, 0.0, cap) - wp.clamp(d_human, 0.0, cap))
        if human_near:
            error += wp.length(region_spot[robot] - target_spot[human])
        score = wp.exp(-wp.max(error - tol, 0.0) / scale)
        weight = float(0.0)  # noqa: UP018
        if d_robot < touch or d_human < touch:
            weight = 1.0
        elif d_robot < cap or human_near:
            weight = near
        weight_sum += weight
        score_sum += weight * score
    hand = float(1.0)  # noqa: UP018
    if weight_sum > 0.0:
        hand = score_sum / wp.max(weight_sum, 1.0e-6)
    hand_out[tid] = hand


@wp.kernel
def append_dexduet_terms(
    pre_terms: wp.array(dtype=wp.float32),
    pre_count: int,
    term_count: int,
    hand_graph_column: int,
    ho_contact_column: int,
    hand_graph_side: wp.array(dtype=wp.float32),
    ho_contact_hand: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    """Copy the earlier term columns bitwise, append hand_graph / ho_contact (mean of the two hands), add to score."""
    world = wp.tid()
    for term in range(pre_count):
        terms[world * term_count + term] = pre_terms[world * pre_count + term]
    if hand_graph_column >= 0:
        hand_graph = 0.5 * (hand_graph_side[world * 2] + hand_graph_side[world * 2 + 1])
        terms[world * term_count + hand_graph_column] = hand_graph
        score[world] = score[world] + frame_dt * weights[hand_graph_column] * hand_graph
    if ho_contact_column >= 0:
        ho_contact = 0.5 * (ho_contact_hand[world * 2] + ho_contact_hand[world * 2 + 1])
        terms[world * term_count + ho_contact_column] = ho_contact
        score[world] = score[world] + frame_dt * weights[ho_contact_column] * ho_contact


# --------------------------------------------------------------------------------------------------------------------
# Device state
# --------------------------------------------------------------------------------------------------------------------


def _per_world_body_ids(model, embodiment, side: str, names: tuple[str, ...]) -> list[int]:
    """Per-world body IDs of named hand links, through their collision shapes (checked against semantic frames)."""
    hand = embodiment.hand(side)
    shape_body = np.asarray(model.shape_body.numpy(), dtype=np.int64)
    by_name = {link.link_name: link for link in hand.link_geometry}
    out = []
    for suffix in names:
        name = f"{side}_{suffix}"
        if name not in by_name:
            raise ValueError(f"DexDuet needs hand link {name!r}; {side} links are {tuple(by_name)}")
        bodies = {int(shape_body[shape]) for shape in by_name[name].shape_ids}
        if len(bodies) != 1:
            raise ValueError(f"DexDuet hand link {name!r} shapes belong to bodies {sorted(bodies)}")
        out.append(bodies.pop())
    frames = {frame.name: frame.body_id for frame in (hand.palm_frame, *hand.dp_frames)}
    for suffix, body_id in zip(names, out, strict=True):
        name = f"{side}_{suffix}"
        if name in frames and frames[name] != body_id:
            raise ValueError(f"DexDuet body of {name!r} ({body_id}) differs from its semantic frame ({frames[name]})")
    return out


@dataclass
class DexDuetTerms:
    """Device buffers and kernels of the enabled DexDuet terms of one ReconBody objective."""

    world_count: int
    num_frames: int
    bodies_per_world: int
    object_body_id: int
    frame_dt: float
    pre_count: int
    term_count: int
    hand_graph_column: int
    ho_contact_column: int
    beta_dir: float
    beta_gap: float
    cap: float
    tol: float
    scale: float
    touch: float
    near: float
    sha256: dict
    # hand graph
    node_body_ids: wp.array | None = None
    tip_offsets: wp.array | None = None
    palm_axis: wp.array | None = None
    target_palm: wp.array | None = None
    target_fingers: wp.array | None = None
    target_gaps: wp.array | None = None
    finger_weights: wp.array | None = None
    hand_graph_side: wp.array | None = None
    hand_graph_palm_err: wp.array | None = None
    hand_graph_finger_err: wp.array | None = None
    hand_graph_gap_err: wp.array | None = None
    # hand-object contact
    points: wp.array | None = None
    point_body_ids: wp.array | None = None
    points_per_side: int = 0
    region_start: wp.array | None = None
    region_count: wp.array | None = None
    region_slots: wp.array | None = None
    slots_per_world: int = 0
    grid_dist: wp.array | None = None
    grid_near: wp.array | None = None
    grid_origin: tuple = (0.0, 0.0, 0.0)
    grid_spacing: float = 0.0
    grid_dims: tuple = (0, 0, 0)
    target_dist: wp.array | None = None
    target_spot: wp.array | None = None
    region_dist: wp.array | None = None
    region_spot: wp.array | None = None
    region_grid_dist: wp.array | None = None
    region_touching: wp.array | None = None
    ho_contact_hand: wp.array | None = None

    @classmethod
    def build(
        cls,
        config,
        model,
        embodiment,
        reference,
        contact,
        object_body_id: int,
        bodies_per_world: int,
        world_count: int,
        frame_dt: float,
        pre_count: int,
        device=None,
    ) -> DexDuetTerms:
        if tuple(config.sides) != DEXDUET_SIDES:
            raise ValueError(f"DexDuet terms need objective sides {DEXDUET_SIDES}, got {config.sides}")
        metadata = getattr(reference, "metadata", None)
        if metadata is not None and getattr(metadata, "is_resampled", False):
            raise ValueError("DexDuet targets are per source frame; the reference must use source_frame_playback")
        hand_graph = config.hand_graph_weight != 0.0
        ho_contact = config.ho_contact_weight != 0.0
        sidecars = load_dexduet_sidecars(
            config.dexduet_targets_dir,
            getattr(metadata, "sequence_id", None),
            int(reference.num_frames),
            hand_graph=hand_graph,
            ho_contact=ho_contact,
            gap_scale=config.hand_graph_gap_scale,
            contact_cap=config.ho_contact_cap,
        )
        term_names = config.term_names
        terms = cls(
            world_count=world_count,
            num_frames=int(reference.num_frames),
            bodies_per_world=bodies_per_world,
            object_body_id=object_body_id,
            frame_dt=frame_dt,
            pre_count=pre_count,
            term_count=len(term_names),
            hand_graph_column=term_names.index("hand_graph") if hand_graph else -1,
            ho_contact_column=term_names.index("ho_contact") if ho_contact else -1,
            beta_dir=config.hand_graph_beta_dir,
            beta_gap=config.hand_graph_beta_gap,
            cap=config.ho_contact_cap,
            tol=config.ho_contact_tol,
            scale=config.ho_contact_scale,
            touch=config.ho_contact_touch,
            near=config.ho_contact_near,
            sha256=dict(sidecars.sha256),
        )
        f32 = lambda values: wp.array(np.ascontiguousarray(values, dtype=np.float32), dtype=wp.float32, device=device)  # noqa: E731
        i32 = lambda values: wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=wp.int32, device=device)  # noqa: E731
        v3 = lambda values: wp.array(  # noqa: E731
            np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1, 3)), dtype=wp.vec3, device=device
        )
        if hand_graph:
            node_ids = [i for side in DEXDUET_SIDES for i in _per_world_body_ids(model, embodiment, side, DEXDUET_GRAPH_LINKS)]
            terms.node_body_ids = i32(node_ids)
            terms.tip_offsets = v3(dexduet_tip_offsets(config.hand_graph_tip_offset))
            terms.palm_axis = v3(sidecars.palm_axis)
            terms.target_palm = v3(sidecars.hg_palm)
            terms.target_fingers = v3(sidecars.hg_fingers)
            terms.target_gaps = f32(sidecars.hg_gaps.reshape(-1))
            terms.finger_weights = f32(dexduet_finger_weights(config.hand_graph_thumb_weight))
            terms.hand_graph_side = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
            terms.hand_graph_palm_err = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
            terms.hand_graph_finger_err = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
            terms.hand_graph_gap_err = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
        if ho_contact:
            layout = contact.layout
            if tuple(layout.sides) != DEXDUET_SIDES:
                raise ValueError(f"DexDuet contact needs contact sides {DEXDUET_SIDES}, got {layout.sides}")
            if layout.num_objects != 1:
                raise ValueError("DexDuet contact supports one object body")
            points, bodies, starts, counts, slots = [], [], [], [], []
            for hand_id, side in enumerate(DEXDUET_SIDES):
                side_points = sidecars.points[side]
                link_bodies = _per_world_body_ids(
                    model, embodiment, side, tuple(name.removeprefix(f"{side}_") for name in side_points["links"])
                )
                points.append(side_points["points"])
                bodies.extend(link_bodies[int(i)] for i in side_points["point_link"])
                link_index = {link.link_name: i for i, link in enumerate(embodiment.hand(side).contact_links)}
                for region, (start, stop) in zip(DEXDUET_REGIONS, side_points["slices"], strict=True):
                    starts.append(int(start))
                    counts.append(int(stop - start))
                    names = [f"{side}_{suffix}" for suffix in DEXDUET_REGION_LINKS[region]]
                    point_links = {side_points["links"][int(i)] for i in side_points["point_link"][start:stop]}
                    if point_links != set(names):
                        raise ValueError(f"DexDuet {side} {region} points lie on {sorted(point_links)}, expected {names}")
                    region_slots = [layout.hand_slot_starts[hand_id] + link_index[name] for name in names]
                    slots.extend(region_slots + [-1] * (2 - len(region_slots)))
            terms.points = v3(np.concatenate(points, 0))
            terms.point_body_ids = i32(bodies)
            terms.points_per_side = int(sidecars.points["left"]["points"].shape[0])
            terms.region_start = i32(starts)
            terms.region_count = i32(counts)
            terms.region_slots = i32(slots)
            terms.slots_per_world = int(layout.slots_per_world)
            grid = sidecars.grid
            terms.grid_dist = f32(grid["dist"].reshape(-1))
            terms.grid_near = v3(grid["nearest"])
            terms.grid_origin = tuple(float(v) for v in np.asarray(grid["origin"], dtype=np.float64))
            terms.grid_spacing = float(grid["spacing"])
            terms.grid_dims = tuple(int(v) for v in grid["dims"])
            terms.target_dist = f32(sidecars.ho_dist.reshape(-1))
            terms.target_spot = v3(sidecars.ho_spot)
            entries = world_count * 2 * len(DEXDUET_REGIONS)
            terms.region_dist = wp.zeros(entries, dtype=wp.float32, device=device)
            terms.region_spot = wp.zeros(entries, dtype=wp.vec3, device=device)
            terms.region_grid_dist = wp.zeros(entries, dtype=wp.float32, device=device)
            terms.region_touching = wp.zeros(entries, dtype=wp.int32, device=device)
            terms.ho_contact_hand = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
        # A disabled term's side buffer is never read (its column is -1), but the kernel needs a valid array.
        placeholder = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
        if terms.hand_graph_side is None:
            terms.hand_graph_side = placeholder
        if terms.ho_contact_hand is None:
            terms.ho_contact_hand = placeholder
        print(
            "[dexduet] terms "
            + json.dumps(
                {
                    "sequence_id": sidecars.sequence_id,
                    "hand_graph_column": terms.hand_graph_column,
                    "ho_contact_column": terms.ho_contact_column,
                    "sha256": terms.sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return terms

    def evaluate(self, state, timestep, contact, pre_terms, weights, terms, score) -> None:
        if self.hand_graph_column >= 0:
            wp.launch(
                dexduet_hand_graph,
                dim=self.world_count * 2,
                inputs=[
                    state.body_q,
                    timestep,
                    self.object_body_id,
                    self.bodies_per_world,
                    self.num_frames,
                    self.node_body_ids,
                    self.tip_offsets,
                    self.palm_axis,
                    self.target_palm,
                    self.target_fingers,
                    self.target_gaps,
                    self.finger_weights,
                    self.beta_dir,
                    self.beta_gap,
                ],
                outputs=[
                    self.hand_graph_side,
                    self.hand_graph_palm_err,
                    self.hand_graph_finger_err,
                    self.hand_graph_gap_err,
                ],
            )
        if self.ho_contact_column >= 0:
            nx, ny, nz = self.grid_dims
            wp.launch(
                dexduet_ho_contact_regions,
                dim=self.world_count * 2 * len(DEXDUET_REGIONS),
                inputs=[
                    state.body_q,
                    self.object_body_id,
                    self.bodies_per_world,
                    self.points,
                    self.point_body_ids,
                    self.points_per_side,
                    self.region_start,
                    self.region_count,
                    len(DEXDUET_REGIONS),
                    self.region_slots,
                    contact.contact_active,
                    contact.contact_pos_o,
                    self.slots_per_world,
                    self.grid_dist,
                    self.grid_near,
                    wp.vec3(*self.grid_origin),
                    self.grid_spacing,
                    nx,
                    ny,
                    nz,
                ],
                outputs=[self.region_dist, self.region_spot, self.region_grid_dist, self.region_touching],
            )
            wp.launch(
                dexduet_ho_contact_score,
                dim=self.world_count * 2,
                inputs=[
                    timestep,
                    self.num_frames,
                    len(DEXDUET_REGIONS),
                    self.region_dist,
                    self.region_spot,
                    self.target_dist,
                    self.target_spot,
                    self.cap,
                    self.tol,
                    self.scale,
                    self.touch,
                    self.near,
                ],
                outputs=[self.ho_contact_hand],
            )
        wp.launch(
            append_dexduet_terms,
            dim=self.world_count,
            inputs=[
                pre_terms,
                self.pre_count,
                self.term_count,
                self.hand_graph_column,
                self.ho_contact_column,
                self.hand_graph_side,
                self.ho_contact_hand,
                weights,
                self.frame_dt,
            ],
            outputs=[terms, score],
        )
