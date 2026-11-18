# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""The DexDuet sidecar loader refuses a wrong episode, frame count, gap scale, grid or hash."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from flash_chord.objectives.dexduet import load_dexduet_sidecars

SEQ = "2026-01-01_00-00-00_test_object_01"
T = 6


def _write(root, *, sequence=SEQ, frames=T, gap_scale=1.5, margin=0.06):
    rng = np.random.default_rng(0)
    episode = root / SEQ
    episode.mkdir(parents=True, exist_ok=True)
    axis = rng.normal(size=(2, 3))
    np.savez(
        episode / "dexduet_targets.npz",
        sequence_id=np.asarray(sequence),
        T=np.asarray(frames),
        fps=np.asarray(30.0),
        sides=np.asarray(["left", "right"]),
        fingers=np.asarray(["thumb", "index", "middle"]),
        regions=np.asarray(["palm", "thumb", "index", "middle"]),
        hg_palm=rng.normal(size=(frames, 2, 2, 3)).astype(np.float32),
        hg_fingers=rng.normal(size=(frames, 2, 3, 3)).astype(np.float32),
        hg_gaps=rng.uniform(size=(frames, 2, 3)).astype(np.float32),
        hg_gap_scale=np.asarray(gap_scale),
        palm_axis=(axis / np.linalg.norm(axis, axis=-1, keepdims=True)).astype(np.float32),
        palm_axis_resultant=np.asarray([0.99, 0.98]),
        ho_dist=rng.uniform(size=(frames, 2, 4)).astype(np.float32),
        ho_spot=rng.normal(size=(frames, 2, 4, 3)).astype(np.float32),
    )
    np.savez(
        episode / "object_grid.npz",
        origin=np.zeros(3, np.float32),
        spacing=np.float32(0.003),
        margin=np.float32(margin),
        dims=np.asarray([3, 4, 5]),
        dist=np.ones((3, 4, 5), np.float32),
        nearest=np.zeros((3, 4, 5, 3), np.float32),
    )
    points = {}
    for side in ("left", "right"):
        points[f"points_{side}"] = rng.normal(size=(10, 3)).astype(np.float32)
        points[f"links_{side}"] = np.asarray([f"{side}_hand_palm_link", f"{side}_hand_thumb_1_link"])
        points[f"point_link_{side}"] = np.asarray([0] * 4 + [1] * 6)
        points[f"region_slices_{side}"] = np.asarray([[0, 4], [4, 6], [6, 8], [8, 10]])
    np.savez(root / "dex3_region_points.npz", regions=np.asarray(["palm", "thumb", "index", "middle"]), **points)
    lines = []
    for path in sorted(root.rglob("*.npz")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "TARGETS.sha256").write_text("\n".join(lines) + "\n")


def _load(root, sequence=SEQ, frames=T, gap_scale=1.5):
    return load_dexduet_sidecars(
        str(root), sequence, frames, hand_graph=True, ho_contact=True, gap_scale=gap_scale, contact_cap=0.05
    )


def test_matching_sidecars_load_with_their_hashes(tmp_path):
    _write(tmp_path)
    sidecars = _load(tmp_path)
    assert sidecars.hg_palm.shape == (T, 2, 2, 3) and sidecars.ho_spot.shape == (T, 2, 4, 3)
    assert set(sidecars.sha256) == {f"{SEQ}/dexduet_targets.npz", f"{SEQ}/object_grid.npz", "dex3_region_points.npz"}


def test_wrong_episode_or_frame_count_is_refused(tmp_path):
    _write(tmp_path, sequence="another_episode")
    with pytest.raises(ValueError, match="targets are for"):
        _load(tmp_path)
    _write(tmp_path)
    with pytest.raises(ValueError, match="T=6"):
        _load(tmp_path, frames=T + 1)
    with pytest.raises(FileNotFoundError):
        _load(tmp_path, sequence="missing_episode")


def test_gap_scale_grid_margin_and_hash_mismatches_are_refused(tmp_path):
    _write(tmp_path, gap_scale=1.2)
    with pytest.raises(ValueError, match="scale"):
        _load(tmp_path)
    _write(tmp_path, margin=0.03)
    with pytest.raises(ValueError, match="margin"):
        _load(tmp_path)
    _write(tmp_path)
    with open(tmp_path / SEQ / "object_grid.npz", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="sha256"):
        _load(tmp_path)
    (tmp_path / "TARGETS.sha256").unlink()
    with pytest.raises(FileNotFoundError, match="TARGETS.sha256"):
        _load(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        load_dexduet_sidecars("", SEQ, T, hand_graph=True, ho_contact=False, gap_scale=1.5, contact_cap=0.05)
