# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""The Warp kernels against DexDuet's own torch functions.

``fixtures/dexduet_parity.npz`` holds the outputs of the unchanged torch functions of the DexDuet repository
(github.com/HaoranZhangumich/DexDuet): RewardModule.hand_graph_palm_frame / compute_hand_graph_reward; ho_contact
PartDistanceGrid.query, anchor_contacts, HandObjectContact.robot_distances and score. The same random link/object
poses, points, targets and simulator contacts go through objectives/dexduet.py; |delta| <= 1e-5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dexduet_parity.npz"
GRAPH = ("palm", "thumb_0", "index_0", "middle_0", "thumb_2", "index_1", "middle_1")
REGION_LINKS = (("palm",), ("thumb_1", "thumb_2"), ("index_1",), ("middle_1",))


def q_rot(q, v):
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _launch():
    import warp as wp

    from flash_chord.objectives.dexduet import (
        dexduet_hand_graph,
        dexduet_ho_contact_regions,
        dexduet_ho_contact_score,
    )

    z = {key: np.asarray(value) for key, value in np.load(FIXTURE).items()}
    links = [str(name) for name in z["links"]]
    n = z["obj_pos"].shape[0]
    frames = z["t_palm"].shape[0]
    bodies = 1 + 2 * len(links)  # 0 object, then left links, then right links (fixture link order)
    body = lambda side, name: 1 + side * len(links) + links.index(name)  # noqa: E731
    transforms = []
    for world in range(n):
        transforms.append(wp.transform(wp.vec3(*z["obj_pos"][world]), wp.quat(*z["obj_q"][world])))
        for side in range(2):
            for k in range(len(links)):
                transforms.append(wp.transform(wp.vec3(*z["link_pos"][world, side, k]), wp.quat(*z["link_q"][world, side, k])))
    f32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float32).reshape(-1), dtype=wp.float32)  # noqa: E731
    i32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.int32).reshape(-1), dtype=wp.int32)  # noqa: E731
    v3 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3)  # noqa: E731
    out = {}
    with wp.ScopedDevice("cpu"):
        body_q = wp.array(transforms, dtype=wp.transform)
        timestep = i32(z["timestep"])
        side_out = [wp.zeros(n * 2, dtype=wp.float32) for _ in range(4)]
        wp.launch(
            dexduet_hand_graph,
            dim=n * 2,
            inputs=[
                body_q, timestep, 0, bodies, frames,
                i32([body(side, name) for side in range(2) for name in GRAPH]),
                v3(z["tips"]), v3(z["axis"]), v3(z["t_palm"]), v3(z["t_fingers"]), f32(z["t_gaps"]),
                f32(np.array([0.2, 0.4, 0.4])), 5.0, 20.0,
            ],
            outputs=side_out,
        )
        out["hg"] = [array.numpy().reshape(n, 2) for array in side_out]

        slices = z["slices"]
        local = z["local_points"]
        point_body = [body(side, links[int(k)]) for side in range(2) for k in z["point_link"]]
        region_slots = []
        for side in range(2):
            for region_links in REGION_LINKS:
                slots = [side * len(links) + links.index(name) for name in region_links]
                region_slots.extend(slots + [-1] * (2 - len(slots)))
        obj_q = z["obj_q"].astype(np.float64)
        conj = np.concatenate([-obj_q[:, :3], obj_q[:, 3:]], -1)
        contact_o = q_rot(conj[:, None, None], z["contact_w"].astype(np.float64) - z["obj_pos"][:, None, None])
        entries = n * 2 * 4
        region = [wp.zeros(entries, dtype=wp.float32), wp.zeros(entries, dtype=wp.vec3), wp.zeros(entries, dtype=wp.float32),
                  wp.zeros(entries, dtype=wp.int32)]
        dims = tuple(int(v) for v in z["grid_dims"])
        wp.launch(
            dexduet_ho_contact_regions,
            dim=entries,
            inputs=[
                body_q, 0, bodies, v3(np.concatenate([local, local])), i32(point_body), len(local),
                i32(np.concatenate([slices[:, 0], slices[:, 0]])), i32(np.concatenate([slices[:, 1] - slices[:, 0]] * 2)), 4,
                i32(region_slots), i32(z["valid"].astype(np.int32)), v3(contact_o), 2 * len(links),
                f32(z["grid_dist"]), v3(z["grid_nearest"]), wp.vec3(*z["grid_origin"]), float(z["grid_spacing"]), *dims,
            ],
            outputs=region,
        )
        hand = wp.zeros(n * 2, dtype=wp.float32)
        wp.launch(
            dexduet_ho_contact_score,
            dim=n * 2,
            inputs=[timestep, frames, 4, region[0], region[1], f32(z["d_h"]), v3(z["s_h"]), 0.05, 0.01, 0.02, 0.01, 0.5],
            outputs=[hand],
        )
        out["region"] = (region[0].numpy().reshape(n, 2, 4), region[1].numpy().reshape(n, 2, 4, 3),
                         region[2].numpy().reshape(n, 2, 4), region[3].numpy().reshape(n, 2, 4))
        out["hand"] = hand.numpy().reshape(n, 2)
    return z, out


def test_kernels_match_dexduet_torch_functions():
    z, out = _launch()
    reward, palm_err, finger_err, gap_err = out["hg"]
    d_r, s_r, grid_d, touching = out["region"]
    for s, side in enumerate(("left", "right")):
        np.testing.assert_allclose(finger_err[:, s], z[f"hg_finger_err_{side}"], atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(gap_err[:, s], z[f"hg_gap_err_{side}"], atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(0.5 * (palm_err[:, s] + finger_err[:, s]), z[f"hg_dir_err_{side}"], atol=1.0e-5, rtol=0)
        expected = np.exp(-5.0 * z[f"hg_dir_err_{side}"].astype(np.float64) - 20.0 * z[f"hg_gap_err_{side}"])
        np.testing.assert_allclose(reward[:, s], expected, atol=1.0e-5, rtol=1.0e-4)
        np.testing.assert_allclose(grid_d[:, s], z[f"hoc_grid_dist_{side}"], atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(d_r[:, s], z[f"hoc_dist_{side}"], atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(s_r[:, s], z[f"hoc_spot_{side}"], atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(out["hand"][:, s], z[f"hoc_hand_{side}"], atol=1.0e-5, rtol=0)
    # DexDuet's hand-graph reward is the weighted mean of the two hands (weight 1 in the fixture)
    np.testing.assert_allclose(reward.mean(axis=1), z["hg_reward_mean"], atol=1.0e-5, rtol=1.0e-4)
    # the fixture exercises overrides (distal thumb link of world 5), the off-grid border (world 3)
    assert touching.sum() > 0 and touching[5, 0, 1] == 1
    assert np.all(grid_d[3, 0] > 0.05)
