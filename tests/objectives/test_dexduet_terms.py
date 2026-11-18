# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""DexDuet hand terms (objectives/dexduet.py) inside the ReconBody objective: enabled behaviour and formula parity.

The references below are float64 NumPy transcriptions of DexDuet (github.com/HaoranZhangumich/DexDuet):
``dexmachina/envs/rewards.py`` ``hand_graph_palm_frame`` (L654-672) and ``compute_hand_graph_reward`` (L674-714);
``dexmachina/envs/ho_contact.py`` ``PartDistanceGrid.query`` (grid_sample trilinear, align_corners=True, border
padding), ``anchor_contacts`` (L271-285), ``HandObjectContact.robot_distances`` (L302-324) and ``score`` (L326-339);
with one object part, the Dex3 nodes and regions, and the calibrated palm axis (R_wrist a). They share no code with
the Warp kernels. tests/objectives/test_dexduet_parity.py checks the same kernels against DexDuet's own torch
functions (fixture computed with DexDuet's code).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

STOCK_TERMS = 12
WORLDS = 7
FRAMES = 9
BODIES = 21  # 0 pelvis, 1-2 stock wrists, 3 object, 4 spare, 5-12 left hand links, 13-20 right hand links
OBJECT = 3
SLOTS = 16  # 8 contact links per hand (palm, thumb_0..2, middle_0..1, index_0..1), as G1 Dex3
# hand link order per side (body offset from 5 + 8 * side), and their contact-link index (G1 Dex3 link_geometry order)
LINKS = ("palm", "thumb_0", "index_0", "middle_0", "thumb_1", "thumb_2", "index_1", "middle_1")
CONTACT_LINK_INDEX = {"palm": 0, "thumb_0": 1, "thumb_1": 2, "thumb_2": 3, "middle_0": 4, "middle_1": 5, "index_0": 6, "index_1": 7}
GRAPH = ("palm", "thumb_0", "index_0", "middle_0", "thumb_2", "index_1", "middle_1")
REGIONS = (("palm", ("palm",)), ("thumb", ("thumb_1", "thumb_2")), ("index", ("index_1",)), ("middle", ("middle_1",)))
POINTS_PER_LINK = {"palm": 8, "thumb_1": 3, "thumb_2": 3, "index_1": 4, "middle_1": 4}
SPHERE = 0.1  # the test object: a sphere of radius 0.1 m at the object origin, on a 1 cm grid


def body_id(side: int, link: str) -> int:
    return 5 + 8 * side + LINKS.index(link)


# --------------------------------------------------------------------------------------------------------------------
# NumPy references (float64)
# --------------------------------------------------------------------------------------------------------------------


def q_normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def q_rotate(q, v):
    """Rotate v by the xyzw quaternion q (q assumed unit)."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def q_conj(q):
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.0e-6)


def ref_hand_graph(pos, quat, timestep, axis, tips, t_palm, t_fingers, t_gaps, finger_w, beta_dir, beta_gap):
    """DexDuet compute_hand_graph_reward per (world, side) -> reward, palm_err, finger_err, gap_err (W, 2)."""
    out = np.zeros((4, WORLDS, 2))
    for world in range(WORLDS):
        frame = min(max(int(timestep[world]), 0), FRAMES - 1)
        rot = q_normalize(quat[world, OBJECT])
        for side in range(2):
            node = {link: pos[world, body_id(side, link)] for link in GRAPH}
            tip = [
                pos[world, body_id(side, link)] + q_rotate(quat[world, body_id(side, link)], tips[side, f])
                for f, link in enumerate(("thumb_2", "index_1", "middle_1"))
            ]
            base = [node["thumb_0"], node["index_0"], node["middle_0"]]
            palm_q = q_normalize(quat[world, body_id(side, "palm")])
            normal = unit(q_rotate(palm_q, axis[side]))
            forward = unit(0.5 * (node["index_0"] + node["middle_0"]) - node["palm"])
            palm = np.stack([normal, forward])
            palm_cos = np.clip((palm * q_rotate(rot, t_palm[frame, side])).sum(-1), -1.0, 1.0)
            fingers = unit(np.stack(tip) - np.stack(base))
            finger_cos = np.clip((fingers * q_rotate(rot, t_fingers[frame, side])).sum(-1), -1.0, 1.0)
            palm_err = (1.0 - palm_cos).mean()
            finger_err = ((1.0 - finger_cos) * finger_w).sum()
            dir_err = 0.5 * (palm_err + finger_err)
            gaps = np.array([np.linalg.norm(tip[a] - tip[b]) for a, b in ((0, 1), (0, 2), (1, 2))])
            gap_err = np.abs(gaps - t_gaps[frame, side]).mean()
            out[:, world, side] = (math.exp(-beta_dir * dir_err - beta_gap * gap_err), palm_err, finger_err, gap_err)
    return out


def ref_grid_query(grid, points):
    """PartDistanceGrid.query: trilinear with align_corners=True and border padding."""
    dims = np.asarray(grid["dims"])
    u = np.clip((points - grid["origin"]) / grid["spacing"], 0.0, dims - 1)
    i = np.minimum(np.floor(u).astype(int), dims - 2)
    f = u - i
    dist, near = 0.0, np.zeros(3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                dist += w * grid["dist"][i[0] + dx, i[1] + dy, i[2] + dz]
                near += w * grid["nearest"][i[0] + dx, i[1] + dy, i[2] + dz]
    return dist, near


def ref_ho_contact(pos, quat, timestep, points, point_links, active, contact_o, d_h, s_h, grid, params):
    """robot_distances + anchor_contacts + score per (world, side) -> hand score (W, 2), d_r, s_r, grid d (W, 2, 4)."""
    cap, tol, scale, touch, near = params
    hand = np.zeros((WORLDS, 2))
    d_r = np.zeros((WORLDS, 2, 4))
    s_r = np.zeros((WORLDS, 2, 4, 3))
    grid_d = np.zeros((WORLDS, 2, 4))
    for world in range(WORLDS):
        frame = min(max(int(timestep[world]), 0), FRAMES - 1)
        obj_p, obj_q = pos[world, OBJECT], quat[world, OBJECT]
        for side in range(2):
            for r, (_, links) in enumerate(REGIONS):
                best, best_point = np.inf, None
                for point, link in zip(points[side][r], point_links[side][r]):
                    b = body_id(side, link)
                    world_point = pos[world, b] + q_rotate(quat[world, b], point)
                    local = q_rotate(q_conj(obj_q), world_point - obj_p)
                    d, _ = ref_grid_query(grid, local)
                    if d < best:
                        best, best_point = d, local
                spot = ref_grid_query(grid, best_point)[1]
                grid_d[world, side, r] = best
                touching = False
                for link in links:  # proximal to distal: the most distal touching link wins
                    slot = 8 * side + CONTACT_LINK_INDEX[link]
                    if active[world, slot]:
                        touching = True
                        spot = contact_o[world, slot]
                d_r[world, side, r] = 0.0 if touching else best
                s_r[world, side, r] = spot
            dr = np.clip(d_r[world, side], 0.0, cap)
            dh_raw = d_h[frame, side]
            dh = np.clip(dh_raw, 0.0, cap)
            human_near = dh_raw < cap
            err = np.abs(dr - dh) + np.linalg.norm(s_r[world, side] - s_h[frame, side], axis=-1) * human_near
            score = np.exp(-np.maximum(err - tol, 0.0) / scale)
            is_touch = (d_r[world, side] < touch) | (dh_raw < touch)
            is_near = (d_r[world, side] < cap) | human_near
            weight = np.where(is_touch, 1.0, np.where(is_near, near, 0.0))
            hand[world, side] = (weight * score).sum() / max(weight.sum(), 1e-6) if weight.sum() > 0 else 1.0
    return hand, d_r, s_r, grid_d


# --------------------------------------------------------------------------------------------------------------------
# Fixture: a small CPU objective with real kernels and stand-in runtime dependencies
# --------------------------------------------------------------------------------------------------------------------


def random_quats(rng, count):
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def sphere_grid():
    spacing, count = 0.01, 41
    origin = np.full(3, -0.2)
    axes = origin[0] + spacing * np.arange(count)
    nodes = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), -1)
    radius = np.linalg.norm(nodes, axis=-1, keepdims=True)
    nearest = SPHERE * nodes / np.maximum(radius, 1e-9)
    nearest[radius[..., 0] < 1e-9] = (SPHERE, 0.0, 0.0)
    return {
        "origin": origin.astype(np.float32).astype(np.float64),
        "spacing": float(np.float32(spacing)),
        "dims": (count, count, count),
        "dist": np.abs(radius[..., 0] - SPHERE).astype(np.float32).astype(np.float64),
        "nearest": nearest.astype(np.float32).astype(np.float64),
    }


class Scenario:
    """Random hand/object poses plus edge cases; every array is float32-exact so the references see kernel inputs."""

    def __init__(self, seed: int, *, zero_tips: bool = False):
        rng = np.random.default_rng(seed)
        f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)  # noqa: E731
        self.grid = sphere_grid()
        pos = rng.normal(scale=0.4, size=(WORLDS, BODIES, 3))
        quat = random_quats(rng, WORLDS * BODIES).reshape(WORLDS, BODIES, 4)
        for world in range(WORLDS):
            obj = rng.normal(scale=0.3, size=3)
            pos[world, OBJECT] = obj
            for side in range(2):
                # hand links scattered within ~4-14 cm of the object centre, so regions touch, are near or are far
                for link in LINKS:
                    direction = rng.normal(size=3)
                    pos[world, body_id(side, link)] = obj + direction / np.linalg.norm(direction) * rng.uniform(0.07, 0.16)
        # world 5: the left hand far outside the grid (border clamp, every weight 0 when the human is far too)
        for link in LINKS:
            pos[5, body_id(0, link)] = pos[5, OBJECT] + np.array([0.6, 0.4, -0.5]) + rng.normal(scale=0.01, size=3)
        if zero_tips:
            # world 6: the left thumb tip link sits exactly on the thumb base (zero-length edge with zero tip offsets)
            pos[6, body_id(0, "thumb_2")] = pos[6, body_id(0, "thumb_0")]
        self.pos, self.quat = f32(pos), f32(quat)
        self.timestep = np.array([0, 3, 8, 11, -2, 4, 6], dtype=np.int32)[:WORLDS]
        tips = np.zeros((2, 3, 3)) if zero_tips else np.array(
            [[[0.0, -0.045, 0.0], [0.045, 0.0, 0.0], [0.045, 0.0, 0.0]], [[0.0, 0.045, 0.0], [0.045, 0.0, 0.0], [0.045, 0.0, 0.0]]]
        )
        self.tips = f32(tips)
        self.axis = f32(np.stack([unit(rng.normal(size=3)) for _ in range(2)]))
        self.t_palm = f32(unit(rng.normal(size=(FRAMES, 2, 2, 3))))
        self.t_fingers = f32(unit(rng.normal(size=(FRAMES, 2, 3, 3))))
        self.t_gaps = f32(rng.uniform(0.02, 0.12, size=(FRAMES, 2, 3)))
        # cos clamping: frame 3 left targets equal the robot's own world vectors of world 1 (cos = 1 up to rounding)
        self.points = [[], []]
        self.point_links = [[], []]
        for side in range(2):
            for _, links in REGIONS:
                pts, owners = [], []
                for link in links:
                    pts.append(rng.uniform(-0.02, 0.02, size=(POINTS_PER_LINK[link], 3)))
                    owners.extend([link] * POINTS_PER_LINK[link])
                self.points[side].append(f32(np.concatenate(pts)))
                self.point_links[side].append(owners)
        self.d_h = f32(rng.choice([0.002, 0.008, 0.02, 0.04, 0.07, 0.3], size=(FRAMES, 2, 4)))
        self.s_h = f32(unit(rng.normal(size=(FRAMES, 2, 4, 3))) * SPHERE)
        self.d_h[4, 0] = 0.3  # frame 4 (world 5): the human left hand is far too -> every weight 0 -> hand = 1
        active = np.zeros((WORLDS, SLOTS), dtype=np.int32)
        active[1, 8 * 0 + CONTACT_LINK_INDEX["thumb_1"]] = 1  # proximal thumb only
        active[2, 8 * 0 + CONTACT_LINK_INDEX["thumb_1"]] = 1  # both thumb links: the distal one wins
        active[2, 8 * 0 + CONTACT_LINK_INDEX["thumb_2"]] = 1
        active[3, 8 * 1 + CONTACT_LINK_INDEX["palm"]] = 1
        active[4, 8 * 1 + CONTACT_LINK_INDEX["index_1"]] = 1
        active[4, 8 * 1 + CONTACT_LINK_INDEX["thumb_0"]] = 1  # not a region link: ignored
        self.active = active
        self.contact_o = f32(rng.normal(scale=0.1, size=(WORLDS, SLOTS, 3)))
        self.finger_w = np.asarray(np.array([0.5, 1.0, 1.0]) / 2.5, dtype=np.float32).astype(np.float64)

    def state(self):
        import warp as wp

        transforms = [
            wp.transform(wp.vec3(*self.pos[w, b].astype(np.float32)), wp.quat(*self.quat[w, b].astype(np.float32)))
            for w in range(WORLDS)
            for b in range(BODIES)
        ]
        rng = np.random.default_rng(99)
        return (
            SimpleNamespace(
                body_q=wp.array(transforms, dtype=wp.transform),
                joint_q=wp.array(rng.normal(scale=0.3, size=WORLDS * 10).astype(np.float32), dtype=wp.float32),
            ),
            wp.array(self.timestep, dtype=wp.int32),
            wp.array((np.arange(WORLDS) % 3 == 0).astype(np.int32), dtype=wp.int32),
        )


def make_objective(config, scenario: Scenario):
    """ReconBodyObjective with real buffers/kernels; DexDuetTerms filled from the scenario (no sidecar files)."""
    import warp as wp

    from flash_chord.objectives.dexduet import DexDuetTerms
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjective
    from flash_chord.objectives.wrench import friction_cone_angles, sample_wrench_basis

    rng = np.random.default_rng(5)
    f32 = lambda values: wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32)  # noqa: E731
    i32 = lambda values: wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32)  # noqa: E731
    v3 = lambda values: wp.array(np.asarray(values, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3)  # noqa: E731
    q = 10
    reference_joint_q = rng.normal(scale=0.3, size=(FRAMES, q)).astype(np.float32)
    reference_joint_q[:, 3:7] = random_quats(rng, FRAMES)
    num_basis = 16
    contact = SimpleNamespace(
        contact_pos_o=v3(scenario.contact_o),
        contact_force_direction_o=v3(unit(rng.normal(size=(WORLDS * SLOTS, 3)))),
        contact_active=i32(scenario.active.reshape(-1)),
        hand_slot_starts=i32([0, 8]),
        link_counts=i32([8, 8]),
        layout=SimpleNamespace(slots_per_world=SLOTS, sides=("left", "right"), hand_slot_starts=(0, 8), num_objects=1),
    )
    pre_count = STOCK_TERMS + int(config.object_task_enabled)
    dexduet = None
    if config.dexduet_enabled:
        names = config.term_names
        dexduet = DexDuetTerms(
            world_count=WORLDS,
            num_frames=FRAMES,
            bodies_per_world=BODIES,
            object_body_id=OBJECT,
            frame_dt=0.02,
            pre_count=pre_count,
            term_count=len(names),
            hand_graph_column=names.index("hand_graph") if config.hand_graph_enabled else -1,
            ho_contact_column=names.index("ho_contact") if config.ho_contact_enabled else -1,
            beta_dir=config.hand_graph_beta_dir,
            beta_gap=config.hand_graph_beta_gap,
            cap=config.ho_contact_cap,
            tol=config.ho_contact_tol,
            scale=config.ho_contact_scale,
            touch=config.ho_contact_touch,
            near=config.ho_contact_near,
            sha256={},
        )
        if config.hand_graph_enabled:
            dexduet.node_body_ids = i32([body_id(side, link) for side in range(2) for link in GRAPH])
            dexduet.tip_offsets = v3(scenario.tips)
            dexduet.palm_axis = v3(scenario.axis)
            dexduet.target_palm = v3(scenario.t_palm)
            dexduet.target_fingers = v3(scenario.t_fingers)
            dexduet.target_gaps = f32(scenario.t_gaps.reshape(-1))
            dexduet.finger_weights = f32(scenario.finger_w)
            for name in ("hand_graph_side", "hand_graph_palm_err", "hand_graph_finger_err", "hand_graph_gap_err"):
                setattr(dexduet, name, wp.zeros(WORLDS * 2, dtype=wp.float32))
        if config.ho_contact_enabled:
            points, bodies, starts, counts, slots = [], [], [], [], []
            for side in range(2):
                cursor = 0
                for r, (_, links) in enumerate(REGIONS):
                    points.append(scenario.points[side][r])
                    bodies.extend(body_id(side, link) for link in scenario.point_links[side][r])
                    starts.append(cursor)
                    counts.append(len(scenario.points[side][r]))
                    cursor += len(scenario.points[side][r])
                    region_slots = [8 * side + CONTACT_LINK_INDEX[link] for link in links]
                    slots.extend(region_slots + [-1] * (2 - len(region_slots)))
            grid = scenario.grid
            dexduet.points = v3(np.concatenate(points))
            dexduet.point_body_ids = i32(bodies)
            dexduet.points_per_side = sum(counts[:4])
            dexduet.region_start = i32(starts)
            dexduet.region_count = i32(counts)
            dexduet.region_slots = i32(slots)
            dexduet.slots_per_world = SLOTS
            dexduet.grid_dist = f32(grid["dist"].reshape(-1))
            dexduet.grid_near = v3(grid["nearest"])
            dexduet.grid_origin = tuple(grid["origin"])
            dexduet.grid_spacing = grid["spacing"]
            dexduet.grid_dims = grid["dims"]
            dexduet.target_dist = f32(scenario.d_h.reshape(-1))
            dexduet.target_spot = v3(scenario.s_h)
            entries = WORLDS * 2 * 4
            dexduet.region_dist = wp.zeros(entries, dtype=wp.float32)
            dexduet.region_spot = wp.zeros(entries, dtype=wp.vec3)
            dexduet.region_grid_dist = wp.zeros(entries, dtype=wp.float32)
            dexduet.region_touching = wp.zeros(entries, dtype=wp.int32)
            dexduet.ho_contact_hand = wp.zeros(WORLDS * 2, dtype=wp.float32)
        placeholder = wp.zeros(WORLDS * 2, dtype=wp.float32)
        if dexduet.hand_graph_side is None:
            dexduet.hand_graph_side = placeholder
        if dexduet.ho_contact_hand is None:
            dexduet.ho_contact_hand = placeholder
    cone_cos, cone_sin = friction_cone_angles(8)
    return ReconBodyObjective(
        config=config,
        world_count=WORLDS,
        frame_dt=0.02,
        num_frames=FRAMES,
        num_hands=2,
        num_bodies=1,
        num_basis=num_basis,
        num_edges=8,
        num_joints=4,
        bodies_per_world=BODIES,
        q_per_world=q,
        pelvis_body_id=0,
        object_body_id=OBJECT,
        episode_start_frame=i32(rng.integers(0, 4, size=WORLDS)),
        reference_joint_q_offset=f32(rng.normal(scale=0.01, size=WORLDS * q)),
        reference=SimpleNamespace(
            joint_q=wp.array(reference_joint_q, dtype=wp.float32),
            wrist_pos_w=v3(rng.normal(scale=0.3, size=(FRAMES * 2, 3))),
            wrist_quat_w=wp.array(random_quats(rng, FRAMES * 2).astype(np.float32), dtype=wp.quat),
        ),
        command=SimpleNamespace(
            object_radius=f32(np.full(WORLDS, 0.1)),
            body_target_pos_w=v3(rng.normal(scale=0.5, size=(WORLDS, 3))),
            body_target_quat_w=wp.array(random_quats(rng, WORLDS).astype(np.float32), dtype=wp.quat),
        ),
        contact=contact,
        action=SimpleNamespace(
            action_rate_l2=f32(rng.uniform(0.0, 2.0, size=WORLDS)),
            action_l2=f32(rng.uniform(0.0, 2.0, size=WORLDS)),
            sonic_q_ids=i32([7, 8, 9]),
        ),
        wrist_frames=SimpleNamespace(
            body_ids=i32([1, 2]),
            body_to_frame_pos=v3(rng.normal(scale=0.05, size=(2, 3))),
            body_to_frame_quat=wp.array(random_quats(rng, 2).astype(np.float32), dtype=wp.quat),
        ),
        scalar_q_ids=i32([6, 7, 8, 9]),
        soft_lower=f32([-0.2, -0.3, -0.1, -0.4]),
        soft_upper=f32([0.2, 0.3, 0.1, 0.4]),
        contact_active=f32((rng.uniform(size=(FRAMES, 2)) > 0.3).astype(np.float32).reshape(-1)),
        wrench_basis=wp.array(sample_wrench_basis(num_basis, 1.0, 0), dtype=wp.spatial_vector),
        cone_cos=f32(cone_cos),
        cone_sin=f32(cone_sin),
        wrench_contact_enabled=wp.ones(SLOTS, dtype=wp.int32),
        current_wrench_support=wp.zeros(WORLDS * 2 * num_basis, dtype=wp.float32),
        force_closure=wp.zeros(WORLDS, dtype=wp.float32),
        weights=wp.array(config.term_weights, dtype=wp.float32),
        terms=wp.zeros(WORLDS * len(config.term_names), dtype=wp.float32),
        score=wp.zeros(WORLDS, dtype=wp.float32),
        stock_terms=(
            wp.zeros(WORLDS * len(RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32) if config.object_task_enabled else None
        ),
        pre_dexduet_terms=wp.zeros(WORLDS * pre_count, dtype=wp.float32) if dexduet is not None else None,
        dexduet=dexduet,
    )


def record_launches(monkeypatch):
    import warp as wp

    launches = []
    original = wp.launch

    def recording(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", recording)
    return launches


OBJECT_TASK_KWARGS = dict(object_task_weight=5.0, object_task_position_beta=10.0, object_task_rotation_beta=1.0)
INTERACTION_WEIGHTS = dict(hand_graph_weight=2.5, ho_contact_weight=5.0)


def run(config, scenario):
    import warp as wp

    with wp.ScopedDevice("cpu"):
        objective = make_objective(config, scenario)
        state, timestep, terminated = scenario.state()
        objective.evaluate(state, timestep, terminated)
        return objective, objective.terms.numpy().reshape(WORLDS, -1).copy(), objective.score.numpy().copy()


def references(scenario, config):
    hg = ref_hand_graph(
        scenario.pos, scenario.quat, scenario.timestep, scenario.axis, scenario.tips, scenario.t_palm,
        scenario.t_fingers, scenario.t_gaps, scenario.finger_w, config.hand_graph_beta_dir, config.hand_graph_beta_gap,
    )
    params = (config.ho_contact_cap, config.ho_contact_tol, config.ho_contact_scale, config.ho_contact_touch,
              config.ho_contact_near)
    ho = ref_ho_contact(
        scenario.pos, scenario.quat, scenario.timestep, scenario.points, scenario.point_links, scenario.active,
        scenario.contact_o, scenario.d_h, scenario.s_h, scenario.grid, params,
    )
    return hg, ho


# --------------------------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "names", "kernels"),
    [
        (INTERACTION_WEIGHTS, ("hand_graph", "ho_contact"),
         ["dexduet_hand_graph", "dexduet_ho_contact_regions", "dexduet_ho_contact_score", "append_dexduet_terms"]),
        ({"hand_graph_weight": 2.5}, ("hand_graph",), ["dexduet_hand_graph", "append_dexduet_terms"]),
        ({"ho_contact_weight": 5.0}, ("ho_contact",),
         ["dexduet_ho_contact_regions", "dexduet_ho_contact_score", "append_dexduet_terms"]),
    ],
)
@pytest.mark.parametrize("object_task", [True, False])
def test_enabled_terms_append_after_the_earlier_columns(monkeypatch, extra, names, kernels, object_task):
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    base_kwargs = OBJECT_TASK_KWARGS if object_task else {}
    off_config = ReconBodyObjectiveConfig(**base_kwargs)
    on_config = ReconBodyObjectiveConfig(**base_kwargs, **extra)
    pre = STOCK_TERMS + int(object_task)
    assert on_config.term_names == off_config.term_names + names
    assert on_config.term_names[:STOCK_TERMS] == RECON_BODY_OBJECTIVE_TERM_NAMES
    assert on_config.term_weights[:pre] == off_config.term_weights
    scenario = Scenario(seed=11)
    launches = record_launches(monkeypatch)
    _, off_terms, off_score = run(off_config, scenario)
    off_launches = list(launches)
    del launches[:]
    objective, on_terms, on_score = run(on_config, scenario)
    assert launches == off_launches + kernels
    assert on_terms.shape == (WORLDS, pre + len(names))
    np.testing.assert_array_equal(on_terms[:, :pre], off_terms)  # earlier columns: bitwise
    hg, ho = references(scenario, on_config)
    expected = {"hand_graph": hg[0].mean(axis=1), "ho_contact": ho[0].mean(axis=1)}
    added = np.zeros(WORLDS)
    for offset, name in enumerate(names):
        np.testing.assert_allclose(on_terms[:, pre + offset], expected[name], atol=1.0e-5, rtol=0)
        added += 0.02 * on_config.term_weights[pre + offset] * on_terms[:, pre + offset].astype(np.float64)
    np.testing.assert_allclose(on_score, off_score + added, rtol=1.0e-6, atol=1.0e-7)
    assert np.all(on_terms[:, pre:] > 0.0) and np.all(on_terms[:, pre:] <= 1.0)


def test_hand_graph_matches_dexduet_formula_per_side():
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    for seed in (1, 2, 3):
        scenario = Scenario(seed)
        config = ReconBodyObjectiveConfig(**OBJECT_TASK_KWARGS, **INTERACTION_WEIGHTS)
        objective, _, _ = run(config, scenario)
        hg, _ = references(scenario, config)
        d = objective.dexduet
        for k, array in enumerate((d.hand_graph_side, d.hand_graph_palm_err, d.hand_graph_finger_err, d.hand_graph_gap_err)):
            np.testing.assert_allclose(array.numpy().reshape(WORLDS, 2), hg[k], atol=1.0e-5, rtol=0)


def test_hand_graph_zero_length_edge_and_exact_alignment():
    """A zero-length finger edge gives unit() = 0 (1 - cos = 1, DexDuet's clamp(min=1e-6)); aligned targets clamp to 1."""
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    scenario = Scenario(4, zero_tips=True)
    config = ReconBodyObjectiveConfig(hand_graph_weight=1.0)
    # frame 6 (world 6) left targets: the robot's own normal/forward/edges in the object frame (cos = 1 before clamping)
    world, side, frame = 6, 0, 6
    pos, quat = scenario.pos[world], scenario.quat[world]
    rot = q_normalize(quat[OBJECT])
    palm_q = q_normalize(quat[body_id(side, "palm")])
    normal = unit(q_rotate(palm_q, scenario.axis[side]))
    forward = unit(0.5 * (pos[body_id(side, "index_0")] + pos[body_id(side, "middle_0")]) - pos[body_id(side, "palm")])
    scenario.t_palm[frame, side] = np.asarray(q_rotate(q_conj(rot), np.stack([normal, forward])), np.float32)
    objective, _, _ = run(config, scenario)
    hg, _ = references(scenario, config)
    d = objective.dexduet
    finger_err = d.hand_graph_finger_err.numpy().reshape(WORLDS, 2)
    palm_err = d.hand_graph_palm_err.numpy().reshape(WORLDS, 2)
    np.testing.assert_allclose(finger_err, hg[2], atol=1.0e-5, rtol=0)
    # the thumb edge of world 6 has zero length: its 1 - cos is exactly 1 (weight 0.2) next to the other two fingers
    rest = 0.0
    for f, (tip, base) in enumerate((("index_1", "index_0"), ("middle_1", "middle_0")), start=1):
        edge = unit(pos[body_id(side, tip)] - pos[body_id(side, base)])
        target = q_rotate(rot, scenario.t_fingers[frame, side, f])
        rest += 0.4 * (1.0 - np.clip(np.dot(edge, target), -1.0, 1.0))
    assert finger_err[world, side] == pytest.approx(0.2 + rest, abs=1.0e-5)
    assert palm_err[world, side] == pytest.approx(0.0, abs=2.0e-6)  # aligned palm: clamp keeps 1 - cos >= 0
    assert np.all(palm_err >= 0.0) and np.all(finger_err >= 0.0)


def test_ho_contact_matches_dexduet_formula_with_overrides_and_border():
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    for seed in (1, 2, 3):
        scenario = Scenario(seed)
        config = ReconBodyObjectiveConfig(**OBJECT_TASK_KWARGS, **INTERACTION_WEIGHTS)
        objective, _, _ = run(config, scenario)
        _, (hand, d_r, s_r, grid_d) = references(scenario, config)
        d = objective.dexduet
        np.testing.assert_allclose(d.region_grid_dist.numpy().reshape(WORLDS, 2, 4), grid_d, atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(d.region_dist.numpy().reshape(WORLDS, 2, 4), d_r, atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(d.region_spot.numpy().reshape(WORLDS, 2, 4, 3), s_r, atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(d.ho_contact_hand.numpy().reshape(WORLDS, 2), hand, atol=1.0e-5, rtol=0)
        touching = d.region_touching.numpy().reshape(WORLDS, 2, 4)
        # overrides: world 1 proximal thumb, world 2 both thumb links (distal contact point), world 3 right palm,
        # world 4 right index (the right thumb_0 contact is not a region link)
        assert touching[1, 0, 1] == 1 and touching[2, 0, 1] == 1 and touching[3, 1, 0] == 1 and touching[4, 1, 2] == 1
        assert touching.sum() == 4
        thumb2 = 8 * 0 + CONTACT_LINK_INDEX["thumb_2"]
        np.testing.assert_allclose(d.region_spot.numpy().reshape(WORLDS, 2, 4, 3)[2, 0, 1], scenario.contact_o[2, thumb2], atol=0)
        assert d.region_dist.numpy().reshape(WORLDS, 2, 4)[2, 0, 1] == 0.0
        # world 5: the left hand is outside the grid (border values > cap) and the human is far: every weight 0 -> 1
        assert np.all(d.region_grid_dist.numpy().reshape(WORLDS, 2, 4)[5, 0] > config.ho_contact_cap)
        assert d.ho_contact_hand.numpy().reshape(WORLDS, 2)[5, 0] == 1.0


def test_set_weights_requires_exactly_the_active_names():
    import warp as wp

    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    stock = {name: float(i + 1) for i, name in enumerate(RECON_BODY_OBJECTIVE_TERM_NAMES)}
    with wp.ScopedDevice("cpu"):
        objective = make_objective(ReconBodyObjectiveConfig(**OBJECT_TASK_KWARGS, **INTERACTION_WEIGHTS), Scenario(0))
        np.testing.assert_array_equal(objective.weights.numpy()[-3:], [5.0, 2.5, 5.0])
        for missing in ({**stock, "object_task": 5.0}, {**stock, "object_task": 5.0, "hand_graph": 1.0}):
            with pytest.raises(ValueError, match="exactly"):
                objective.set_weights(missing)
        objective.set_weights({**stock, "object_task": 5.0, "hand_graph": 7.0, "ho_contact": 9.0})
        np.testing.assert_array_equal(objective.weights.numpy()[-3:], [5.0, 7.0, 9.0])


def test_config_defaults_are_off_and_invalid_values_are_rejected():
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    config = ReconBodyObjectiveConfig()
    assert not config.dexduet_enabled and not config.hand_graph_enabled and not config.ho_contact_enabled
    assert (config.hand_graph_beta_dir, config.hand_graph_beta_gap, config.hand_graph_thumb_weight) == (5.0, 20.0, 0.5)
    assert (config.hand_graph_gap_scale, config.hand_graph_tip_offset) == (1.5, 0.045)
    assert (config.ho_contact_cap, config.ho_contact_tol, config.ho_contact_scale) == (0.05, 0.01, 0.02)
    assert (config.ho_contact_touch, config.ho_contact_near, config.dexduet_targets_dir) == (0.01, 0.5, "")
    for kwargs in (
        {"hand_graph_weight": math.nan},
        {"ho_contact_weight": math.inf},
        {"hand_graph_beta_dir": 0.0},
        {"hand_graph_gap_scale": -1.0},
        {"ho_contact_touch": 0.06},
        {"ho_contact_scale": 0.0},
        {"ho_contact_near": 1.5},
    ):
        with pytest.raises(ValueError, match="DexDuet"):
            ReconBodyObjectiveConfig(**kwargs)


def test_tip_offsets_and_finger_weights_follow_dexduet_dex3():
    from flash_chord.objectives.dexduet import dexduet_finger_weights, dexduet_tip_offsets

    tips = dexduet_tip_offsets(0.045)
    np.testing.assert_allclose(tips[0], [[0, -0.045, 0], [0.045, 0, 0], [0.045, 0, 0]], atol=1e-7)  # dex3_left_6dof.urdf
    np.testing.assert_allclose(tips[1], [[0, 0.045, 0], [0.045, 0, 0], [0.045, 0, 0]], atol=1e-7)  # dex3_right_6dof.urdf
    np.testing.assert_allclose(dexduet_finger_weights(0.5), [0.2, 0.4, 0.4], atol=1e-7)
