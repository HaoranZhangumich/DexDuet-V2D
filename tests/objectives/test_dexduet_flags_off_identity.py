# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""With the DexDuet weights at their default 0, the ReconBody objective is bit-identical to the one before them.

``fixtures/recon_body_before_dexduet.py`` is a copy of ``src/flash_chord/objectives/recon_body.py`` before the DexDuet
rewards were added (NVIDIA's objective plus the object-tracking term ``object_task``), with only its comments edited;
FROZEN_BLOB pins it by its git blob id. Both objectives get identical random inputs and must launch the same kernels
(stock; stock + object_task) and produce bit-equal terms and scores.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FROZEN = Path(__file__).resolve().parent / "fixtures" / "recon_body_before_dexduet.py"
FROZEN_BLOB = "1c5768680237f855d9fa32fa006a2348589ca6cf"
WORLDS, FRAMES, BODIES, Q, SLOTS = 9, 13, 5, 10, 4


def frozen_module():
    name = "flash_chord_frozen_recon_body_before_dexduet"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, FROZEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_copy_is_the_pinned_blob():
    data = FROZEN.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == FROZEN_BLOB


def _random_quats(rng, count):
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _objective(module, config, seed=0):
    import warp as wp

    from flash_chord.objectives.wrench import friction_cone_angles, sample_wrench_basis

    rng = np.random.default_rng(seed)
    f32 = lambda values: wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32)  # noqa: E731
    i32 = lambda values: wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32)  # noqa: E731
    reference_joint_q = rng.normal(scale=0.3, size=(FRAMES, Q)).astype(np.float32)
    reference_joint_q[:, 3:7] = _random_quats(rng, FRAMES)
    contact_pos = rng.normal(scale=0.05, size=(WORLDS * SLOTS, 3)).astype(np.float32)
    contact_dir = rng.normal(size=(WORLDS * SLOTS, 3))
    contact_dir = (contact_dir / np.linalg.norm(contact_dir, axis=-1, keepdims=True)).astype(np.float32)
    num_basis = 32
    cone_cos, cone_sin = friction_cone_angles(8)
    kwargs = dict(
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
        q_per_world=Q,
        pelvis_body_id=0,
        object_body_id=3,
        episode_start_frame=i32(rng.integers(0, 4, size=WORLDS)),
        reference_joint_q_offset=f32(rng.normal(scale=0.01, size=WORLDS * Q)),
        reference=SimpleNamespace(
            joint_q=wp.array(reference_joint_q, dtype=wp.float32),
            wrist_pos_w=wp.array(rng.normal(scale=0.3, size=(FRAMES * 2, 3)).astype(np.float32), dtype=wp.vec3),
            wrist_quat_w=wp.array(_random_quats(rng, FRAMES * 2).astype(np.float32), dtype=wp.quat),
        ),
        command=SimpleNamespace(
            object_radius=f32(np.full(WORLDS, 0.1)),
            body_target_pos_w=wp.array(rng.normal(scale=0.5, size=(WORLDS, 3)).astype(np.float32), dtype=wp.vec3),
            body_target_quat_w=wp.array(_random_quats(rng, WORLDS).astype(np.float32), dtype=wp.quat),
        ),
        contact=SimpleNamespace(
            contact_pos_o=wp.array(contact_pos, dtype=wp.vec3),
            contact_force_direction_o=wp.array(contact_dir, dtype=wp.vec3),
            hand_slot_starts=i32([0, 2]),
            link_counts=i32([2, 2]),
            layout=SimpleNamespace(slots_per_world=SLOTS),
        ),
        action=SimpleNamespace(
            action_rate_l2=f32(rng.uniform(0.0, 2.0, size=WORLDS)),
            action_l2=f32(rng.uniform(0.0, 2.0, size=WORLDS)),
            sonic_q_ids=i32([7, 8, 9]),
        ),
        wrist_frames=SimpleNamespace(
            body_ids=i32([1, 2]),
            body_to_frame_pos=wp.array(rng.normal(scale=0.05, size=(2, 3)).astype(np.float32), dtype=wp.vec3),
            body_to_frame_quat=wp.array(_random_quats(rng, 2).astype(np.float32), dtype=wp.quat),
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
            wp.zeros(WORLDS * len(module.RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32)
            if config.object_task_enabled
            else None
        ),
    )
    return module.ReconBodyObjective(**kwargs)


def _inputs(seed):
    import warp as wp

    rng = np.random.default_rng(seed)
    positions = rng.normal(scale=0.5, size=(WORLDS * BODIES, 3)).astype(np.float32)
    quats = _random_quats(rng, WORLDS * BODIES).astype(np.float32)
    state = SimpleNamespace(
        body_q=wp.array(
            [wp.transform(wp.vec3(*positions[i]), wp.quat(*quats[i])) for i in range(WORLDS * BODIES)],
            dtype=wp.transform,
        ),
        joint_q=wp.array(rng.normal(scale=0.3, size=WORLDS * Q).astype(np.float32), dtype=wp.float32),
    )
    timestep = wp.array(rng.integers(-2, FRAMES + 3, size=WORLDS).astype(np.int32), dtype=wp.int32)
    terminated = wp.array((rng.uniform(size=WORLDS) > 0.6).astype(np.int32), dtype=wp.int32)
    return state, timestep, terminated


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # stock objective (CHORD (PPO))
        # object_task as in DexDuet without interaction rewards
        {"object_task_weight": 5.0, "object_task_position_beta": 10.0, "object_task_rotation_beta": 1.0},
    ],
)
def test_flags_off_objective_is_bit_identical_to_the_frozen_copy(monkeypatch, kwargs):
    import warp as wp

    from flash_chord.objectives import recon_body as current

    frozen = frozen_module()
    new_config = current.ReconBodyObjectiveConfig(**kwargs)
    old_config = frozen.ReconBodyObjectiveConfig(**kwargs)
    assert not new_config.dexduet_enabled
    assert new_config.term_names == old_config.term_names
    assert new_config.term_weights == old_config.term_weights
    launches = []
    original = wp.launch

    def recording(kernel, *args, **kw):
        launches.append(kernel.key)
        return original(kernel, *args, **kw)

    monkeypatch.setattr(wp, "launch", recording)
    with wp.ScopedDevice("cpu"):
        old = _objective(frozen, old_config)
        new = _objective(current, new_config)
        assert new.pre_dexduet_terms is None and new.dexduet is None
        for seed in (1, 2, 3, 4):
            del launches[:]
            old.evaluate(*_inputs(seed))
            old_launches = list(launches)
            del launches[:]
            new.evaluate(*_inputs(seed))
            assert launches == old_launches
            expected = ["batched_wrench_support", "reduce_recon_body_force_closure", "compose_recon_body_objective"]
            if kwargs:
                expected.append("add_recon_body_object_task")
            assert launches == expected
            np.testing.assert_array_equal(new.terms.numpy(), old.terms.numpy())
            np.testing.assert_array_equal(new.score.numpy(), old.score.numpy())
            np.testing.assert_array_equal(new.force_closure.numpy(), old.force_closure.numpy())
            assert np.abs(new.score.numpy()).max() > 0.0
