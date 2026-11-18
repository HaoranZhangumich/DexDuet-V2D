# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""DexDuet's environment (g1_recon_body_coord_rsi_r2_dexduet) on a real Tier-1 episode, eager vs CUDA-graph-captured
step (GPU, sequence data).

Needs ``HUMAN_MOTION_DATA_DIR`` (Tier-1 data) and ``DEXDUET_TARGETS_DIR`` (scripts/dexduet sidecars, TARGETS.sha256);
skipped without them (the data comes from the environment, not the in-tree ``sequence_data`` corpus).
"""

import os
from pathlib import Path

import numpy as np
import pytest

SEQUENCE = "2026-03-12_15-06-36_spray_bottle_left_desk_ground_02"


def _paths():
    data = os.environ.get("HUMAN_MOTION_DATA_DIR")
    targets = os.environ.get("DEXDUET_TARGETS_DIR")
    if not data or not targets:
        pytest.skip("needs HUMAN_MOTION_DATA_DIR and DEXDUET_TARGETS_DIR")
    parquet = Path(data) / "whole_body" / "soma" / f"sequence_id={SEQUENCE}" / "robot_name=g1"
    if not parquet.is_dir() or not (Path(targets) / SEQUENCE).is_dir():
        pytest.skip("episode data or sidecars missing")
    return parquet, targets


@pytest.mark.gpu
@pytest.mark.slow
def test_dexduet_env_eager_and_captured_steps_match():
    import warp as wp

    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.objectives.recon_body import ReconBodyObjective
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    parquet, targets = _paths()
    worlds = 8
    config = compose_config(
        "train",
        [
            "experiment=g1_recon_body_coord_rsi_r2_dexduet",
            f"task.parquet='{parquet}'",
            f"scene.world_count={worlds}",
            "logging.mode=disabled",
            f"objective.dexduet_targets_dir={targets}",
        ],
    )
    environment_config = instantiate_typed(config.env, RLEnvConfig)
    embodiment = instantiate_typed(config.embodiment, Embodiment)
    collision = instantiate_typed(config.collision, CollisionPolicy)
    with wp.ScopedDevice("cuda:0"):
        setup = setup_scene(
            parquet=config.task.parquet,
            control_fps=environment_config.sim.fps,
            motion_speed=config.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=worlds,
            include_support=config.scene.support,
            decompose_objects=config.scene.decompose_objects,
            contact_friction=config.scene.contact_friction,
            object_free_joint_damping=config.scene.object_free_joint_damping,
            source_frame_playback=bool(config.task.source_frame_playback),
            motion_start_frame=int(config.task.motion_start_frame),
            motion_end_frame=int(config.task.motion_end_frame),
        )
        environment = RLEnv(setup.scene, setup.reference, config=environment_config)
        objective = environment.objective
        assert isinstance(objective, ReconBodyObjective)
        assert objective.term_names[-3:] == ("object_task", "hand_graph", "ho_contact")
        assert objective.dexduet is not None and objective.pre_dexduet_terms is not None
        assert objective.dexduet.sha256  # sidecars checked against TARGETS.sha256
        names = objective.term_names
        environment.set_voc_scale(0.0)

        action = wp.zeros(environment.action.action_dim * environment.world_count, dtype=wp.float32)
        weights = np.asarray(objective.weights.numpy(), dtype=np.float64)
        # GPU physics is not bitwise reproducible run to run (contact atomics), so two rollouts cannot be compared
        # bitwise. Instead every captured transition's objective (stock + object_task + DexDuet kernels inside the
        # CUDA graph) is compared with an eager evaluation of the same objective on the same post-step state.
        environment.reset(frame_id=0)
        environment.capture_step()
        environment.reset(frame_id=0)
        for _ in range(4):
            observation, reward, terminated, truncated = environment.step(action)
            assert not terminated.numpy().any() and not truncated.numpy().any()
            graph_terms = objective.terms.numpy().reshape(worlds, len(names)).copy()
            graph_score = objective.score.numpy().copy()
            previous = wp.array(environment.timestep.numpy() - 1, dtype=wp.int32)
            objective.evaluate(environment.state_0, previous, environment.terminated)
            eager_terms = objective.terms.numpy().reshape(worlds, len(names))
            np.testing.assert_array_equal(graph_terms, eager_terms)
            np.testing.assert_array_equal(graph_score, objective.score.numpy())
            assert np.isfinite(graph_terms).all() and np.isfinite(observation.numpy()).all()
            interaction = graph_terms[:, -2:]
            assert np.all(interaction > 0.0) and np.all(interaction <= 1.0)
            # score = frame_dt * sum(weight * term) over all 15 columns (hand_graph 2.5, ho_contact 5.0 last)
            expected = environment.frame_dt * (graph_terms.astype(np.float64) * weights).sum(axis=1)
            np.testing.assert_allclose(graph_score, expected, rtol=1.0e-5, atol=1.0e-6)
        assert weights[-2:].tolist() == [2.5, 5.0]
