# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Configs and export of DexDuet's experiment (experiment/g1_recon_body_coord_rsi_r2_dexduet.yaml).

* Existing experiments compose without any DexDuet key and build objectives with both DexDuet terms off.
* DexDuet's experiment differs from g1_recon_body_coord_rsi_r2 (DexDuet without interaction rewards) only by the new
  objective fields, the curriculum keys hand_graph/ho_contact and the logging names.
* A mismatched objective/curriculum pair fails the exact-name validation.
* The export strips every DexDuet key and term; the exported stock config equals that of g1_recon_body_coord_rsi_r2.
"""

from __future__ import annotations

import pytest

from flash_chord.configuration import compose_config, instantiate_typed, resolved_dict

PARQUET = "task.parquet=/tmp/reference.parquet"
WITHOUT_IR = "g1_recon_body_coord_rsi_r2"
DEXDUET = "g1_recon_body_coord_rsi_r2_dexduet"
EXISTING = (
    "g1_recon_body_ppo",
    "g1_recon_body_rsi",
    "g1_recon_body_coord",
    "g1_recon_body_coord_rsi",
    "g1_recon_body_coord_rsi_wide",
    "g1_recon_body_coord_rsi_r2",
    "g1_recon_body_flash_sac",
)
DEXDUET_FIELDS = {
    "hand_graph_weight": 2.5,
    "hand_graph_beta_dir": 5.0,
    "hand_graph_beta_gap": 20.0,
    "hand_graph_thumb_weight": 0.5,
    "hand_graph_gap_scale": 1.5,
    "hand_graph_tip_offset": 0.045,
    "ho_contact_weight": 5.0,
    "ho_contact_cap": 0.05,
    "ho_contact_tol": 0.01,
    "ho_contact_scale": 0.02,
    "ho_contact_touch": 0.01,
    "ho_contact_near": 0.5,
    "dexduet_targets_dir": "",
}


def _compose(experiment, *overrides, config_name="train"):
    return resolved_dict(compose_config(config_name, [f"experiment={experiment}", PARQUET, *overrides]))


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out.update(_flatten(child, f"{prefix}{key}."))
        return out
    if isinstance(value, list) and any(isinstance(child, dict) for child in value):
        out = {}
        for index, child in enumerate(value):
            out.update(_flatten(child, f"{prefix}{index}."))
        return out
    return {prefix.rstrip("."): value}


@pytest.mark.parametrize("experiment", EXISTING)
def test_existing_experiments_have_no_dexduet_keys(experiment):
    from flash_chord.envs.rl import RLEnvConfig

    config_name = "train_flash_sac" if "flash_sac" in experiment else "train"
    composed = compose_config(config_name, [f"experiment={experiment}", PARQUET])
    flat = _flatten(resolved_dict(composed))
    assert not [key for key in flat if any(tag in key for tag in ("hand_graph", "ho_contact", "dexduet"))]
    env = instantiate_typed(composed.env, RLEnvConfig)
    assert not env.objective.dexduet_enabled
    assert "hand_graph" not in env.objective.term_names and "ho_contact" not in env.objective.term_names


def test_dexduet_differs_from_without_ir_only_by_the_dexduet_keys():
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.ppo.config import TrainingConfig

    base = _flatten(_compose(WITHOUT_IR))
    dexduet_config = compose_config("train", [f"experiment={DEXDUET}", PARQUET])
    dexduet = _flatten(resolved_dict(dexduet_config))
    differing = {
        key for key in base.keys() | dexduet.keys() if base.get(key, "<absent>") != dexduet.get(key, "<absent>")
    }
    expected = {
        **{f"{root}.{key}": value for root in ("objective", "env.objective") for key, value in DEXDUET_FIELDS.items()},
        **{
            f"{root}.stages.0.objective_weights.{name}": weight
            for root in ("curriculum", "training.curriculum")
            for name, weight in (("hand_graph", 2.5), ("ho_contact", 5.0))
        },
        # logging names (training.* interpolates them)
        "logging.experiment": "g1_sonic_recon_body_coord_rsi_r2_dexduet",
        "logging.run_name": "g1_recon_body_coord_rsi_r2_dexduet",
        "training.experiment_name": "g1_sonic_recon_body_coord_rsi_r2_dexduet",
        "training.run_name": "g1_recon_body_coord_rsi_r2_dexduet",
    }
    assert differing == set(expected)
    assert {key: dexduet[key] for key in expected} == expected
    assert all(key not in base for key in expected if "name" not in key and not key.startswith("logging."))
    # the object task of g1_recon_body_coord_rsi_r2 (weight 5, DexDuet betas 10/1) is untouched
    assert (dexduet["objective.object_task_weight"], dexduet["objective.object_task_position_beta"]) == (5.0, 10.0)
    assert dexduet["objective.object_task_rotation_beta"] == 1.0

    env = instantiate_typed(dexduet_config.env, RLEnvConfig)
    assert env.objective.term_names[-3:] == ("object_task", "hand_graph", "ho_contact")
    assert env.objective.term_weights[-3:] == (5.0, 2.5, 5.0)
    training = instantiate_typed(dexduet_config.training, TrainingConfig)
    for stage in training.curriculum.stages:
        assert stage.weights_for(env.objective.term_names)[-3:] == (5.0, 2.5, 5.0)
    assert training.max_iterations == 45000  # the comparison caps it at 20000 on the command line


def test_command_line_budget_and_targets_override():
    dexduet = _flatten(_compose(DEXDUET, "training.max_iterations=20000", "objective.dexduet_targets_dir=/x/targets"))
    assert dexduet["training.max_iterations"] == 20000
    assert dexduet["objective.dexduet_targets_dir"] == "/x/targets"
    assert dexduet["env.objective.dexduet_targets_dir"] == "/x/targets"


def test_mismatched_dexduet_objective_and_curriculum_fail_name_validation():
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.ppo.config import TrainingConfig

    only_objective = compose_config(
        "train", [f"experiment={WITHOUT_IR}", PARQUET, "objective=recon_body_objtask_dexduet"]
    )
    env = instantiate_typed(only_objective.env, RLEnvConfig)
    stage = instantiate_typed(only_objective.training, TrainingConfig).curriculum.stages[0]
    with pytest.raises(ValueError, match=r"missing=\['hand_graph', 'ho_contact'\]"):
        stage.weights_for(env.objective.term_names)
    only_curriculum = compose_config(
        "train", [f"experiment={WITHOUT_IR}", PARQUET, "curriculum=recon_body_rsi_objtask_dexduet"]
    )
    env = instantiate_typed(only_curriculum.env, RLEnvConfig)
    stage = instantiate_typed(only_curriculum.training, TrainingConfig).curriculum.stages[0]
    with pytest.raises(ValueError, match=r"unexpected=\['hand_graph', 'ho_contact'\]"):
        stage.weights_for(env.objective.term_names)


def test_export_strips_every_dexduet_key_and_equals_the_without_ir_export():
    import flash_chord.training.ppo.coordinate.export as exporter
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES

    exported, notes = exporter.stock_config_with_notes(_compose(DEXDUET, "objective.dexduet_targets_dir=/x/targets"))
    without_ir_exported, without_ir_notes = exporter.stock_config_with_notes(_compose(WITHOUT_IR))
    flat = _flatten(exported)
    assert not [key for key in flat if any(tag in key for tag in ("hand_graph", "ho_contact", "dexduet", "object_task"))]
    for curriculum in (exported["curriculum"], exported["training"]["curriculum"]):
        for stage in curriculum["stages"]:
            assert tuple(stage["objective_weights"]) == RECON_BODY_OBJECTIVE_TERM_NAMES
    removed = set(notes["removed_keys"]) - set(without_ir_notes["removed_keys"])
    assert removed == {
        f"{root}.{key}" for root in ("objective", "env.objective") for key in DEXDUET_FIELDS
    } | {
        f"{root}.stages.0.objective_weights.{name}"
        for root in ("curriculum", "training.curriculum")
        for name in ("hand_graph", "ho_contact")
    }
    # apart from the logging names (kept by export), the exported stock config is that of g1_recon_body_coord_rsi_r2
    without_ir_flat = _flatten(without_ir_exported)
    differing = {key for key in flat.keys() | without_ir_flat.keys() if flat.get(key) != without_ir_flat.get(key)}
    assert differing == {"logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name"}
