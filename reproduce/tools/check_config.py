#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Check a composed training config (train script ``--cfg job --resolve``) against the paper's recipe and budget.

Exit 0 and print one line when it matches; exit 1 with the differences otherwise. Used by reproduce/train.sh.
"""
import argparse
import sys

import yaml

BUDGET_ITERS = 20000
G1_DEX3 = "flash_chord.embodiments.g1_dex3.G1Dex3"


def get(cfg, dotted):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--method", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--targets", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    want = {
        "embodiment._target_": G1_DEX3,
        "scene.world_count": 4096,
        "reset.seed": 42,
        "training.seed": 42,
        "task.parquet": a.parquet,
        "training.max_iterations": BUDGET_ITERS,
        "training.rollout_steps": 24,
    }
    if a.method == "dexduet":
        want.update({
            "objective.hand_graph_weight": 2.5,
            "objective.ho_contact_weight": 5.0,
            "objective.dexduet_targets_dir": a.targets,
        })
    elif a.method == "dexduet_no_interaction_rewards":
        want.update({"objective.hand_graph_weight": 0.0, "objective.ho_contact_weight": 0.0})
    bad = {k: (get(cfg, k), v) for k, v in want.items() if get(cfg, k) != v}
    if a.method == "dexduet_no_interaction_rewards":  # the two reward weights are absent (default 0) or 0
        for k in ("objective.hand_graph_weight", "objective.ho_contact_weight"):
            if get(cfg, k) in (None, 0, 0.0):
                bad.pop(k, None)
    if bad:
        for k, (got, exp) in bad.items():
            print(f"MISMATCH {k}: got {got!r}, want {exp!r}")
        sys.exit(1)
    budget = f"{BUDGET_ITERS} x 4096 x 24 = {BUDGET_ITERS * 4096 * 24:,} environment steps"
    print(f"config ok: {a.method}, {budget}, seed 42, G1Dex3")


if __name__ == "__main__":
    main()
