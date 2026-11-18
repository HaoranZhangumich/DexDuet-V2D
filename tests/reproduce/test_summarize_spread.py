# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""reproduce/summarize.py: method means with the spread over evaluator runs, on a small results folder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SUMMARIZE = Path(__file__).resolve().parents[2] / "reproduce" / "summarize.py"
KEYS = ("add_auc", "spider_sr_uncentered", "maniptrans_sr", "mppe_cm")


def _summarize():
    spec = importlib.util.spec_from_file_location("v2d_summarize", SUMMARIZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_method(root: Path, method: str, runs: np.ndarray, lanes: np.ndarray, skip=()):
    """runs: (30, trials) ADD-AUC per trial; lanes: (29, recordings) Contact-AUC per recording (episode 29 has none)."""
    for ep in range(30):
        if ep in skip:
            continue
        trials = [{"trial": k + 1, "status": "evaluated", "add_auc": float(v), "spider_sr_uncentered": 1.0,
                   "maniptrans_sr": 0.0, "mppe_cm": 2.0} for k, v in enumerate(runs[ep])]
        mean = {key: float(np.mean([t[key] for t in trials])) for key in KEYS}
        official = root / method / "official" / f"ep{ep:02d}.json"
        official.parent.mkdir(parents=True, exist_ok=True)
        official.write_text(json.dumps({"method": method, "episode": ep, "trials": trials,
                                        "n_evaluated": len(trials), "mean": mean}))
        if ep < 29:
            score = root / method / "contact" / "score" / f"ep{ep:02d}.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            values = [float(v) for v in lanes[ep]]
            score.write_text(json.dumps({"episode": ep, "results": {f"{method}:all": {
                "slip_auc": float(np.mean(values)), "lanes": {"values": values}}}}))


def test_spread_is_the_sd_of_the_run_means(tmp_path):
    summarize = _summarize()
    rng = np.random.default_rng(1)
    runs, lanes = rng.uniform(size=(30, 4)), rng.uniform(size=(29, 4))
    _write_method(tmp_path, "dexduet", runs, lanes)
    vals = summarize.load(tmp_path, "dexduet")

    sd, means = summarize.run_spread(vals, "add_auc")
    assert means == pytest.approx(runs.mean(axis=0).tolist())
    assert sd == pytest.approx(float(np.std(runs.mean(axis=0), ddof=1)))
    sd, means = summarize.run_spread(vals, "contact_auc")
    assert means == pytest.approx(lanes.mean(axis=0).tolist())
    assert sd == pytest.approx(float(np.std(lanes.mean(axis=0), ddof=1)))
    assert summarize.run_spread(vals, "spider_sr_uncentered")[0] == 0.0


def test_spread_uses_the_evaluated_episodes_and_needs_equal_run_counts(tmp_path):
    summarize = _summarize()
    rng = np.random.default_rng(2)
    runs, lanes = rng.uniform(size=(30, 4)), rng.uniform(size=(29, 4))
    _write_method(tmp_path, "chord_ppo", runs, lanes, skip=(7,))
    vals = summarize.load(tmp_path, "chord_ppo")
    keep = [e for e in range(30) if e != 7]
    sd, _ = summarize.run_spread(vals, "add_auc")
    assert sd == pytest.approx(float(np.std(runs[keep].mean(axis=0), ddof=1)))

    vals[3]["runs"]["add_auc"] = vals[3]["runs"]["add_auc"][:3]  # one episode with three runs: no spread
    assert summarize.run_spread(vals, "add_auc") == (None, None)


def test_summary_files_carry_the_spread(tmp_path, monkeypatch):
    summarize = _summarize()
    rng = np.random.default_rng(3)
    for method in ("chord_ppo", "dexduet"):
        _write_method(tmp_path / "results", method, rng.uniform(size=(30, 4)), rng.uniform(size=(29, 4)))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["summarize.py", "--results", str(tmp_path / "results"), "--out", str(out),
                                      "--methods", "chord_ppo,dexduet"])
    summarize.main()
    summary = json.loads((out / "summary.json").read_text())
    for method in ("chord_ppo", "dexduet"):
        entry = summary["methods"][method]
        assert entry["n"]["add_auc"] == 30 and entry["n"]["contact_auc"] == 29
        assert entry["sd_over_runs"]["add_auc"] == pytest.approx(float(np.std(entry["run_means"]["add_auc"], ddof=1)))
    table = (out / "summary.md").read_text()
    assert " ± " in table and summary["paired_vs_dexduet"]["chord_ppo"]["add_auc"]["n"] == 30
