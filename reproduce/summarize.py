#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Results table of the V2D comparison: per-method means and paired differences against DexDuet.

    python reproduce/summarize.py [--results DIR] [--out DIR] [--methods dexduet,chord_ppo,...]

Reads, per method and episode (written by evaluate.sh and contact_auc.sh):
    <results>/<method>/official/epNN.json             the official trials and their mean ("trials", "mean": add_auc,
                                                      spider_sr_uncentered, ...)
    <results>/<method>/contact/score/epNN.json        Contact-AUC (surface rule), also per recording ("lanes")
and writes <out>/summary.md and <out>/summary.json (default out = results).

Rules (as in the paper):
  * An episode value is the mean over its official trials. A method's value is the mean over its evaluated episodes;
    an episode without an evaluation (for example a policy that outputs no finite action, so its export fails) is left
    out and n says how many remain.
  * Contact-AUC is averaged over the episodes in which the human holds the object (29; episode 29 has no hold).
  * The +- after a method's value is the spread over the evaluator runs: the mean over episodes of each run (the k-th
    official trial of every episode; for Contact-AUC the k-th recording), then the sample SD of these run means. It is
    left out when the episodes do not all have the same number of runs.
  * Paired differences (DexDuet minus each other method) use the episodes both methods have, with a 95 % percentile
    bootstrap CI over episodes (10,000 resamples, numpy default_rng(0) for every comparison and metric), win/tie/loss
    counts (tie: |difference| < 0.5 points; MPPE: equal) and an exact two-sided sign test on wins against losses.
Only numpy is needed.
"""
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

NAMES = {
    "dexduet": "DexDuet",
    "dexduet_no_interaction_rewards": "DexDuet without interaction rewards",
    "chord_ppo": "CHORD (PPO)",
}
ORDER = ("chord_ppo", "dexduet_no_interaction_rewards", "dexduet")
# (key, label, scale, decimals, direction): direction +1 = higher is better. The kit's keys: add_auc = ADD-AUC,
# spider_sr_uncentered = SP-SR (the leaderboard's key), maniptrans_sr = MP-SR, mppe_cm = MPPE.
METRICS = (
    ("add_auc", "ADD-AUC", 100, 1, 1),
    ("spider_sr_uncentered", "SP-SR", 100, 1, 1),
    ("maniptrans_sr", "MP-SR", 100, 1, 1),
    ("mppe_cm", "MPPE (cm)", 1, 2, -1),
    ("contact_auc", "Contact-AUC", 100, 1, 1),
)
TIE = {"add_auc": 0.005, "spider_sr_uncentered": 0.005, "maniptrans_sr": 0.005, "contact_auc": 0.005, "mppe_cm": 0.0}
N_BOOT = 10000


def contact_value(path):
    """Contact-AUC of one episode and its values per recording (None when the file has none)."""
    d = json.loads(path.read_text())
    res = d["results"]
    methods = sorted({k.rsplit(":", 1)[0] for k in res})
    if len(methods) != 1:
        raise SystemExit(f"{path}: expected one method in results, found {methods}")
    r = res[f"{methods[0]}:all"]
    lanes = (r.get("lanes") or {}).get("values")
    return r["slip_auc"], None if not lanes or None in lanes else [float(x) for x in lanes]


def load(results, method):
    """{episode: {metric: episode value, "runs": {metric: [value of each evaluator run]}}}"""
    vals = {}
    for ep in range(30):
        row = {"runs": {}}
        f = results / method / "official" / f"ep{ep:02d}.json"
        if f.is_file():
            d = json.loads(f.read_text())
            if d.get("n_evaluated", 0) > 0:
                row.update({k: float(d["mean"][k]) for k, *_ in METRICS[:4] if d["mean"].get(k) is not None})
                trials = [t for t in d.get("trials", []) if t.get("status") == "evaluated"]
                for k, *_ in METRICS[:4]:
                    if k in row and trials and all(t.get(k) is not None for t in trials):
                        row["runs"][k] = [float(t[k]) for t in trials]
        c = results / method / "contact" / "score" / f"ep{ep:02d}.json"
        if c.is_file():
            value, lanes = contact_value(c)
            row["contact_auc"] = float(value)
            if lanes:
                row["runs"]["contact_auc"] = lanes
        vals[ep] = row
    return vals


def run_spread(vals, key):
    """(sample SD of the run means, run means) over the episodes that have `key`; (None, None) unless every one of
    them has the same number (at least 2) of evaluator runs."""
    runs = [vals[e]["runs"].get(key) for e in range(30) if key in vals[e]]
    if not runs or any(r is None for r in runs) or len({len(r) for r in runs}) != 1 or len(runs[0]) < 2:
        return None, None
    means = np.asarray(runs, dtype=np.float64).mean(axis=0)
    return float(np.std(means, ddof=1)), [float(x) for x in means]


def bootstrap_ci(diffs):
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def sign_test_p(win, loss):
    m = win + loss
    if m == 0:
        return 1.0
    tail = Fraction(sum(math.comb(m, i) for i in range(min(win, loss) + 1)), 2 ** m)
    return float(min(Fraction(1), 2 * tail))


def paired(x, y, key, direction):
    eps = [e for e in range(30) if key in x[e] and key in y[e]]
    if not eps:
        return None
    d = [x[e][key] - y[e][key] for e in eps]
    win = tie = loss = 0
    for v in d:
        if abs(v) < TIE[key] or v == 0:
            tie += 1
        elif direction * v > 0:
            win += 1
        else:
            loss += 1
    return {"n": len(eps), "episodes": eps, "mean_x": float(np.mean([x[e][key] for e in eps])),
            "mean_y": float(np.mean([y[e][key] for e in eps])), "mean_diff": float(np.mean(d)),
            "ci95": bootstrap_ci(d), "win": win, "tie": tie, "loss": loss, "sign_p": sign_test_p(win, loss)}


def fmt(v, scale, nd, signed=False):
    return "n/a" if v is None else format(v * scale, f"{'+' if signed else ''}.{nd}f")


def fmt_sd(v, sd, scale, nd):
    return fmt(v, scale, nd) + ("" if v is None or sd is None else f" ± {fmt(sd, scale, nd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=None, help="default: $RESULTS or ./runs/results")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--methods", default=",".join(ORDER))
    a = ap.parse_args()
    import os
    results = a.results or Path(os.environ.get("RESULTS") or Path(os.environ.get("RUNS", "runs")) / "results")
    out = a.out or results
    methods = [m for m in a.methods.split(",") if m]
    for m in methods:
        if m not in NAMES:
            sys.exit(f"unknown method {m}")
    data = {m: load(results, m) for m in methods}
    lines = ["# V2D challenge, Tier 1 (30 episodes, G1 + Dex3, official evaluator)", "",
             f"Results folder: `{results}`. ADD-AUC, SP-SR, MP-SR and Contact-AUC in %, MPPE in cm. Each method: mean "
             "over its evaluated episodes (n) ± the sample SD over the evaluator runs of that mean (run k: the k-th "
             "official trial of every episode; for Contact-AUC the k-th recording).", "",
             "| Method | n | ADD-AUC | SP-SR | MP-SR | MPPE (cm) | Contact-AUC (n) |", "|---|---|---|---|---|---|---|"]
    summary = {"results": str(results), "methods": {}, "paired_vs_dexduet": {}}
    for m in methods:
        v = data[m]
        means, ns, sds, run_means = {}, {}, {}, {}
        for key, *_ in METRICS:
            xs = [v[e][key] for e in range(30) if key in v[e]]
            means[key], ns[key] = (float(np.mean(xs)) if xs else None), len(xs)
            sds[key], run_means[key] = run_spread(v, key)
        missing = [e for e in range(30) if "add_auc" not in v[e]]
        summary["methods"][m] = {"name": NAMES[m], "means": means, "n": ns, "sd_over_runs": sds, "run_means": run_means,
                                 "episodes_without_evaluation": missing}
        cells = [fmt_sd(means[k], sds[k], s, nd) for k, _, s, nd, _ in METRICS[:4]]
        lines.append(f"| {NAMES[m]} | {ns['add_auc']} | " + " | ".join(cells)
                     + f" | {fmt_sd(means['contact_auc'], sds['contact_auc'], 100, 1)} ({ns['contact_auc']}) |")
    n_table = len(lines)
    for m in methods:
        miss = summary["methods"][m]["episodes_without_evaluation"]
        if miss and len(miss) < 30:
            lines.append(f"\n{NAMES[m]}: no official evaluation for episode(s) {', '.join(f'{e:02d}' for e in miss)}.")
    if "dexduet" in methods:
        lines += ["", "## DexDuet minus each method (paired episodes)", "",
                  "Mean difference [95 % bootstrap CI]; W/T/L from DexDuet's side; sign-test p.", "",
                  "| Compared with | Metric | n | Difference [95 % CI] | W/T/L | p |", "|---|---|---|---|---|---|"]
        for m in methods:
            if m == "dexduet":
                continue
            summary["paired_vs_dexduet"][m] = {}
            for key, label, scale, nd, direction in METRICS:
                st = paired(data["dexduet"], data[m], key, direction)
                summary["paired_vs_dexduet"][m][key] = st
                if st is None:
                    continue
                lo, hi = st["ci95"]
                lines.append(f"| {NAMES[m]} | {label} | {st['n']} | {fmt(st['mean_diff'], scale, nd, True)} "
                             f"[{fmt(lo, scale, nd, True)}, {fmt(hi, scale, nd, True)}] | "
                             f"{st['win']}/{st['tie']}/{st['loss']} | {st['sign_p']:.2g} |")
    lines += ["", "## Per episode: ADD-AUC / Contact-AUC (%)", "",
              "| ep | " + " | ".join(NAMES[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for e in range(30):
        cells = []
        for m in methods:
            r = data[m][e]
            cells.append(f"{fmt(r.get('add_auc'), 100, 1)} / {fmt(r.get('contact_auc'), 100, 1)}")
        lines.append(f"| {e:02d} | " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines[:n_table]))
    print(f"\nwrote {out / 'summary.md'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
