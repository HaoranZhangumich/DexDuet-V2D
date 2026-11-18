#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers for reproduce/evaluate.sh and reproduce/contact_auc.sh (standard library only).

    ckpt.py select <run dir>                   print "<file> <iteration>" of the checkpoint to evaluate
    ckpt.py parquet <checkpoint>               print the task parquet path stored in the checkpoint
    ckpt.py relocate <in> <out> <parquet>      copy <in> to <out> with the stored task parquet set to <parquet>
    ckpt.py summary <method> <episode> <trials root> <trials> <checkpoint> <evaluated file> [--note TEXT]...
                                               write the per-episode mean of the official trials (JSON on stdout)
    ckpt.py show <epNN.json>                   print one line from such a file

Which checkpoint is evaluated (the paper's rule, no best-checkpoint selection):
  * a RUN_INFO.json with a "checkpoint" entry names it (released checkpoints);
  * else final.json names it (DexDuet runs: model_20000, or the iteration at which the run stopped itself);
  * else model_20000.safetensors (CHORD PPO);
  * else the only model_<N> file in the folder.

The evaluator reads the task parquet path stored in the checkpoint. Checkpoints trained elsewhere store that machine's
path, so evaluate.sh evaluates a copy whose stored path points at the local parquet ("relocate"). Only the JSON
metadata changes; the tensor bytes are copied unchanged.
"""
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

CONFIG_KEY = "user.resolved_config_json"  # FlashCHORD checkpoint metadata: the resolved training config
FINAL_FILE = "model_20000.safetensors"  # the checkpoint at the end of the budget (CHORD PPO)
# The evaluator's keys for ADD-AUC, SP-SR, MP-SR and MPPE (cm). SP-SR is spider_sr_uncentered, the key the challenge's
# leaderboard reads; the evaluator's chord_sr is CHORD's own success rate and is not SP-SR.
METRICS = ("add_auc", "spider_sr_uncentered", "maniptrans_sr", "mppe_cm")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, header


def number(name):
    m = re.match(r"^model_(\d+)\.safetensors$", name)
    return int(m.group(1)) if m else None


def select(run_dir):
    d = Path(run_dir)
    if not d.is_dir():
        sys.exit(f"no checkpoint folder {d}")
    for marker in ("RUN_INFO.json", "final.json"):
        if (d / marker).is_file():
            name = json.loads((d / marker).read_text()).get("checkpoint")
            if isinstance(name, dict):  # released RUN_INFO.json: {"file": ..., "sha256": ...}
                name = name.get("file")
            if name:
                name = Path(name).name
                if not (d / name).is_file():
                    sys.exit(f"{d / marker} names {name}, which is not in {d}")
                return d / name, number(name)
    if (d / FINAL_FILE).is_file():
        return d / FINAL_FILE, number(FINAL_FILE)
    found = sorted(p for p in d.glob("model_*.safetensors") if number(p.name) is not None)
    if len(found) == 1:
        return found[0], number(found[0].name)
    sys.exit(f"cannot tell which checkpoint to evaluate in {d}: {[p.name for p in found] or 'none'}")


def stored_parquet(path):
    _, header = read_header(path)
    config = json.loads(header["__metadata__"][CONFIG_KEY])
    return config["task"]["parquet"]



def relocate(src, dst, parquet):
    n, header = read_header(src)
    config = json.loads(header["__metadata__"][CONFIG_KEY])
    config["task"]["parquet"] = str(parquet)
    header["__metadata__"][CONFIG_KEY] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        fin.seek(8 + n)
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for block in iter(lambda: fin.read(1 << 20), b""):
            fout.write(block)
    tmp.replace(dst)


def summary(method, episode, root, trials, checkpoint, evaluated, notes):
    ep = int(episode)
    rows = []
    for t in range(1, int(trials) + 1):
        path = Path(root) / f"t{t:02d}" / "results" / f"episode_{ep:06d}.json"
        if not path.is_file():
            rows.append({"trial": t, "status": "missing"})
            continue
        r = json.loads(path.read_text())
        m = r.get("metrics", {})
        rows.append({"trial": t, "status": "evaluated", **{k: m.get(k) for k in METRICS},
                     "checkpoint_sha256": r.get("checkpoint_sha256"), "result": str(path)})
    ok = [r for r in rows if r["status"] == "evaluated"]
    mean = {k: (sum(float(r[k]) for r in ok) / len(ok)) if ok else None for k in METRICS}
    out = {"method": method, "episode": ep, "trials": rows, "n_evaluated": len(ok), "mean": mean,
           "checkpoint": {"file": str(checkpoint), "sha256": sha256(checkpoint) if Path(checkpoint).is_file() else None},
           "evaluated_file": str(evaluated) if evaluated != "-" else None,
           "evaluated_sha256": sha256(evaluated) if evaluated != "-" and Path(evaluated).is_file() else None,
           "notes": notes}
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "select":
        path, n = select(argv[2])
        print(path, n if n is not None else "-")
    elif cmd == "parquet":
        print(stored_parquet(argv[2]))
    elif cmd == "relocate":
        relocate(argv[2], argv[3], argv[4])
    elif cmd == "show":
        d = json.loads(Path(argv[2]).read_text())
        m = d["mean"]
        f = lambda v, nd: "n/a" if v is None else f"{v:.{nd}f}"  # noqa: E731
        print(f"{d['method']} ep{d['episode']:02d}: {d['n_evaluated']} trials, ADD-AUC {f(m['add_auc'], 3)}, "
              f"SP-SR {f(m['spider_sr_uncentered'], 3)}, MP-SR {f(m['maniptrans_sr'], 3)}, MPPE {f(m['mppe_cm'], 2)} cm"
              + "".join(f"; {n}" for n in d.get("notes", [])))
    elif cmd == "summary":
        notes, rest = [], argv[2:]
        while "--note" in rest:
            i = rest.index("--note")
            notes.append(rest[i + 1])
            del rest[i:i + 2]
        summary(*rest, notes)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
