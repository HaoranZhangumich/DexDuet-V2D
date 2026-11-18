#!/usr/bin/env bash
# Contact-AUC of one evaluated policy on the V2D challenge, with the paper's surface rule.
#
#   bash reproduce/contact_auc.sh <method> <episode> [gpu]
#
# Run evaluate.sh first: this script scores the same file the official trials evaluated (the exported policy), read
# from $RESULTS/<method>/official/epNN.json.
#
# Steps:
#   1. record:  TRIALS=4 further official evaluations, each through scripts/dexduet/contact_auc/record_eval.py, which
#               runs the kit's unchanged evaluate_policy.py and copies the simulated body poses after every step
#               (recording.npz). A trial is tried at most 3 times.
#   2. human:   the human hand side of the episode (reproduce/tools/human_soma.py, $PY_V2D, CPU): the 2,847 SOMA hand
#               vertices per hand in the evaluator's world. Computed once per episode into $RESULTS/human/
#               (or set CAUC_HUMAN to a folder that already holds epNN.npz / epNN.json).
#   3. score:   scripts/dexduet/contact_auc/contact_auc_surface.py (CPU, a few minutes): 64 surface points on each of
#               the 8 collision links of each Dex3 hand; a hand touches the object when a point is within 5 mm of it.
#
# Output: $RESULTS/<method>/contact/score/epNN.json; the score is results["<method>:all"]["slip_auc"] (0-1).
# Episode 29 has no frame in which the human holds the object, so it has no Contact-AUC (the paper averages 29).
# PY_CAUC: Python for step 3 (default $PY); it needs numpy, scipy and trimesh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

METHOD=${1:?usage: contact_auc.sh <method> <episode> [gpu]}
EP=$(ep2 "${2:?usage: contact_auc.sh <method> <episode> [gpu]}")
GPU=${3:-0}
check_method "$METHOD"
need_data
need_kit
need_official
EPN=$((10#$EP))
PY_CAUC=${PY_CAUC:-$PY}
CROOT=$RESULTS/$METHOD/contact
SCORE=$CROOT/score/ep$EP.json
mkdir -p "$CROOT/logs"
LOG=$CROOT/logs/ep$EP.log
if [ "$EPN" = 29 ]; then say "$METHOD ep29: no human hold in this episode, so no Contact-AUC"; exit 0; fi
if [ -f "$SCORE" ]; then say "$METHOD ep$EP: already scored ($SCORE)"; exit 0; fi

OFFICIAL=$RESULTS/$METHOD/official/ep$EP.json
[ -f "$OFFICIAL" ] || die "run evaluate.sh $METHOD $EPN first ($OFFICIAL missing)"
read -r FILE SHA NOK EXPORT_FAILED < <("$PY" -B -c '
import json, sys
d = json.load(open(sys.argv[1]))
failed = any("export failed" in n for n in d.get("notes", []))
print(d.get("evaluated_file") or "-", d.get("evaluated_sha256") or "-", d.get("n_evaluated", 0), int(failed))' "$OFFICIAL")
if [ "$EXPORT_FAILED" = 1 ]; then
  say "$METHOD ep$EP: the export to the official policy format failed, so no Contact-AUC"; exit 0
fi
[ "$NOK" -gt 0 ] || die "$METHOD ep$EP: $OFFICIAL has no evaluated trial; run evaluate.sh $METHOD $EPN again"
[ "$FILE" != - ] || die "$METHOD ep$EP: $OFFICIAL names no evaluated policy file"

# 1. record
for k in $(seq 1 "$TRIALS"); do
  base=$CROOT/record/r$k/ep$EP
  mkdir -p "$base"
  for n in 1 2 3; do
    if grep -qs '"status": "ok"' "$base"/try*/record.json; then break; fi
    [ -e "$base/try$n" ] && continue
    say "record r$k try$n on gpu $GPU" | tee -a "$LOG"
    (cd "$FC_OFFICIAL" && env -u PYTHONHOME -u PYTHONSTARTUP PYTHONPATH="$FC_OFFICIAL/src" FC_OFFICIAL="$FC_OFFICIAL" \
       V2D_KIT="$(cd "$V2D_KIT" && pwd)" OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$GPU" \
       "$PY" -B "$REPO/scripts/dexduet/contact_auc/record_eval.py" --checkpoint "$FILE" --sha256 "$SHA" \
       --episode "$EPN" --out "$base/try$n") >> "$LOG" 2>&1 || say "record r$k try$n failed (log $LOG)"
  done
  grep -qs '"status": "ok"' "$base"/try*/record.json || die "record r$k failed 3 times (log $LOG)"
done

# 2. human hand side
HUMAN=${CAUC_HUMAN:-$RESULTS/human}
if [ ! -f "$HUMAN/ep$EP.json" ]; then
  need PY_V2D V2D_TRACK2
  RG=${V2D_RG:-$FC_OFFICIAL/..}
  first=$(ls -d "$CROOT"/record/r1/ep$EP/try* | while read -r d; do grep -qs '"status": "ok"' "$d/record.json" && echo "$d"; done | head -n 1)
  say "human hand side of ep$EP -> $HUMAN" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES= "$PY_V2D" -B "$REPRO/tools/human_soma.py" "$EPN" --data "$V2D_DATA" \
    --tier1 "$V2D_TRACK2/tier_1_multiview_caption" --robotic-grounding "$RG/source/robotic_grounding" \
    --recording "$first/recording.npz" --out "$HUMAN" >> "$LOG" 2>&1 || die "human side failed (log $LOG)"
fi

# 3. score
say "score $METHOD ep$EP (CPU)" | tee -a "$LOG"
(cd "$REPO/scripts/dexduet/contact_auc" && env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH PYTHONNOUSERSITE=1 \
   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   DD_CAUC_OUT="$CROOT" DD_CAUC_RUNS="{\"$METHOD\": \"$CROOT/record\"}" DD_CAUC_ARMS="$METHOD" \
   DD_CAUC_TRIALS="$(seq -s, 1 "$TRIALS")" DD_CAUC_HUMAN="$HUMAN" \
   HUMAN_MOTION_DATA_DIR="$V2D_DATA" FC_OFFICIAL="$FC_OFFICIAL" \
   "$PY_CAUC" -B contact_auc_surface.py "$EPN" --touch 0.005 --n 64 --out "$CROOT") >> "$LOG" 2>&1 \
  || die "scoring failed (log $LOG)"
"$PY" -B -c '
import json, sys
d = json.load(open(sys.argv[1]))
r = d["results"][sys.argv[2] + ":all"]
print("%s ep%02d: Contact-AUC %.1f (coverage %.1f)" % (sys.argv[2], d["episode"], 100 * r["slip_auc"], 100 * r["coverage"]))
' "$SCORE" "$METHOD"
