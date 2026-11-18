#!/usr/bin/env bash
# Evaluate one trained policy with the challenge's official Tier-1 evaluator, four times, as in the paper.
#
#   bash reproduce/evaluate.sh <method> <episode> [gpu] [--from-release <dir>]
#
#   --from-release <dir>  evaluate our released checkpoint, <dir>/v2d_g1_dex3/<method>/epNN/ (download_checkpoints.sh),
#                         instead of your own run in $RUNS/<method>/epNN/.
#
# What runs (one world, start at frame 0, explicit reset, one episode per call; the kit's evaluate_policy.py on NVIDIA's
# official runtime $FC_OFFICIAL): scripts/track2/eval_all.py exports the checkpoint to the official policy format (the
# per-hand policy becomes one MLP of the stock layout; the export is checked to give the same actions within a relative
# tolerance of 1e-4, or 2e-4 when 1e-4 is refused, which is then noted), then runs the kit's
# examples/run_policy_evaluation.py.
# The evaluator is not deterministic, so each policy is evaluated TRIALS=4 times and the episode value is the mean.
# A trial whose evaluator run stops with an error (for example when the simulated world of the evaluator becomes
# non-finite) is started once more, at most two tries per trial; the log records each try.
#
# Output: $RESULTS/<method>/official/epNN.json (per-trial and mean ADD-AUC, SP-SR, MP-SR, MPPE), trial files in
# $RESULTS/<method>/official/t0K/. A trial that already exists for the same file is not run again. When no trial is
# evaluated (for example a failed export), epNN.json is still written and the script exits with an error.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

POS=()
RELEASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from-release) RELEASE=$2; shift 2;;
    *) POS+=("$1"); shift;;
  esac
done
METHOD=${POS[0]:?usage: evaluate.sh <method> <episode> [gpu] [--from-release dir]}
EP=$(ep2 "${POS[1]:?usage: evaluate.sh <method> <episode> [gpu] [--from-release dir]}")
GPU=${POS[2]:-0}
check_method "$METHOD"
need_data
need_kit
need_official
EPN=$((10#$EP))
SEQ=$(episode_field "$EP" sequence_id)
PARQ=$(cd "$(dirname "$(parquet_of "$SEQ")")" && pwd)/data.parquet
if [ -n "$RELEASE" ]; then
  CK_DIR=$RELEASE/v2d_g1_dex3/$METHOD/ep$EP
  [ -d "$CK_DIR" ] || CK_DIR=$RELEASE/$METHOD/ep$EP
else
  CK_DIR=$RUNS/$METHOD/ep$EP
fi
EV=$RESULTS/$METHOD/official
STAGE=$EV/runs/ep$EP
mkdir -p "$EV/logs" "$STAGE"
LOG=$EV/logs/ep$EP.log
read -r CK N < <("$PY" -B "$REPRO/tools/ckpt.py" select "$CK_DIR") || die "no checkpoint for $METHOD ep$EP in $CK_DIR"
say "$METHOD ep$EP: checkpoint $CK" | tee -a "$LOG"

# The evaluator loads the task parquet stored in the checkpoint; point it at this machine's copy when it differs.
NOTES=()
STAGED=$STAGE/ep$EP/$(basename "$CK")
mkdir -p "$STAGE/ep$EP"
stored=$("$PY" -B "$REPRO/tools/ckpt.py" parquet "$CK")
if [ -f "$stored" ] && [ "$(cd "$(dirname "$stored")" && pwd -P)" = "$(cd "$(dirname "$PARQ")" && pwd -P)" ]; then
  ln -sfn "$(cd "$(dirname "$CK")" && pwd)/$(basename "$CK")" "$STAGED"
else
  src_sha=$(sha256_of "$CK")
  if [ ! -f "$STAGED" ] || [ -L "$STAGED" ] || [ "$(cat "$STAGED.source_sha256" 2>&1)" != "$src_sha" ]; then
    rm -f "$STAGED"
    "$PY" -B "$REPRO/tools/ckpt.py" relocate "$CK" "$STAGED" "$PARQ"
    echo "$src_sha" > "$STAGED.source_sha256"
  fi
  NOTES+=(--note "evaluated a copy whose stored task parquet points at $PARQ (stored: $stored)")
  say "stored parquet $stored differs from $PARQ: evaluating a relocated copy" | tee -a "$LOG"
fi

EVALUATED=-
tol=1e-4
[ -f "$EV/runs/ep$EP.export_tol_2e-4" ] && tol=2e-4
for t in $(seq 1 "$TRIALS"); do
  res=$EV/$(printf 't%02d' "$t")/results
  for try in 1 2; do
    say "trial $t try $try on gpu $GPU (export tolerance $tol)" | tee -a "$LOG"
    rm -f "$STAGE/eval_report.json"
    "$PY" -B "$REPO/scripts/track2/eval_all.py" --arm-dir "$STAGE" --arm "$METHOD" --iteration "$N" \
      --episodes "$EPN" --gpus "$GPU" --export yes --export-tolerance "$tol" --skip-pack --results "$res" \
      --stock-runtime "$FC_OFFICIAL" --kit-root "$V2D_KIT" --python "$PY" --code "$REPO" \
      --manifest "$V2D_DATA/whole_body/soma/track2_manifest.json" >> "$LOG" 2>&1 || true
    status=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["episodes"][0]["status"])' \
             "$STAGE/eval_report.json" 2>&1 || echo missing)
    if [ "$status" = evaluated ]; then break; fi
    case "$status" in
      export_failed*)
        if [ "$tol" = 1e-4 ]; then
          tol=2e-4
          echo "export refused at 1e-4 ($status); exported at 2e-4" > "$EV/runs/ep$EP.export_tol_2e-4"
          NOTES+=(--note "export refused at 1e-4, exported at 2e-4")
          rm -rf "$STAGE/export/ep$EP"
          continue
        fi
        echo "export failed at 1e-4 and 2e-4 ($status)" > "$EV/runs/ep$EP.export_failed"
        say "$METHOD ep$EP: EXPORT FAILED at both tolerances; the episode has no evaluation" | tee -a "$LOG"
        break 2 ;;
    esac
    say "trial $t try $try: $status (log $LOG)"
  done
  mkdir -p "$EV/$(printf 't%02d' "$t")/reports"
  cp -f "$STAGE/eval_report.json" "$EV/$(printf 't%02d' "$t")/reports/ep$EP.eval_report.json" 2>&1 || true
done
[ -f "$STAGE/export/ep$EP/$(basename "$STAGED")" ] && EVALUATED=$STAGE/export/ep$EP/$(basename "$STAGED")
[ -f "$EV/runs/ep$EP.export_tol_2e-4" ] && [ "$tol" = 2e-4 ] && NOTES+=(--note "exported at tolerance 2e-4")
[ -f "$EV/runs/ep$EP.export_failed" ] && NOTES+=(--note "export failed at 1e-4 and 2e-4: not evaluated")
"$PY" -B "$REPRO/tools/ckpt.py" summary "$METHOD" "$EP" "$EV" "$TRIALS" "$CK" "$EVALUATED" ${NOTES[@]+"${NOTES[@]}"} \
  > "$EV/ep$EP.json"
"$PY" -B "$REPRO/tools/ckpt.py" show "$EV/ep$EP.json"
n_ok=$("$PY" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["n_evaluated"])' "$EV/ep$EP.json")
[ "$n_ok" -gt 0 ] || die "$METHOD ep$EP: no official trial was evaluated (log $LOG)"
