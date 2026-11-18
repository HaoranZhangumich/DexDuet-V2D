#!/usr/bin/env bash
# Reproduce the V2D results: train (or take our released checkpoints), evaluate, score Contact-AUC, summarize.
#
#   bash reproduce/run_all.sh [--from-release DIR] [--methods m1,m2] [--episodes all|0,5] [--gpus 0,1,2,3] [--per-gpu 2]
#
#   --from-release DIR  skip training and use the checkpoints of download_checkpoints.sh (DIR/v2d_g1_dex3/...)
#   --methods           default: dexduet,dexduet_no_interaction_rewards,chord_ppo
#   --episodes          default: all 30
#   --gpus, --per-gpu   jobs run in parallel, --per-gpu at a time on each listed GPU (default 0, 2 per GPU)
#
# One job = train.sh (unless --from-release), then evaluate.sh, then contact_auc.sh, for one (method, episode).
# Each step skips work that is already done, so the command can be re-run after an interruption.
# CHORD (PPO) on episode 07 trains with the non-finite guard (reproduce/README.md, Notes): with FC_GUARD set, this
# script passes --nonfinite-guard to train.sh for that job. Without FC_GUARD it refuses that job, unless the run
# already finished with the guard (then it is evaluated); an unfinished guarded run needs FC_GUARD to resume.
# Logs: $RESULTS/logs/<method>_epNN.log. At the end: summarize.py -> $RESULTS/summary.md. Exits with an error if
# any job failed (its FAILED line names the log).
#
# Time per training run in the paper, with two or three runs sharing a GPU: about 10 h for DexDuet on an RTX PRO 6000,
# 11 h for CHORD (PPO) on an H200. Evaluation and Contact-AUC take about 10 min per (method, episode).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

RELEASE=""
SEL_METHODS=${METHODS// /,}
EPISODES=all
GPUS=0
PER_GPU=2
while [ $# -gt 0 ]; do
  case "$1" in
    --from-release) RELEASE=$(cd "$2" && pwd); shift 2;;
    --methods) SEL_METHODS=$2; shift 2;;
    --episodes) EPISODES=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --per-gpu) PER_GPU=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
IFS=, read -r -a methods <<< "$SEL_METHODS"
for m in "${methods[@]}"; do check_method "$m"; done
if [ "$EPISODES" = all ]; then eps=($(tail -n +2 "$REPRO/episodes.tsv" | cut -f1))
else IFS=, read -r -a eps <<< "$EPISODES"; fi
IFS=, read -r -a gpus <<< "$GPUS"
mkdir -p "$RESULTS/logs"

jobs_list=()
for e in "${eps[@]}"; do for m in "${methods[@]}"; do jobs_list+=("$m $(ep2 "$e")"); done; done
say "${#jobs_list[@]} jobs on GPUs ${GPUS} (${PER_GPU} per GPU)${RELEASE:+, from released checkpoints in $RELEASE}"

one() {  # one <method> <NN> <gpu>
  local m=$1 e=$2 g=$3 rel=() guard=()
  [ -z "$RELEASE" ] || rel=(--from-release "$RELEASE")
  {
    if [ -z "$RELEASE" ] && [ "$m $e" = "chord_ppo 07" ]; then  # trains with the non-finite guard
      if [ -n "${FC_GUARD:-}" ]; then guard=(--nonfinite-guard)
      elif [ ! -f "$RUNS/chord_ppo/ep07/NONFINITE_GUARD" ] \
           || [ ! -f "$RUNS/chord_ppo/ep07/model_${SMOKE:-$BUDGET_ITERS}.safetensors" ]; then
        echo "chord_ppo ep07 trains with --nonfinite-guard: set FC_GUARD (reproduce/README.md, Notes)"; return 1
      fi
    fi
    if [ -z "$RELEASE" ]; then
      bash "$REPRO/train.sh" "$m" "$e" "$g" ${guard[@]+"${guard[@]}"} || { echo "train failed"; return 1; }
    fi
    bash "$REPRO/evaluate.sh" "$m" "$e" "$g" ${rel[@]+"${rel[@]}"} || { echo "evaluate failed"; return 1; }
    bash "$REPRO/contact_auc.sh" "$m" "$e" "$g" || { echo "contact_auc failed"; return 1; }
  } >> "$RESULTS/logs/${m}_ep$e.log" 2>&1
}

slots=$(( ${#gpus[@]} * PER_GPU ))
pids=()
for ((s = 0; s < slots; s++)); do
  (
    g=${gpus[$((s % ${#gpus[@]}))]}
    rc=0
    for ((i = s; i < ${#jobs_list[@]}; i += slots)); do
      read -r m e <<< "${jobs_list[$i]}"
      say "start $m ep$e on gpu $g"
      if one "$m" "$e" "$g"; then say "done $m ep$e"
      else say "FAILED $m ep$e (log $RESULTS/logs/${m}_ep$e.log)"; rc=1; fi
    done
    exit $rc
  ) &
  pids+=($!)
  [ -n "$RELEASE" ] || sleep 60  # stagger the first training starts (JIT compilation and graph capture use extra memory)
done
failed=0
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || failed=1; done
sum_rc=0
"$PY" -B "$REPRO/summarize.py" --results "$RESULTS" --methods "$SEL_METHODS" || sum_rc=$?
[ "$failed" = 0 ] || die "some jobs failed; see the FAILED lines above"
exit "$sum_rc"
