#!/usr/bin/env bash
# Train one method on one Tier-1 episode at the paper's equal budget.
#
#   bash reproduce/train.sh <method> <episode> [gpu] [--check] [--nonfinite-guard]
#
#   method   dexduet | dexduet_no_interaction_rewards | chord_ppo
#   episode  0-29 (reproduce/episodes.tsv)
#   gpu      CUDA device index or UUID (default 0)
#   --check  only compose and check the config (CPU), do not train
#   --nonfinite-guard  chord_ppo only, used for episode 07: train with NVIDIA's code plus
#            patches/chord_ppo_nonfinite_guard.diff, from the separate patched copy $FC_GUARD (reproduce/README.md)
#
# Output: $RUNS/<method>/epNN/ (checkpoints, train.log, config_check.yaml, hydra/, offline W&B files).
#
# Budget: 20,000 iterations x 4,096 worlds x 24 steps = 1,966,080,000 environment steps, seed 42, G1 + Dex3.
#   dexduet, dexduet_no_interaction_rewards: this repository, training.max_iterations=20000. The run may stop earlier by
#     its own rule (a fixed number of iterations after A-RSI has withdrawn all help); final.json names the checkpoint.
#   chord_ppo: NVIDIA's official code ($FC_OFFICIAL), experiment g1_recon_body_ppo, unchanged (its default budget).
#     With --nonfinite-guard the same command runs in $FC_GUARD, which differs from $FC_OFFICIAL by the guard alone.
# Before training, the script composes the full Hydra config (--cfg job) and refuses to train unless the budget, seed,
# robot (G1Dex3) and reward weights are the ones above.
#
# Re-running the same command skips a finished run and resumes an unfinished one from its newest checkpoint.
# Test only: SMOKE_ITERS=<n> trains n iterations after the same check.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

POS=()
CHECK_ONLY=0
GUARD=0
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1;;
    --nonfinite-guard) GUARD=1;;
    *) POS+=("$a");;
  esac
done
METHOD=${POS[0]:?usage: train.sh <method> <episode> [gpu] [--check] [--nonfinite-guard]}
EP=$(ep2 "${POS[1]:?usage: train.sh <method> <episode> [gpu] [--check] [--nonfinite-guard]}")
GPU=${POS[2]:-0}
check_method "$METHOD"
[ "$GUARD" = 0 ] || [ "$METHOD" = chord_ppo ] || die "--nonfinite-guard is for chord_ppo only"
need_data
SEQ=$(episode_field "$EP" sequence_id)
OBJ=$(episode_field "$EP" object)
SUPPORT=$(episode_field "$EP" support)
PARQ=$(parquet_of "$SEQ")
[ -f "$PARQ" ] || die "missing $PARQ (run prepare_data.sh)"
OUT=$RUNS/$METHOD/ep$EP
SMOKE=${SMOKE_ITERS:-}

case "$METHOD" in
  dexduet|dexduet_no_interaction_rewards)
    CODE=$REPO
    SCRIPT=scripts/train_rl.py
    [ -e "$REPO/src/flash_chord/assets/policies/sonic/encoder_batched.onnx" ] || die "assets missing: run reproduce/setup_assets.sh"
    if [ "$METHOD" = dexduet ]; then
      EXP=g1_recon_body_coord_rsi_r2_dexduet
      [ -f "$TARGETS/TARGETS.sha256" ] || die "DexDuet targets missing in $TARGETS (run prepare_data.sh)"
    else
      EXP=g1_recon_body_coord_rsi_r2
    fi
    ARGS=(experiment=$EXP "training.max_iterations=$BUDGET_ITERS" +logging.local_jsonl=true)
    if [ "$METHOD" = dexduet ]; then ARGS+=("objective.dexduet_targets_dir='$TARGETS'"); fi
    ;;
  chord_ppo)
    need_official
    CODE=$FC_OFFICIAL
    if [ "$GUARD" = 1 ]; then need_guard; CODE=$FC_GUARD; fi
    SCRIPT=scripts/train_rl.py
    ARGS=(experiment=g1_recon_body_ppo)
    ;;
esac
ARGS+=("task.parquet='$PARQ'" "reset.seed=$SEED" "output_dir='$OUT'" logging.mode=offline "logging.experiment=$METHOD"
       "logging.run_name=ep${EP}_$OBJ" logging.upload_checkpoints=false)
[ "$SUPPORT" = 1 ] || ARGS+=(scene.support=false)  # episodes whose support file has no shape: floor only

trainer_env() {  # run a command in the trainer's environment (cwd = the code tree)
  local cache=()
  [ -z "${HULL_CACHE:-}" ] || cache=("FLASH_CHORD_CACHE_DIR=$HULL_CACHE")
  (cd "$CODE" && env -u PYTHONHOME -u PYTHONSTARTUP -u WANDB_API_KEY -u WANDB_MODE -u WANDB_ENTITY -u WANDB_PROJECT \
     PYTHONPATH="$CODE/src" HUMAN_MOTION_DATA_DIR="$V2D_DATA" WANDB_DIR="$OUT/wandb_dir" CUDA_VISIBLE_DEVICES="$GPU" \
     ${cache[@]+"${cache[@]}"} "$@")
}

check_config() {  # compose the exact config of this run and refuse anything but the paper's recipe and budget
  mkdir -p "$OUT"
  trainer_env "$PY" -B "$SCRIPT" "${ARGS[@]}" --cfg job --resolve > "$OUT/config_check.yaml"
  "$PY" -B "$REPRO/tools/check_config.py" "$OUT/config_check.yaml" --method "$METHOD" --parquet "$PARQ" \
    --targets "${TARGETS:-}"
}

guard_loaded() {  # the trainer imports the guarded code from $FC_GUARD
  local where
  where=$(trainer_env "$PY" -B -c '
import os, flash_chord.envs.base as b
print(os.path.realpath(b.__file__) if hasattr(b.BaseEnv, "nonfinite_report") else "no guard")' 2>&1 | tail -n 1)
  [ "$where" = "$(cd "$FC_GUARD/src/flash_chord/envs" && pwd -P)/base.py" ] \
    || die "the trainer does not load the guarded code from FC_GUARD (got: $where)"
}

newest() {  # newest <prefix>: largest N of $OUT/<prefix>_N.safetensors, or nothing
  ls "$OUT" 2>&1 | sed -n "s/^$1_\([0-9][0-9]*\)\.safetensors$/\1/p" | sort -n | tail -n 1
}

state() {  # done | resume:<checkpoint> | fresh
  local n iters=${SMOKE:-$BUDGET_ITERS}
  case "$METHOD" in
    dexduet|dexduet_no_interaction_rewards)
      if [ -f "$OUT/final.json" ] && "$PY" -B -c '
import json, sys
d = json.load(open(sys.argv[1] + "/final.json"))
ok = d.get("max_iterations") == int(sys.argv[2]) and d.get("checkpoint") == "model_%s.safetensors" % d.get("iteration")
sys.exit(0 if ok else 1)' "$OUT" "$iters"; then echo done; return; fi
      for n in $(ls "$OUT" 2>&1 | sed -n 's/^model_\([0-9][0-9]*\)\.safetensors$/\1/p' | sort -rn); do
        if [ "$n" -lt "$iters" ] && [ -f "$OUT/rsi_$n.safetensors" ]; then echo "resume:$OUT/model_$n.safetensors"; return; fi
      done
      echo fresh ;;
    chord_ppo)
      if [ -f "$OUT/model_$iters.safetensors" ]; then echo done; return; fi
      n=$(newest model); if [ -n "$n" ]; then echo "resume:$OUT/model_$n.safetensors"; else echo fresh; fi ;;
  esac
}

if [ -n "${HULL_CACHE:-}" ] && [ -n "$(command -v sha256sum)" ]; then  # the paper's runs all used one fixed hull cache
  (cd "$HULL_CACHE/convex" && sha256sum --quiet -c "$REPRO/HULL_CACHE.sha256" 2>&1) \
    || say "note: HULL_CACHE differs from the paper's hull cache (reproduce/HULL_CACHE.sha256)"
fi
mkdir -p "$OUT"
if [ "$CHECK_ONLY" = 1 ]; then
  GPU=  # composing the config needs no GPU
  check_config || die "$METHOD ep$EP: config check failed (see $OUT/config_check.yaml)"
  [ "$GUARD" = 0 ] || guard_loaded
  exit 0
fi
st=$(state)
MARK=$OUT/NONFINITE_GUARD  # present when the run trains with --nonfinite-guard; a resume must use the same code
if [ "$st" = done ]; then
  [ "$GUARD" = 0 ] || [ -f "$MARK" ] \
    || die "$OUT is a finished run trained without --nonfinite-guard; move it away to train with the guard"
  say "$METHOD ep$EP: already trained ($OUT)"; exit 0
fi
check_config || die "$METHOD ep$EP: config check failed, not training (see $OUT/config_check.yaml)"
if [ "$st" = fresh ] && [ -f "$MARK" ]; then  # started with the guard but no checkpoint saved yet
  [ "$GUARD" = 1 ] || die "$OUT was started with --nonfinite-guard; start it again with the same flag, or delete $MARK"
  rm -f "$MARK"  # rewritten below
fi
if [[ $st == resume:* ]] && [ "$METHOD" = chord_ppo ]; then
  [ "$GUARD" = 1 ] || [ ! -f "$MARK" ] || die "$OUT was started with --nonfinite-guard; resume it with the same flag"
  [ "$GUARD" = 0 ] || [ -f "$MARK" ] || die "$OUT was started without --nonfinite-guard; resume it without the flag"
fi
if [ "$GUARD" = 1 ]; then
  guard_loaded
  sha256_of "$GUARD_PATCH" > "$MARK"
fi

RUN_ARGS=("${ARGS[@]}" "hydra.run.dir='$OUT/hydra/$(date +%Y%m%d_%H%M%S)'")
if [[ $st == resume:* ]]; then
  RUN_ARGS+=(training.resume=true "training.checkpoint='${st#resume:}'")
fi
if [ -n "$SMOKE" ]; then
  say "SMOKE TEST: $SMOKE iterations only (not the paper's budget)"
  RUN_ARGS+=("training.max_iterations=$SMOKE" "training.save_interval=$SMOKE")
fi
GUARD_NOTE=""
[ "$GUARD" = 0 ] || GUARD_NOTE=" + $(basename "$GUARD_PATCH")"
{ echo "===== [$(date '+%F %T')] $METHOD ep$EP ($OBJ) gpu $GPU ($st) code $(git -C "$CODE" rev-parse HEAD 2>&1)$GUARD_NOTE"
  printf '%q ' "$PY" "$SCRIPT" "${RUN_ARGS[@]}"; echo; } >> "$OUT/train.log"
say "$METHOD ep$EP ($OBJ) on gpu $GPU ($st) -> $OUT"
rc=0
trainer_env "$PY" -B "$SCRIPT" "${RUN_ARGS[@]}" >> "$OUT/train.log" 2>&1 || rc=$?
st=$(state)
say "$METHOD ep$EP: trainer exit $rc, state $st"
[ "$st" = done ] || exit 1
