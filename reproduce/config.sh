# Shared settings for the scripts in reproduce/. Sourced, not run.
#
# Every path comes from an environment variable. Set the ones a step needs (each script checks them):
#
#   PY           Python of the FlashCHORD environment (train, export, evaluation, Contact-AUC scoring).
#                Default: python
#   PY_V2D       Python of the robotic_grounding environment of NVIDIA video_to_data (SOMA body model and the
#                official G1 retargeter). Needed by prepare_data.sh and by contact_auc.sh (human hand side).
#   FC_OFFICIAL  NVIDIA's official FlashCHORD checkout, video_to_data @ 8c778f9, with LFS files:
#                <video_to_data>/robotic_grounding/flash_chord. CHORD trains and every method is evaluated here.
#   FC_GUARD     Only for train.sh chord_ppo 7 --nonfinite-guard: a separate copy of that checkout with
#                patches/chord_ppo_nonfinite_guard.diff applied (reproduce/README.md, "CHORD (PPO), episode 07").
#   V2D_KIT      The extracted V2D submission kit (contains examples/run_policy_evaluation.py).
#   V2D_DATA     Prepared Tier-1 inputs (prepare_data.sh writes them): the folder that holds whole_body/soma.
#                Exported as HUMAN_MOTION_DATA_DIR for FlashCHORD.
#   V2D_TRACK2   The challenge download, folder track_2 (holds tier_1_multiview_caption). prepare_data.sh and
#                contact_auc.sh read it.
#   RUNS         Training outputs, RUNS/<method>/epNN. Default: <repo>/runs
#   RESULTS      Evaluation, Contact-AUC and summary outputs. Default: $RUNS/results
#   TARGETS      DexDuet's per-episode human targets (prepare_data.sh writes them). Default: $V2D_DATA/dexduet_targets
#   HULL_CACHE   Optional. Convex-hull cache folder (holds convex/*.npz) used by every training run. FlashCHORD builds
#                the hulls on first use when this is unset; the paper's runs all used one fixed cache (HULL_CACHE.sha256).
#
# Methods (names used by every script):
#   dexduet                          DexDuet: per-hand policy + A-RSI + object task reward + two interaction rewards
#   dexduet_no_interaction_rewards   the same without the hand-graph and hand-object contact rewards
#   chord_ppo                        CHORD, NVIDIA's official PPO recipe, unmodified (episode 07: with the non-finite
#                                    guard of patches/chord_ppo_nonfinite_guard.diff)

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPRO=$REPO/reproduce
PY=${PY:-python}
RUNS=${RUNS:-$REPO/runs}
RESULTS=${RESULTS:-$RUNS/results}
OFFICIAL_COMMIT=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
GUARD_PATCH=$REPRO/patches/chord_ppo_nonfinite_guard.diff
METHODS="dexduet dexduet_no_interaction_rewards chord_ppo"

# The equal training budget: 20,000 iterations x 4,096 worlds x 24 steps = 1,966,080,000 environment steps, seed 42.
BUDGET_ITERS=20000
SEED=42
TRIALS=${TRIALS:-4}

# Keep thread pools small: one training run starts many processes, and shared machines often cap the process count.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1

say() { echo "[$(date '+%F %T')] $*"; }
die() { echo "error: $*" >&2; exit 2; }

need() {  # need VAR...: stop with a clear message when a setting is missing
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || die "set $v (see reproduce/config.sh)"
  done
}

need_official() {  # FC_OFFICIAL must be the official checkout at 8c778f9 with its LFS files
  need FC_OFFICIAL
  FC_OFFICIAL=$(cd "$FC_OFFICIAL" && pwd) || die "FC_OFFICIAL=$FC_OFFICIAL does not exist"
  [ -f "$FC_OFFICIAL/scripts/train_rl.py" ] || die "FC_OFFICIAL must point at <video_to_data>/robotic_grounding/flash_chord"
  local head
  head=$(git -C "$FC_OFFICIAL" rev-parse HEAD 2>&1) || head=unknown
  [ "$head" = "$OFFICIAL_COMMIT" ] || die "FC_OFFICIAL is at $head; check out $OFFICIAL_COMMIT"
  [ "$(wc -c < "$FC_OFFICIAL/src/flash_chord/assets/policies/sonic/encoder_batched.onnx")" -gt 100000 ] \
    || die "LFS files missing in FC_OFFICIAL: run git lfs install && git lfs pull in video_to_data"
  if [ -n "$(command -v sha256sum)" ]; then  # NVIDIA's PPO trainer and recipe, evaluator and core runtime
    (cd "$FC_OFFICIAL" && sha256sum --quiet -c "$REPRO/tools/official_pins.sha256" 2>&1) \
      || die "files in FC_OFFICIAL differ from video_to_data@8c778f9 (reproduce/tools/official_pins.sha256)"
  fi
}

need_guard() {  # FC_GUARD: a separate copy of the official checkout with $GUARD_PATCH applied. Checked: its commit,
  # its changed Python/YAML files, added ones under src/ and scripts/ (what the trainer loads; a .venv, outputs/ or
  # wandb/ folder in the copy is fine), the sha256 of the two patched files and of the other pinned files
  need FC_GUARD
  FC_GUARD=$(cd "$FC_GUARD" && pwd) || die "FC_GUARD=$FC_GUARD does not exist"
  [ -f "$FC_GUARD/scripts/train_rl.py" ] || die "FC_GUARD must point at <copy>/robotic_grounding/flash_chord"
  [ "$(cd "$FC_GUARD" && pwd -P)" != "$(cd "$FC_OFFICIAL" && pwd -P)" ] \
    || die "FC_GUARD must be a separate copy of the checkout; FC_OFFICIAL stays unmodified"
  local head changed added rel want have
  head=$(git -C "$FC_GUARD" rev-parse HEAD 2>&1) || head=unknown
  [ "$head" = "$OFFICIAL_COMMIT" ] || die "FC_GUARD is at $head; it must be a copy of $OFFICIAL_COMMIT"
  [ "$(wc -c < "$FC_GUARD/src/flash_chord/assets/policies/sonic/encoder_batched.onnx")" -gt 100000 ] \
    || die "LFS files missing in FC_GUARD: run git lfs pull in its video_to_data"
  changed=$(git -C "$FC_GUARD" diff --name-only --relative HEAD -- '*.py' '*.yaml' | tr '\n' ' ')
  [ "$changed" = "src/flash_chord/envs/base.py src/flash_chord/training/ppo/runner.py " ] \
    || die "FC_GUARD must differ from $OFFICIAL_COMMIT by $GUARD_PATCH alone (changed: ${changed:-nothing});" \
           "apply it with: git -C \"\$FC_GUARD\" apply $GUARD_PATCH"
  added=$(git -C "$FC_GUARD" ls-files --others -- 'src/*.py' 'src/*.yaml' 'scripts/*.py' 'scripts/*.yaml' | tr '\n' ' ')
  [ -z "$added" ] || die "FC_GUARD has Python or YAML files under src/ or scripts/ that $OFFICIAL_COMMIT does not have: $added"
  while read -r want rel; do
    have=$(sha256_of "$FC_GUARD/$rel")
    [ "$have" = "$want" ] || die "FC_GUARD/$rel differs from $OFFICIAL_COMMIT + $(basename "$GUARD_PATCH")"
  done < <(grep -v -e '^#' -e '^$' "${GUARD_PATCH%.diff}.sha256")
  if [ -n "$(command -v sha256sum)" ]; then  # every other pinned file unchanged
    (cd "$FC_GUARD" && grep -v -e ' src/flash_chord/envs/base.py$' -e ' src/flash_chord/training/ppo/runner.py$' \
       "$REPRO/tools/official_pins.sha256" \
       | sha256sum --quiet -c - 2>&1) \
      || die "files in FC_GUARD differ from video_to_data@8c778f9 (reproduce/tools/official_pins.sha256)"
  fi
}

need_data() {
  need V2D_DATA
  V2D_DATA=$(cd "$V2D_DATA" && pwd) || die "V2D_DATA=$V2D_DATA does not exist"
  [ -f "$V2D_DATA/whole_body/soma/track2_manifest.json" ] || die "$V2D_DATA has no whole_body/soma/track2_manifest.json; run prepare_data.sh"
  export HUMAN_MOTION_DATA_DIR=$V2D_DATA
  TARGETS=${TARGETS:-$V2D_DATA/dexduet_targets}
}

need_kit() {  # V2D_KIT must be the extracted V2D submission kit
  need V2D_KIT
  [ -f "$V2D_KIT/examples/run_policy_evaluation.py" ] \
    || die "V2D_KIT=$V2D_KIT has no examples/run_policy_evaluation.py; set it to the extracted V2D submission kit"
}

check_method() {
  case " $METHODS " in *" $1 "*) ;; *) die "unknown method '$1' (one of: $METHODS)";; esac
}

ep2() {  # ep2 7 -> 07
  case "$1" in ''|*[!0-9]*) die "episode must be a number 0-29, got '$1'";; esac
  [ "$((10#$1))" -le 29 ] || die "episode must be 0-29, got '$1'"
  printf '%02d' "$((10#$1))"
}

episode_field() {  # episode_field <NN> <column>: sequence_id | object | support (episodes.tsv)
  awk -F'\t' -v e="$1" -v c="$2" 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next } $1 == e { print $col[c] }' \
    "$REPRO/episodes.tsv"
}

parquet_of() {  # parquet_of <sequence_id>
  echo "$V2D_DATA/whole_body/soma/sequence_id=$1/robot_name=g1/data.parquet"
}

sha256_of() {
  if [ -n "$(command -v sha256sum)" ]; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
