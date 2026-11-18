#!/usr/bin/env bash
# Build the Tier-1 training inputs of the 30 episodes as the paper used them, and DexDuet's human targets.
#
#   bash reproduce/prepare_data.sh [--episodes all|0,5,7] [--gpus 0,1,2,3] [--steps convert,retarget,targets,check]
#
# Steps (each skips what already exists):
#   convert   challenge Track 2 Tier-1 episodes -> SOMA inputs per episode (reproduce/tools/track2_to_soma_inputs.py):
#             human SOMA parameters, object poses (invisible frames interpolated), ground plane, object mesh, support.
#   retarget  NVIDIA's official retargeter, unchanged: robotic_grounding/scripts/retarget/soma_to_g1.py --save --fps 30.
#             Writes sequence_id=<id>/robot_name=g1/data.parquet, the G1 reference motion every method trains on.
#   targets   DexDuet's per-episode human targets for the two interaction rewards, in $TARGETS:
#             scripts/dexduet/soma_hands.py (SOMA hand joints and contact-region vertices, CPU, $PY_V2D),
#             scripts/dexduet/region_points.py (Dex3 contact-region surface points),
#             scripts/dexduet/build_targets.py (hand-graph targets, contact distances, object distance grids, checks; GPU),
#             build_targets.py --finalize (TARGETS.sha256; training refuses targets that do not match it).
#   check     the support column of reproduce/episodes.tsv equals the rule the trainers use (reproduce/tools/support_rule.py).
#
# Needs: V2D_TRACK2 (challenge download, folder track_2), V2D_DATA (output folder), PY_V2D (robotic_grounding
# environment, with the SOMA-X body model assets installed by robotic_grounding's setup), PY (FlashCHORD environment).
# V2D_RG: the robotic_grounding folder of a full video_to_data checkout at 8c778f9 (default: $FC_OFFICIAL/..).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

EPISODES=all
GPUS=0
STEPS=convert,retarget,targets,check
while [ $# -gt 0 ]; do
  case "$1" in
    --episodes) EPISODES=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
need V2D_DATA
mkdir -p "$V2D_DATA"
V2D_DATA=$(cd "$V2D_DATA" && pwd)
TARGETS=${TARGETS:-$V2D_DATA/dexduet_targets}
WORK=${TARGETS_WORK:-$V2D_DATA/dexduet_targets_work}
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }

episode_list() {  # the selected episodes as NN, one per line
  if [ "$EPISODES" = all ]; then tail -n +2 "$REPRO/episodes.tsv" | cut -f1
  else tr ',' '\n' <<< "$EPISODES" | while read -r e; do ep2 "$e"; done; fi
}
comma_list() { episode_list | sed 's/^0\([0-9]\)$/\1/' | paste -sd, -; }

if has convert; then
  need V2D_TRACK2 PY_V2D
  say "convert: Track 2 Tier-1 -> $V2D_DATA/whole_body/soma"
  eps=all
  [ "$EPISODES" = all ] || eps=$(comma_list)
  "$PY_V2D" -B "$REPRO/tools/track2_to_soma_inputs.py" --track-root "$V2D_TRACK2/tier_1_multiview_caption" \
    --motion-root "$V2D_DATA" --episodes "$eps"
fi

if has retarget; then
  need PY_V2D
  V2D_RG=${V2D_RG:-${FC_OFFICIAL:+$FC_OFFICIAL/..}}
  need V2D_RG
  V2D_RG=$(cd "$V2D_RG" && pwd)
  [ -f "$V2D_RG/scripts/retarget/soma_to_g1.py" ] || die "no scripts/retarget/soma_to_g1.py in V2D_RG=$V2D_RG"
  mkdir -p "$V2D_DATA/retarget_logs"
  IFS=, read -r -a gpu_list <<< "$GPUS"
  retarget_one() {  # retarget_one <sequence_id> <gpu>
    local seq=$1 gpu=$2 dir=$V2D_DATA/whole_body/soma/sequence_id=$1
    if [ -f "$dir/robot_name=g1/data.parquet" ]; then say "retarget: $seq exists"; return 0; fi
    say "retarget: $seq on gpu $gpu"
    (cd "$V2D_RG" && CUDA_VISIBLE_DEVICES=$gpu "$PY_V2D" scripts/retarget/soma_to_g1.py "$dir" --save --fps 30 \
       --motion-root "$V2D_DATA") > "$V2D_DATA/retarget_logs/$seq.log" 2>&1 \
      || { echo "retarget failed: $seq (log $V2D_DATA/retarget_logs/$seq.log)"; return 1; }
  }
  i=0
  pids=()
  for e in $(episode_list); do
    retarget_one "$(episode_field "$e" sequence_id)" "${gpu_list[$((i % ${#gpu_list[@]}))]}" &
    pids+=($!)
    i=$((i + 1))
    if [ $((i % ${#gpu_list[@]})) = 0 ]; then for p in "${pids[@]}"; do wait "$p" || die "retargeting failed"; done; pids=(); fi
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || die "retargeting failed"; done
fi

if has targets; then
  need PY_V2D
  [ -n "${SOMA_ASSETS:-}" ] || SOMA_ASSETS=${V2D_RG:-${FC_OFFICIAL:+$FC_OFFICIAL/..}}/source/robotic_grounding/robotic_grounding/assets/body_models/soma
  [ -d "$SOMA_ASSETS" ] || die "SOMA body model folder not found: $SOMA_ASSETS (set SOMA_ASSETS)"
  eps=all
  [ "$EPISODES" = all ] || eps=$(comma_list)
  mkdir -p "$WORK" "$TARGETS"
  say "targets 1/4: SOMA hand joints and contact-region vertices -> $WORK"
  (cd "$REPO" && CUDA_VISIBLE_DEVICES= "$PY_V2D" -B scripts/dexduet/soma_hands.py --data-dir "$V2D_DATA" \
     --work-dir "$WORK" --episodes "$eps" --soma-assets "$SOMA_ASSETS")
  say "targets 2/4: Dex3 contact-region points"
  [ -f "$TARGETS/dex3_region_points.npz" ] || (cd "$REPO" && "$PY" -B scripts/dexduet/region_points.py \
     --out "$TARGETS/dex3_region_points.npz")
  say "targets 3/4: hand-graph and contact targets, object grids (gpu ${GPUS%%,*})"
  (cd "$REPO" && PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PY" -B scripts/dexduet/build_targets.py \
     --data-dir "$V2D_DATA" --work-dir "$WORK" --targets-dir "$TARGETS" --check-dir "$WORK/checks" --episodes "$eps" \
     --code-commit "$(git -C "$REPO" rev-parse HEAD)")
  say "targets 4/4: TARGETS.sha256"
  (cd "$REPO" && "$PY" -B scripts/dexduet/build_targets.py --data-dir "$V2D_DATA" --work-dir "$WORK" \
     --targets-dir "$TARGETS" --check-dir "$WORK/checks" --code-commit "$(git -C "$REPO" rev-parse HEAD)" --finalize)
fi

if has check; then
  say "check: support flags of reproduce/episodes.tsv against reproduce/tools/support_rule.py"
  "$PY" -B - "$REPO" "$V2D_DATA" "$REPRO/episodes.tsv" <<'PYEOF'
import csv, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/reproduce/tools")
import support_rule
data = Path(sys.argv[2])
manifest = support_rule.load_manifest(data)
bad = 0
for row in csv.DictReader(open(sys.argv[3]), delimiter="\t"):
    ep = int(row["ep"])
    if ep not in manifest:
        continue
    seq, obj = manifest[ep]
    support = int(support_rule.support_available(data, seq))
    if seq != row["sequence_id"] or support != int(row["support"]):
        bad += 1
        print(f"ep{ep:02d}: manifest {seq} support {support} != episodes.tsv {row['sequence_id']} {row['support']}")
print(f"{len(manifest)} episodes checked, {bad} differ")
sys.exit(1 if bad else 0)
PYEOF
fi
say "prepare_data done"
