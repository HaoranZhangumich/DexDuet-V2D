#!/usr/bin/env bash
# Download the paper's trained V2D policies, so that evaluate.sh, contact_auc.sh and summarize.py run without training.
#
#   bash reproduce/download_checkpoints.sh [--out DIR] [--zip FILE]
#
# What it fetches:
#   * dexduet_v2d_g1_dex3.zip from the Hugging Face model repository oldTOM/dexduet-v2d-checkpoints (or
#     $DEXDUET_CKPT_REPO when set), at its main revision (or $DEXDUET_CKPT_REVISION: a branch, tag or commit of
#     that repository). It unpacks to
#     <out>/v2d_g1_dex3/<method>/epNN/ for dexduet, dexduet_no_interaction_rewards and chord_ppo; each folder holds the
#     trained checkpoint and RUN_INFO.json.
# The zip and every checkpoint listed in reproduce/CHECKPOINTS.sha256 are checked against it (the zip's sha256 also
# covers its other files); a zip without an entry there is refused unless ALLOW_UNVERIFIED=1. Uses the `hf` command
# when it is installed, else curl.
# The zip must be the release that this checkout's CHECKPOINTS.sha256 lists. When the Hugging Face repository holds
# another release, update this checkout, or set DEXDUET_CKPT_REVISION to the repository revision of that release.
#
#   --out DIR        destination (default <repo>/checkpoints)
#   --zip FILE       use an already downloaded dexduet_v2d_g1_dex3.zip instead of downloading it
# Then:  bash reproduce/run_all.sh --from-release <out>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

OUT=$REPO/checkpoints
ZIP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --zip) ZIP=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
SUMS=$REPRO/CHECKPOINTS.sha256

verify() {  # verify <path relative to OUT> [hint added to a mismatch error] [file, default OUT/<path>]
  local rel=$1 want have
  want=$(awk -v f="$rel" '$1 !~ /^#/ && $2 == f { print $1 }' "$SUMS")
  have=$(sha256_of "${3:-$OUT/$rel}")
  if [ -z "$want" ]; then
    [ "${ALLOW_UNVERIFIED:-0}" = 1 ] || die "$rel has no entry in reproduce/CHECKPOINTS.sha256 (sha256 $have); set ALLOW_UNVERIFIED=1 to use it anyway"
    say "unverified (no entry): $rel $have"
  elif [ "$want" != "$have" ]; then
    die "sha256 mismatch for $rel: $have, expected $want${2:+; $2}"
  else
    say "ok: $rel"
  fi
}

fetch() {  # fetch <repo> <repo type: model|dataset> <file in repo> <destination> <revision>
  local repo=$1 kind=$2 file=$3 dst=$4 rev=$5 url
  mkdir -p "$(dirname "$dst")"
  if [ -n "$(command -v hf)" ]; then
    local tmp=$OUT/.hf_download
    hf download "$repo" "$file" --repo-type "$kind" --revision "$rev" --local-dir "$tmp"
    mv -f "$tmp/$file" "$dst"
    rm -rf "$tmp"
  else
    if [ "$kind" = dataset ]; then url=https://huggingface.co/datasets/$repo/resolve/$rev/$file
    else url=https://huggingface.co/$repo/resolve/$rev/$file; fi
    curl -fL --retry 3 -o "$dst.part" "$url"
    mv -f "$dst.part" "$dst"
  fi
}

OTHER="this zip is another release than reproduce/CHECKPOINTS.sha256 lists: update this checkout, or use the zip"
OTHER+=" of the release it lists"
if [ -z "$ZIP" ]; then
  DEXDUET_CKPT_REPO=${DEXDUET_CKPT_REPO:-oldTOM/dexduet-v2d-checkpoints}
  DEXDUET_CKPT_REVISION=${DEXDUET_CKPT_REVISION:-main}
  ZIP=$OUT/dexduet_v2d_g1_dex3.zip
  [ -f "$ZIP" ] || fetch "$DEXDUET_CKPT_REPO" model dexduet_v2d_g1_dex3.zip "$ZIP" "$DEXDUET_CKPT_REVISION"
  verify dexduet_v2d_g1_dex3.zip "$OTHER (DEXDUET_CKPT_REVISION=<its Hugging Face revision>); delete $ZIP before downloading again"
else
  [ -f "$ZIP" ] || die "no such file: $ZIP"
  verify dexduet_v2d_g1_dex3.zip "$OTHER" "$ZIP"
fi
say "unpack $ZIP -> $OUT"
"$PY" -B - "$ZIP" "$OUT" <<'PYEOF'
import sys, zipfile
from pathlib import Path
zip_path, out = sys.argv[1], Path(sys.argv[2]).resolve()
with zipfile.ZipFile(zip_path) as z:
    for m in z.infolist():
        target = (out / m.filename).resolve()
        if not target.is_relative_to(out):
            sys.exit(f"zip member escapes the folder: {m.filename}")
    z.extractall(out)
PYEOF
while read -r _ rel; do
  case "$rel" in v2d_g1_dex3/dexduet*|v2d_g1_dex3/chord_ppo*)
    [ -f "$OUT/$rel" ] || die "$rel is listed in reproduce/CHECKPOINTS.sha256 but is not in the zip"
    verify "$rel";; esac
done < <(grep -v '^#' "$SUMS" | grep -v '^$' || true)

say "checkpoints in $OUT/v2d_g1_dex3; next: bash reproduce/run_all.sh --from-release $OUT"
