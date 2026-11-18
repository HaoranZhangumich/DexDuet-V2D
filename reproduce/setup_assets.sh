#!/usr/bin/env bash
# Link the large FlashCHORD assets that git does not hold (robot meshes and the SONIC policy) from NVIDIA's official
# checkout into this repository. Run once before training DexDuet.
#
#   FC_OFFICIAL=<video_to_data>/robotic_grounding/flash_chord bash reproduce/setup_assets.sh [--copy]
#
# --copy copies the folders instead of linking them. The folders are listed in .gitignore.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_official
MODE=link
[ "${1:-}" = --copy ] && MODE=copy
for rel in meshes/g1 meshes/sharpawave meshes/vega_sharpa policies/sonic; do
  src=$FC_OFFICIAL/src/flash_chord/assets/$rel
  dst=$REPO/src/flash_chord/assets/$rel
  [ -d "$src" ] || die "missing $src"
  if [ -e "$dst" ] || [ -L "$dst" ]; then say "exists: $dst"; continue; fi
  mkdir -p "$(dirname "$dst")"
  if [ "$MODE" = copy ]; then cp -R "$src" "$dst"; else ln -s "$src" "$dst"; fi
  say "$MODE: $dst"
done
