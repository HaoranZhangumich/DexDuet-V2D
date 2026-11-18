# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""The support rule of the training runs, used by the ``check`` step of reproduce/prepare_data.sh.

An episode trains with its support surface only when the support stage file exists and holds a shape
(``def Cylinder`` or ``def Cube``); otherwise it trains with ``scene.support=false`` (column ``support`` of
reproduce/episodes.tsv).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SUPPORT_PATTERN = re.compile(r"def (Cylinder|Cube)")


def load_manifest(data_dir: Path) -> dict[int, tuple[str, str]]:
    manifest_path = data_dir / "whole_body" / "soma" / "track2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes: dict[int, tuple[str, str]] = {}
    for sequence, record in manifest.items():
        episode = int(record["episode_index"])
        if episode in episodes:
            raise ValueError(f"{manifest_path}: episode {episode} appears twice")
        episodes[episode] = (str(record.get("sequence_id", sequence)), str(record["object"]))
    return episodes


def support_available(data_dir: Path, sequence: str) -> bool:
    support = data_dir / "whole_body" / "reconstructed_stage" / f"{sequence}_support.usda"
    if not support.is_file():
        return False
    return SUPPORT_PATTERN.search(support.read_text(errors="replace")) is not None
