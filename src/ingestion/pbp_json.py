"""Load cached PlayByPlay JSON from ``data/raw/pbp/{game_id}.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import repo_root


def pbp_json_path(game_id: str, root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "data" / "raw" / "pbp" / f"{game_id}.json"


def load_pbp_actions_dataframe(game_id: str, root: Path | None = None) -> pd.DataFrame:
    """Return ``game.actions`` as a DataFrame sorted by ``actionNumber``."""
    path = pbp_json_path(game_id, root=root)
    with path.open(encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)
    actions = payload.get("game", {}).get("actions")
    if not actions:
        raise ValueError(f"No actions in {path}")
    df = pd.DataFrame(actions)
    if "actionNumber" in df.columns:
        df = df.sort_values("actionNumber", kind="mergesort").reset_index(drop=True)
    return df
