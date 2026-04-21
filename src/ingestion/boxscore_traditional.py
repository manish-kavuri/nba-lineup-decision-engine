"""Traditional box score (v3) — roster and opening lineups without deprecated v2."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3


def _player_stats_df_from_box(box: boxscoretraditionalv3.BoxScoreTraditionalV3) -> pd.DataFrame:
    df = box.player_stats.get_data_frame().copy()
    df["PLAYER_ID"] = df["personId"].astype(int)
    df["TEAM_ID"] = df["teamId"].astype(int)
    df["PLAYER_NAME"] = (df["firstName"].astype(str) + " " + df["familyName"].astype(str)).str.strip()
    return df


def _opening_lineups_from_raw(
    raw: dict[str, Any], *, n_starters: int = 5
) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for key in ("homeTeam", "awayTeam"):
        team = raw[key]
        tid = int(team["teamId"])
        players = team.get("players") or []
        if len(players) < n_starters:
            raise ValueError(f"Team {tid} has only {len(players)} player rows; expected at least {n_starters}.")
        ids = [int(p["personId"]) for p in players[:n_starters]]
        out[tid] = set(ids)
    return out


def load_boxscore_traditional_v3(
    game_id: str, *, n_starters: int = 5
) -> tuple[pd.DataFrame, dict[int, set[int]], int, int]:
    """
    One API call: player stats DataFrame, opening lineups by team_id, and
    ``(home_team_id, away_team_id)`` from the box score (for PBP home/away).

    Starters are the first ``n_starters`` players in each team's ``players``
    list (NBA list order; validated for sample game ``0042500111``).
    """
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    df = _player_stats_df_from_box(box)
    raw = box.get_dict()["boxScoreTraditional"]
    lineups = _opening_lineups_from_raw(raw, n_starters=n_starters)
    home_team_id = int(raw["homeTeamId"])
    away_team_id = int(raw["awayTeamId"])
    return df, lineups, home_team_id, away_team_id


def load_player_stats_traditional_v3(game_id: str) -> pd.DataFrame:
    """
    All players in the game with v3 column names plus legacy-style aliases
    for substitution name resolution (PLAYER_ID, PLAYER_NAME, TEAM_ID).
    """
    df, _, _, _ = load_boxscore_traditional_v3(game_id)
    return df


def opening_lineups_by_team_id_traditional_v3(game_id: str, n_starters: int = 5) -> dict[int, set[int]]:
    """
    Opening five per team from BoxScoreTraditionalV3.

    The NBA feed lists ``homeTeam.players`` / ``awayTeam.players`` with starters
    first (validated against box score for game ``0042500111``). We take the
    first ``n_starters`` ``personId`` values per team in API list order.
    """
    return load_boxscore_traditional_v3(game_id, n_starters=n_starters)[1]


def home_away_team_ids_traditional_v3(game_id: str) -> tuple[int, int]:
    """Return ``(home_team_id, away_team_id)`` from the traditional box score."""
    _, _, home_id, away_id = load_boxscore_traditional_v3(game_id)
    return home_id, away_id
