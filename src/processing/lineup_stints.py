"""Build a lineup-stint table from sorted PBP actions and opening lineups."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from src.processing.clock import parse_clock_seconds_remaining, period_length_seconds
from src.processing.substitutions import apply_substitution, substitution_player_ids

logger = logging.getLogger(__name__)


def _prep_pbp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scoreHome"] = pd.to_numeric(out["scoreHome"], errors="coerce").ffill().fillna(0)
    out["scoreAway"] = pd.to_numeric(out["scoreAway"], errors="coerce").ffill().fillna(0)
    out["sec_left_period"] = out["clock"].map(parse_clock_seconds_remaining)
    return out


def _duration_same_period(seg_sec: float, end_sec: float) -> float:
    d = float(seg_sec) - float(end_sec)
    return max(0.0, d)


def _fmt_player_csv(team_ids: set[int]) -> str:
    return ",".join(map(str, sorted(team_ids)))


def build_lineup_stints(
    game_id: str,
    pbp: pd.DataFrame,
    players: pd.DataFrame,
    starters_by_team: dict[int, set[int]],
    home_team_id: int,
    away_team_id: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Walk PBP in ``actionNumber`` order, apply substitutions, and emit one row per
    continuous segment where the 10 players on the floor are unchanged.

    Stints are split at period boundaries. Each row stores the **lineup on the
    floor during** that segment (before any substitution that ends the segment).

    Returns ``(stint_dataframe, warnings)``. A substitution is **skipped** (with
    a warning) if the outgoing player is not on the floor — this can happen when
    the PBP feed omits substitution rows that other NBA endpoints include.
    """
    df = _prep_pbp(pbp.sort_values("actionNumber", kind="mergesort").reset_index(drop=True))
    warnings: list[str] = []

    lineups: dict[int, set[int]] = {
        home_team_id: set(starters_by_team[home_team_id]),
        away_team_id: set(starters_by_team[away_team_id]),
    }
    if len(lineups[home_team_id]) != 5 or len(lineups[away_team_id]) != 5:
        raise ValueError("Starters must be 5 players per team")

    first = df.iloc[0]
    seg_period = int(first["period"])
    seg_sec = float(first["sec_left_period"]) if pd.notna(first["sec_left_period"]) else period_length_seconds(seg_period)
    seg_clock = str(first["clock"])
    score_home_start = int(first["scoreHome"])
    score_away_start = int(first["scoreAway"])

    rows_out: list[dict[str, Any]] = []
    prev_period: Optional[int] = None

    def append_stint(
        *,
        stint_period: int,
        start_clock: str,
        start_sec: float,
        end_clock: str,
        end_sec: float,
        home_ids: str,
        away_ids: str,
        ended_sub: bool,
        sh_s: int,
        sa_s: int,
        sh_e: int,
        sa_e: int,
    ) -> None:
        dur = _duration_same_period(start_sec, end_sec)
        rows_out.append(
            {
                "game_id": game_id,
                "period": stint_period,
                "start_clock": start_clock,
                "end_clock": end_clock,
                "duration_seconds": dur,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_player_ids": home_ids,
                "away_player_ids": away_ids,
                "score_home_start": sh_s,
                "score_away_start": sa_s,
                "score_home_end": sh_e,
                "score_away_end": sa_e,
                "ended_by_substitution": ended_sub,
            }
        )

    for i in range(len(df)):
        row = df.iloc[i]
        period = int(row["period"])
        clock = str(row["clock"])
        sec_left = float(row["sec_left_period"]) if pd.notna(row["sec_left_period"]) else seg_sec
        sh = int(row["scoreHome"])
        sa = int(row["scoreAway"])

        # New period: close previous period's stint at 0:00
        if prev_period is not None and period > prev_period:
            prev_row = df.iloc[i - 1]
            append_stint(
                stint_period=prev_period,
                start_clock=seg_clock,
                start_sec=seg_sec,
                end_clock="PT00M00.00S",
                end_sec=0.0,
                home_ids=_fmt_player_csv(lineups[home_team_id]),
                away_ids=_fmt_player_csv(lineups[away_team_id]),
                ended_sub=False,
                sh_s=score_home_start,
                sa_s=score_away_start,
                sh_e=int(prev_row["scoreHome"]),
                sa_e=int(prev_row["scoreAway"]),
            )
            seg_period = period
            seg_sec = sec_left
            seg_clock = clock
            score_home_start = sh
            score_away_start = sa

        prev_period = period

        if str(row.get("actionType", "")).strip().lower() == "substitution":
            before_h = frozenset(lineups[home_team_id])
            before_a = frozenset(lineups[away_team_id])
            try:
                tid, pid_in, pid_out = substitution_player_ids(row, players)
                apply_substitution(lineups, tid, pid_in, pid_out)
            except ValueError as e:
                msg = f"actionNumber={row.get('actionNumber')}: {e}"
                warnings.append(msg)
                logger.debug("%s", msg)
                continue
            after_h = frozenset(lineups[home_team_id])
            after_a = frozenset(lineups[away_team_id])
            if (before_h, before_a) == (after_h, after_a):
                continue
            append_stint(
                stint_period=seg_period,
                start_clock=seg_clock,
                start_sec=seg_sec,
                end_clock=clock,
                end_sec=sec_left,
                home_ids=_fmt_player_csv(before_h),
                away_ids=_fmt_player_csv(before_a),
                ended_sub=True,
                sh_s=score_home_start,
                sa_s=score_away_start,
                sh_e=sh,
                sa_e=sa,
            )
            seg_period = period
            seg_sec = sec_left
            seg_clock = clock
            score_home_start = sh
            score_away_start = sa

    last = df.iloc[-1]
    last_sec = float(last["sec_left_period"]) if pd.notna(last["sec_left_period"]) else 0.0
    append_stint(
        stint_period=seg_period,
        start_clock=seg_clock,
        start_sec=seg_sec,
        end_clock=str(last["clock"]),
        end_sec=last_sec,
        home_ids=_fmt_player_csv(lineups[home_team_id]),
        away_ids=_fmt_player_csv(lineups[away_team_id]),
        ended_sub=False,
        sh_s=score_home_start,
        sa_s=score_away_start,
        sh_e=int(last["scoreHome"]),
        sa_e=int(last["scoreAway"]),
    )

    return pd.DataFrame(rows_out), warnings


def build_lineup_stints_for_game(
    game_id: str,
    *,
    pbp: Optional[pd.DataFrame] = None,
    root: Optional[Any] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load box score (API) + cached PBP JSON unless ``pbp`` is passed, then build stints.
    """
    from src.ingestion.boxscore_traditional import load_boxscore_traditional_v3
    from src.ingestion.pbp_json import load_pbp_actions_dataframe

    players, starters, home_id, away_id = load_boxscore_traditional_v3(game_id)
    if pbp is None:
        pbp = load_pbp_actions_dataframe(game_id, root=root)
    return build_lineup_stints(game_id, pbp, players, starters, home_id, away_id)
