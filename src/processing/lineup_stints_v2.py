"""
Build NBA lineup stints from cached play-by-play JSON plus boxscore/player data.

Returns:
    fact_lineup_stint: one row per continuous lineup stint
    quality_df: one-row validation summary for the game
    lineup_debug_df: period-start lineup inference debug table
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from src.ingestion.pbp_json import load_pbp_actions_dataframe
from src.ingestion.boxscore_traditional import load_boxscore_traditional_v3
from src.processing.clock import parse_clock_seconds_remaining


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def canon_name(s: Any) -> str:
    """Normalize player names for matching substitution text to boxscore names."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _to_int(x: Any) -> Optional[int]:
    """Safely convert a value to int."""
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def get_period_start_clock(period: int) -> str:
    """NBA regulation periods are 12 minutes; overtime periods are 5 minutes."""
    return "PT12M00.00S" if int(period) <= 4 else "PT05M00.00S"


def get_period_start_seconds(period: int) -> float:
    """NBA regulation periods are 720 seconds; overtime periods are 300 seconds."""
    return 720.0 if int(period) <= 4 else 300.0


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Pick the first matching column from a list of possible names.
    Handles different naming styles from nba_api/project helpers.
    """
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {df.columns.tolist()}"
    )


def build_player_master_from_boxscore(
    players: pd.DataFrame,
    starters_by_team: dict[int, set[int]],
) -> pd.DataFrame:
    """
    Standardize the boxscore players dataframe into:
        team_id, player_id, display_name, starter
    """

    team_col = _pick_col(
        players,
        ["team_id", "TEAM_ID", "teamId", "TeamID"],
    )

    player_col = _pick_col(
        players,
        ["player_id", "PLAYER_ID", "personId", "PERSON_ID", "playerId", "person_id"],
    )

    name_col = _pick_col(
        players,
        [
            "display_name",
            "PLAYER_NAME",
            "player_name",
            "name",
            "NAME",
            "familyName",
            "playerName",
        ],
    )

    out = players[[team_col, player_col, name_col]].copy()

    out = out.rename(
        columns={
            team_col: "team_id",
            player_col: "player_id",
            name_col: "display_name",
        }
    )

    out["team_id"] = out["team_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["display_name"] = out["display_name"].astype(str)

    starter_pairs = set()

    for tid, starter_ids in starters_by_team.items():
        for pid in starter_ids:
            starter_pairs.add((int(tid), int(pid)))

    out["starter"] = out.apply(
        lambda r: (int(r["team_id"]), int(r["player_id"])) in starter_pairs,
        axis=1,
    )

    out = (
        out.drop_duplicates(subset=["team_id", "player_id"])
        .reset_index(drop=True)
    )

    return out


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def build_lineup_stints_for_game(
    game_id: str,
    *,
    pbp: Optional[pd.DataFrame] = None,
    root: Optional[Any] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build lineup stints for a single NBA game.

    Args:
        game_id:
            NBA game ID.
        pbp:
            Optional play-by-play dataframe. If None, cached JSON is loaded.
        root:
            Optional project root passed to load_pbp_actions_dataframe.

    Returns:
        fact_lineup_stint:
            One row per continuous lineup segment.
        quality_df:
            One-row quality summary.
        lineup_debug_df:
            Debug table showing inferred period-start lineups.
    """

    game_id = str(game_id).zfill(10)

    # -----------------------------------------------------------------
    # Load play-by-play
    # -----------------------------------------------------------------
    if pbp is None:
        pbp_df = load_pbp_actions_dataframe(game_id, root=root).copy()
    else:
        pbp_df = pbp.copy()

    pbp_df = (
        pbp_df
        .sort_values("actionNumber", kind="mergesort")
        .reset_index(drop=True)
    )

    pbp_df["clock_seconds_remaining"] = pbp_df["clock"].apply(
        parse_clock_seconds_remaining
    )

    pbp_df["scoreHome"] = (
        pd.to_numeric(pbp_df["scoreHome"], errors="coerce")
        .ffill()
        .fillna(0)
        .astype(int)
    )

    pbp_df["scoreAway"] = (
        pd.to_numeric(pbp_df["scoreAway"], errors="coerce")
        .ffill()
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------------------
    # Load boxscore/player data using your existing project helper
    # -----------------------------------------------------------------
    players, starters_by_team, home_team_id, away_team_id = load_boxscore_traditional_v3(
        game_id
    )

    home_team_id = int(home_team_id)
    away_team_id = int(away_team_id)

    player_master_df = build_player_master_from_boxscore(
        players=players,
        starters_by_team=starters_by_team,
    )

    player_id_to_name = dict(
        zip(player_master_df["player_id"], player_master_df["display_name"])
    )

    normalized_name_to_player_id = {
        (int(r["team_id"]), canon_name(r["display_name"])): int(r["player_id"])
        for _, r in player_master_df.iterrows()
    }

    q1_home_starters = set(
        player_master_df[
            (player_master_df["team_id"] == home_team_id)
            & (player_master_df["starter"])
        ]["player_id"].astype(int).tolist()
    )

    q1_away_starters = set(
        player_master_df[
            (player_master_df["team_id"] == away_team_id)
            & (player_master_df["starter"])
        ]["player_id"].astype(int).tolist()
    )

    if len(q1_home_starters) != 5 or len(q1_away_starters) != 5:
        raise ValueError(
            f"Starter issue for game {game_id}: "
            f"home={len(q1_home_starters)}, away={len(q1_away_starters)}"
        )

    # -----------------------------------------------------------------
    # Parse substitutions
    # -----------------------------------------------------------------
    action_type = pbp_df["actionType"].astype(str).str.strip().str.lower()
    sub_df = pbp_df[action_type.eq("substitution")].copy()

    sub_groups_df = pd.DataFrame(
        columns=["period", "clock", "n_subs", "substitutions"]
    )

    sub_group_lookup: dict[tuple[int, str], list[dict[str, Any]]] = {}

    if not sub_df.empty:
        sub_pattern = re.compile(
            r"^SUB:\s*(?P<player_in>.+?)\s+FOR\s+(?P<player_out>.+?)\s*$",
            re.IGNORECASE,
        )

        parsed = sub_df["description"].astype(str).str.extract(sub_pattern)

        sub_df["team_id"] = pd.to_numeric(
            sub_df["teamId"],
            errors="coerce",
        ).astype("Int64")

        # In the NBA live PBP feed, personId on substitution row is usually player OUT.
        sub_df["player_out_id"] = pd.to_numeric(
            sub_df["personId"],
            errors="coerce",
        ).astype("Int64")

        sub_df["player_in_name"] = parsed["player_in"].astype(str).str.strip()

        sub_df["player_in_id"] = sub_df.apply(
            lambda r: normalized_name_to_player_id.get(
                (int(r["team_id"]), canon_name(r["player_in_name"]))
            )
            if pd.notna(r["team_id"])
            else None,
            axis=1,
        )

        sub_events_clean = sub_df[
            [
                "actionNumber",
                "period",
                "clock",
                "team_id",
                "player_out_id",
                "player_in_id",
                "description",
            ]
        ].copy()

        sub_events_clean["player_out_id"] = pd.to_numeric(
            sub_events_clean["player_out_id"],
            errors="coerce",
        ).astype("Int64")

        sub_events_clean["player_in_id"] = pd.to_numeric(
            sub_events_clean["player_in_id"],
            errors="coerce",
        ).astype("Int64")

        sub_groups = []

        for (period, clock), g in sub_events_clean.groupby(
            ["period", "clock"],
            sort=False,
        ):
            sub_groups.append(
                {
                    "period": int(period),
                    "clock": clock,
                    "n_subs": len(g),
                    "substitutions": (
                        g.sort_values("actionNumber").to_dict("records")
                    ),
                }
            )

        sub_groups_df = pd.DataFrame(sub_groups)

        sub_group_lookup = {
            (int(r["period"]), r["clock"]): r["substitutions"]
            for _, r in sub_groups_df.iterrows()
        }

    # -----------------------------------------------------------------
    # Group all PBP events by timestamp
    # -----------------------------------------------------------------
    event_groups_df = (
        pbp_df.groupby(["period", "clock", "clock_seconds_remaining"], sort=False)
        .agg(
            n_events=("actionNumber", "count"),
            action_numbers=("actionNumber", list),
            action_types=("actionType", list),
        )
        .reset_index()
        .sort_values(
            ["period", "clock_seconds_remaining"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    sub_keys = set(
        zip(
            sub_groups_df["period"].astype(int),
            sub_groups_df["clock"],
        )
    )

    event_groups_df["has_substitution"] = event_groups_df.apply(
        lambda r: (int(r["period"]), r["clock"]) in sub_keys,
        axis=1,
    )

    # -----------------------------------------------------------------
    # Game-specific helper functions
    # -----------------------------------------------------------------
    def ids_to_names(ids: set[int] | list[int]) -> list[str]:
        return [
            player_id_to_name.get(int(i), f"id:{int(i)}")
            for i in sorted(set(int(x) for x in ids))
        ]

    def apply_sub_batch_ids(
        home_ids: set[int],
        away_ids: set[int],
        sub_batch: list[dict[str, Any]],
    ) -> tuple[set[int], set[int], list[dict[str, Any]]]:
        home = set(home_ids)
        away = set(away_ids)
        debug = []

        for sub in sub_batch or []:
            team_id = _to_int(sub.get("team_id"))
            out_id = _to_int(sub.get("player_out_id"))
            in_id = _to_int(sub.get("player_in_id"))

            lineup = None

            if team_id == home_team_id:
                lineup = home
            elif team_id == away_team_id:
                lineup = away

            if lineup is None:
                debug.append(
                    {
                        "team_id": team_id,
                        "out_id": out_id,
                        "in_id": in_id,
                        "applied": False,
                        "reason": "unknown_team",
                    }
                )
                continue

            ok = (
                out_id is not None
                and in_id is not None
                and out_id in lineup
                and in_id not in lineup
            )

            if ok:
                lineup.remove(out_id)
                lineup.add(in_id)
                reason = ""
            else:
                reason = "out_not_on_floor_or_in_already_on_floor"

            debug.append(
                {
                    "team_id": team_id,
                    "out_id": out_id,
                    "in_id": in_id,
                    "applied": ok,
                    "reason": reason,
                }
            )

        return home, away, debug

    def get_first_sub_roles_for_period(period: int) -> dict[str, Any]:
        q = sub_groups_df[sub_groups_df["period"].eq(period)].copy()

        if q.empty:
            return {
                "home": {},
                "away": {},
                "home_out": set(),
                "home_in": set(),
                "away_out": set(),
                "away_in": set(),
            }

        q["sec"] = q["clock"].apply(parse_clock_seconds_remaining)
        q = q.sort_values("sec", ascending=False)

        home_roles = {}
        away_roles = {}

        for _, row in q.iterrows():
            for sub in row["substitutions"]:
                team_id = _to_int(sub.get("team_id"))
                out_id = _to_int(sub.get("player_out_id"))
                in_id = _to_int(sub.get("player_in_id"))

                if team_id == home_team_id:
                    roles = home_roles
                elif team_id == away_team_id:
                    roles = away_roles
                else:
                    continue

                if out_id is not None and out_id not in roles:
                    roles[out_id] = "OUT"

                if in_id is not None and in_id not in roles:
                    roles[in_id] = "IN"

        return {
            "home": home_roles,
            "away": away_roles,
            "home_out": {pid for pid, role in home_roles.items() if role == "OUT"},
            "home_in": {pid for pid, role in home_roles.items() if role == "IN"},
            "away_out": {pid for pid, role in away_roles.items() if role == "OUT"},
            "away_in": {pid for pid, role in away_roles.items() if role == "IN"},
        }

    def get_first_sub_clock(period: int) -> Optional[str]:
        q = sub_groups_df[sub_groups_df["period"].eq(period)].copy()

        if q.empty:
            return None

        q["sec"] = q["clock"].apply(parse_clock_seconds_remaining)

        return str(q.sort_values("sec", ascending=False).iloc[0]["clock"])

    def get_players_seen_before_first_sub(
        period: int,
        first_sub_clock: Optional[str],
    ) -> tuple[set[int], set[int]]:
        q = pbp_df[pbp_df["period"].eq(period)].copy()

        fs = (
            parse_clock_seconds_remaining(first_sub_clock)
            if first_sub_clock is not None
            else None
        )

        period_start_sec = get_period_start_seconds(period)

        if fs is not None:
            q = q[
                (q["clock_seconds_remaining"] < period_start_sec)
                & (q["clock_seconds_remaining"] > fs)
            ]

        q = q[
            (~q["actionType"].astype(str).str.lower().eq("period"))
            & (~q["actionType"].astype(str).str.lower().eq("substitution"))
        ].copy()

        q["personId_int"] = pd.to_numeric(q["personId"], errors="coerce")
        q["teamId_int"] = pd.to_numeric(q["teamId"], errors="coerce")

        q = q[
            (q["personId_int"].notna())
            & (q["personId_int"] != 0)
            & (q["teamId_int"].isin([home_team_id, away_team_id]))
        ]

        home_seen = set(
            q[q["teamId_int"].eq(home_team_id)]["personId_int"]
            .astype(int)
            .tolist()
        )

        away_seen = set(
            q[q["teamId_int"].eq(away_team_id)]["personId_int"]
            .astype(int)
            .tolist()
        )

        return home_seen, away_seen

    def period_end_lineups_from_openers(
        period: int,
        home_open: set[int],
        away_open: set[int],
    ) -> tuple[set[int], set[int]]:
        q = sub_groups_df[sub_groups_df["period"].eq(period)].copy()

        if q.empty:
            return set(home_open), set(away_open)

        q["sec"] = q["clock"].apply(parse_clock_seconds_remaining)
        q = q.sort_values("sec", ascending=False)

        h = set(home_open)
        a = set(away_open)

        for _, row in q.iterrows():
            h, a, _ = apply_sub_batch_ids(
                h,
                a,
                sub_group_lookup.get((int(row["period"]), row["clock"]), []),
            )

        return h, a

    # -----------------------------------------------------------------
    # Infer period starting lineups
    # -----------------------------------------------------------------
    periods = sorted(pbp_df["period"].dropna().astype(int).unique().tolist())

    period_start_lineups_inferred = {
        1: {
            "home": set(q1_home_starters),
            "away": set(q1_away_starters),
        }
    }

    prev_home_end, prev_away_end = period_end_lineups_from_openers(
        1,
        q1_home_starters,
        q1_away_starters,
    )

    debug_rows = [
        {
            "game_id": game_id,
            "period": 1,
            "first_sub_clock": get_first_sub_clock(1),
            "home_final": ids_to_names(q1_home_starters),
            "away_final": ids_to_names(q1_away_starters),
            "home_ids": sorted(q1_home_starters),
            "away_ids": sorted(q1_away_starters),
            "home_size": len(q1_home_starters),
            "away_size": len(q1_away_starters),
            "method": "boxscore_starters",
        }
    ]

    for period in periods:
        if period == 1:
            continue

        roles = get_first_sub_roles_for_period(period)
        first_sub_clock = get_first_sub_clock(period)

        seen_home, seen_away = get_players_seen_before_first_sub(
            period,
            first_sub_clock,
        )

        # Q3 often resets to original starters after halftime.
        # Other periods use prior period end as fallback unless evidence says otherwise.
        fallback_home = set(q1_home_starters if period == 3 else prev_home_end)
        fallback_away = set(q1_away_starters if period == 3 else prev_away_end)

        home_candidates = (
            list(sorted(roles["home_out"]))
            + list(sorted(seen_home - roles["home_in"]))
            + list(sorted(fallback_home - roles["home_in"]))
        )

        away_candidates = (
            list(sorted(roles["away_out"]))
            + list(sorted(seen_away - roles["away_in"]))
            + list(sorted(fallback_away - roles["away_in"]))
        )

        inferred_home = []
        inferred_away = []

        for pid in home_candidates:
            if pid not in inferred_home:
                inferred_home.append(pid)
            if len(inferred_home) == 5:
                break

        for pid in away_candidates:
            if pid not in inferred_away:
                inferred_away.append(pid)
            if len(inferred_away) == 5:
                break

        period_start_lineups_inferred[period] = {
            "home": set(inferred_home),
            "away": set(inferred_away),
        }

        debug_rows.append(
            {
                "game_id": game_id,
                "period": period,
                "first_sub_clock": first_sub_clock,
                "home_final": ids_to_names(inferred_home),
                "away_final": ids_to_names(inferred_away),
                "home_ids": sorted(inferred_home),
                "away_ids": sorted(inferred_away),
                "home_size": len(inferred_home),
                "away_size": len(inferred_away),
                "method": "inferred_from_first_sub_seen_and_fallback",
            }
        )

        prev_home_end, prev_away_end = period_end_lineups_from_openers(
            period,
            set(inferred_home),
            set(inferred_away),
        )

    lineup_debug_df = pd.DataFrame(debug_rows)

    # -----------------------------------------------------------------
    # Build fact_lineup_stint
    # -----------------------------------------------------------------
    def score_snapshot(
        period: int,
        clock: str,
        last_known: tuple[int, int] = (0, 0),
    ) -> tuple[int, int]:
        rows_at_clock = pbp_df[
            (pbp_df["period"].eq(period))
            & (pbp_df["clock"].eq(clock))
        ].copy()

        if rows_at_clock.empty:
            return last_known

        sh = pd.to_numeric(rows_at_clock["scoreHome"], errors="coerce").dropna()
        sa = pd.to_numeric(rows_at_clock["scoreAway"], errors="coerce").dropna()

        return (
            int(sh.iloc[-1]) if len(sh) else int(last_known[0]),
            int(sa.iloc[-1]) if len(sa) else int(last_known[1]),
        )

    def build_row(
        period: int,
        start_clock: str,
        end_clock: str,
        home_ids: set[int],
        away_ids: set[int],
        sh0: int,
        sa0: int,
        sh1: int,
        sa1: int,
        ended_by_sub: bool,
    ) -> dict[str, Any]:
        s0 = float(parse_clock_seconds_remaining(start_clock))
        s1 = float(parse_clock_seconds_remaining(end_clock))

        home_ids_sorted = sorted(int(i) for i in home_ids)
        away_ids_sorted = sorted(int(i) for i in away_ids)

        return {
            "game_id": game_id,
            "period": int(period),
            "start_clock": start_clock,
            "end_clock": end_clock,
            "start_seconds_remaining": s0,
            "end_seconds_remaining": s1,
            "duration_seconds": s0 - s1,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_lineup_ids": home_ids_sorted,
            "away_lineup_ids": away_ids_sorted,
            "home_lineup_names": [
                player_id_to_name.get(i, f"id:{i}") for i in home_ids_sorted
            ],
            "away_lineup_names": [
                player_id_to_name.get(i, f"id:{i}") for i in away_ids_sorted
            ],
            "home_player_ids": ",".join(map(str, home_ids_sorted)),
            "away_player_ids": ",".join(map(str, away_ids_sorted)),
            "score_home_start": int(sh0),
            "score_away_start": int(sa0),
            "score_home_end": int(sh1),
            "score_away_end": int(sa1),
            "score_margin_home_start": int(sh0) - int(sa0),
            "score_margin_home_end": int(sh1) - int(sa1),
            "home_points_in_stint": int(sh1) - int(sh0),
            "away_points_in_stint": int(sa1) - int(sa0),
            "net_points_home": (int(sh1) - int(sh0)) - (int(sa1) - int(sa0)),
            "ended_by_substitution": bool(ended_by_sub),
        }

    rows = []
    last_known_score = (0, 0)

    for period in periods:
        if period not in period_start_lineups_inferred:
            continue

        home_state = set(period_start_lineups_inferred[period]["home"])
        away_state = set(period_start_lineups_inferred[period]["away"])

        start_clock = get_period_start_clock(period)
        sh_start, sa_start = score_snapshot(period, start_clock, last_known_score)

        cur_start_clock = start_clock
        cur_sh = sh_start
        cur_sa = sa_start

        period_groups = (
            event_groups_df[event_groups_df["period"].eq(period)]
            .sort_values("clock_seconds_remaining", ascending=False)
            .reset_index(drop=True)
        )

        for _, g in period_groups.iterrows():
            clock = str(g["clock"])

            sh_here, sa_here = score_snapshot(period, clock, last_known_score)
            last_known_score = (sh_here, sa_here)

            if not bool(g["has_substitution"]):
                continue

            rows.append(
                build_row(
                    period=period,
                    start_clock=cur_start_clock,
                    end_clock=clock,
                    home_ids=home_state,
                    away_ids=away_state,
                    sh0=cur_sh,
                    sa0=cur_sa,
                    sh1=sh_here,
                    sa1=sa_here,
                    ended_by_sub=True,
                )
            )

            home_state, away_state, _ = apply_sub_batch_ids(
                home_state,
                away_state,
                sub_group_lookup.get((int(period), clock), []),
            )

            cur_start_clock = clock
            cur_sh = sh_here
            cur_sa = sa_here

        end_clock = "PT00M00.00S"
        sh_end, sa_end = score_snapshot(period, end_clock, last_known_score)
        last_known_score = (sh_end, sa_end)

        rows.append(
            build_row(
                period=period,
                start_clock=cur_start_clock,
                end_clock=end_clock,
                home_ids=home_state,
                away_ids=away_state,
                sh0=cur_sh,
                sa0=cur_sa,
                sh1=sh_end,
                sa1=sa_end,
                ended_by_sub=False,
            )
        )

    fact_lineup_stint = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Quality checks
    # -----------------------------------------------------------------
    if fact_lineup_stint.empty:
        quality_df = pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "n_stints": 0,
                    "negative_durations": None,
                    "zero_durations": None,
                    "bad_home_lineup_size": None,
                    "bad_away_lineup_size": None,
                    "total_seconds": 0.0,
                    "boundary_issues": None,
                    "period_start_home_size_issues": int(
                        (lineup_debug_df["home_size"] != 5).sum()
                    )
                    if not lineup_debug_df.empty
                    else None,
                    "period_start_away_size_issues": int(
                        (lineup_debug_df["away_size"] != 5).sum()
                    )
                    if not lineup_debug_df.empty
                    else None,
                    "status": "failed",
                    "reason": "No stint rows produced",
                }
            ]
        )

        return fact_lineup_stint, quality_df, lineup_debug_df

    neg_dur = int((fact_lineup_stint["duration_seconds"] < 0).sum())
    zero_dur = int((fact_lineup_stint["duration_seconds"] == 0).sum())

    bad_home = int(
        fact_lineup_stint["home_lineup_ids"]
        .apply(lambda x: len(set(x)) != 5)
        .sum()
    )

    bad_away = int(
        fact_lineup_stint["away_lineup_ids"]
        .apply(lambda x: len(set(x)) != 5)
        .sum()
    )

    total_sec = float(fact_lineup_stint["duration_seconds"].sum())

    f = (
        fact_lineup_stint
        .sort_values(
            ["period", "start_seconds_remaining"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    boundary_issues = 0

    for i in range(len(f) - 1):
        a = f.iloc[i]
        b = f.iloc[i + 1]

        if int(a["period"]) != int(b["period"]):
            continue

        lineup_changed = (
            set(a["home_lineup_ids"]) != set(b["home_lineup_ids"])
            or set(a["away_lineup_ids"]) != set(b["away_lineup_ids"])
        )

        if lineup_changed and (
            not bool(a["ended_by_substitution"])
            or str(a["end_clock"]) != str(b["start_clock"])
        ):
            boundary_issues += 1

    period_start_home_size_issues = int((lineup_debug_df["home_size"] != 5).sum())
    period_start_away_size_issues = int((lineup_debug_df["away_size"] != 5).sum())

    status = "success"
    reason = ""

    if neg_dur > 0:
        status = "failed"
        reason = "Negative durations found"
    elif bad_home > 0 or bad_away > 0:
        status = "failed"
        reason = "Bad lineup size found"
    elif boundary_issues > 0:
        status = "warning"
        reason = "Boundary issues found"
    elif period_start_home_size_issues > 0 or period_start_away_size_issues > 0:
        status = "warning"
        reason = "Period start lineup size issue found"

    quality_df = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "n_stints": len(fact_lineup_stint),
                "negative_durations": neg_dur,
                "zero_durations": zero_dur,
                "bad_home_lineup_size": bad_home,
                "bad_away_lineup_size": bad_away,
                "total_seconds": total_sec,
                "boundary_issues": boundary_issues,
                "period_start_home_size_issues": period_start_home_size_issues,
                "period_start_away_size_issues": period_start_away_size_issues,
                "status": status,
                "reason": reason,
            }
        ]
    )

    return fact_lineup_stint, quality_df, lineup_debug_df