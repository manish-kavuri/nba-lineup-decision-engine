"""Parse substitution rows and resolve names to ``PLAYER_ID`` via roster."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd


def _fold_ascii(s: str) -> str:
    """Lowercase ASCII-ish fold for last-name matching (handles Vučević vs Vucevic)."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.lower()

_SUB_RE = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$", re.IGNORECASE)


def parse_sub_description(description: object) -> Optional[tuple[str, str]]:
    if not isinstance(description, str):
        return None
    m = _SUB_RE.match(description.strip())
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _norm_tokens(s: str) -> list[str]:
    s = s.lower().replace(".", " ")
    parts = [p for p in re.split(r"\s+", s.strip()) if p]
    junk = {"jr", "sr", "ii", "iii", "iv", "v"}
    return [p for p in parts if p not in junk]


def resolve_name_to_player_id(name: str, team_id: int, players: pd.DataFrame) -> int:
    """Map a PBP sub fragment (e.g. ``Drummond``, ``Oubre Jr.``) to ``PLAYER_ID``."""
    name = str(name).strip()
    team_players = players[players["TEAM_ID"] == int(team_id)]
    if team_players.empty:
        raise ValueError(f"No players for team_id={team_id}")

    name_fold = _fold_ascii(name)
    exact = team_players[team_players["PLAYER_NAME"].map(_fold_ascii) == name_fold]
    if len(exact) == 1:
        return int(exact.iloc[0]["PLAYER_ID"])

    toks = _norm_tokens(name)
    last = _fold_ascii(toks[-1]) if toks else name_fold

    last_fold = team_players["PLAYER_NAME"].map(
        lambda x: _fold_ascii(str(x).split()[-1]) if pd.notna(x) else ""
    )
    last_name_match = team_players[last_fold == last]
    if len(last_name_match) == 1:
        return int(last_name_match.iloc[0]["PLAYER_ID"])

    sub = team_players[team_players["PLAYER_NAME"].map(_fold_ascii).str.contains(re.escape(last), na=False)]
    if len(sub) == 1:
        return int(sub.iloc[0]["PLAYER_ID"])

    raise ValueError(f"Could not resolve {name!r} for team {team_id}")


def substitution_player_ids(
    row: pd.Series, players: pd.DataFrame
) -> tuple[int, int, int]:
    """
    Return ``(team_id, player_in_id, player_out_id)`` for a substitution row.

    The feed's ``personId`` is the **player going out** (verified on sample rows).
    The incoming player is resolved from the first name in ``SUB: X FOR Y``.
    """
    if str(row.get("actionType", "")).strip().lower() != "substitution":
        raise ValueError("Not a substitution row")
    desc = row.get("description")
    parsed = parse_sub_description(desc) if desc is not None else None
    if not parsed:
        raise ValueError(f"Could not parse substitution description: {desc!r}")
    player_in_name, player_out_name = parsed
    tid = int(row["teamId"])
    pid_out = int(row["personId"])
    pid_in = resolve_name_to_player_id(player_in_name, tid, players)
    # Optional sanity: parsed "FOR <out>" should match personId when resolvable
    try:
        parsed_out = resolve_name_to_player_id(player_out_name, tid, players)
        if parsed_out != pid_out:
            pass  # trust personId from feed
    except ValueError:
        pass
    return tid, pid_in, pid_out


def apply_substitution(lineups: dict[int, set[int]], team_id: int, pid_in: int, pid_out: int) -> None:
    """Mutate ``lineups[team_id]``: swap ``pid_out`` for ``pid_in``."""
    s = lineups[team_id]
    if pid_out not in s:
        raise ValueError(f"player {pid_out} not on floor for team {team_id}, have {s}")
    if pid_in in s:
        raise ValueError(f"player {pid_in} already on floor for team {team_id}")
    s.remove(pid_out)
    s.add(pid_in)
