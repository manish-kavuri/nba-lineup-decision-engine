from pathlib import Path
import pandas as pd
import numpy as np
import re



PROJECT_ROOT = Path("/Users/manishkavuri/Desktop/nba-lineup-decision-engine")

INPUT_PATH = PROJECT_ROOT / "data/processed/final/fact_lineup_stints_v2_2024_25.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "team_lineup_stint_features_v1_2024_25.parquet"
SAMPLE_PATH = OUTPUT_DIR / "team_lineup_stint_features_v1_2024_25_sample.csv"





def parse_lineup_ids(x):
    """
    Robustly parse lineup IDs from:
    - Python lists
    - NumPy arrays
    - pandas arrays
    - comma-separated strings
    - bracketed space-separated strings from Parquet display
    """
    if x is None:
        return []

    if isinstance(x, float) and pd.isna(x):
        return []

    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        return [int(i) for i in list(x)]

    # Handles strings like:
    # "201143,201950,1627759"
    # "[201143, 201950, 1627759]"
    # "[ 201143  201950 1627759 1628369 1628401]"
    ids = re.findall(r"\d+", str(x))

    return [int(i) for i in ids]


def lineup_key(x):
    ids = parse_lineup_ids(x)

    if len(ids) == 0:
        return ""

    return "-".join(map(str, sorted(ids)))
def safe_rate(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def build_team_perspective_dataset(stints: pd.DataFrame) -> pd.DataFrame:
    stints = stints.copy()

    # Stable stint number within each game
    stints = stints.sort_values(
        ["game_id", "period", "start_seconds_remaining"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    stints["stint_number"] = stints.groupby("game_id").cumcount() + 1
    stints["stint_id"] = (
        stints["game_id"].astype(str)
        + "_"
        + stints["stint_number"].astype(str).str.zfill(3)
    )

    stints["duration_minutes"] = stints["duration_seconds"] / 60
    stints["is_overtime"] = stints["period"] > 4

    # -----------------------------
    # Home team perspective
    # -----------------------------
    home = pd.DataFrame({
        "stint_id": stints["stint_id"],
        "game_id": stints["game_id"],
        "stint_number": stints["stint_number"],
        "period": stints["period"],
        "is_overtime": stints["is_overtime"],
        "start_clock": stints["start_clock"],
        "end_clock": stints["end_clock"],
        "start_seconds_remaining": stints["start_seconds_remaining"],
        "end_seconds_remaining": stints["end_seconds_remaining"],
        "duration_seconds": stints["duration_seconds"],
        "duration_minutes": stints["duration_minutes"],

        "team_id": stints["home_team_id"],
        "opponent_team_id": stints["away_team_id"],
        "is_home": 1,

        "lineup_ids": stints["home_lineup_ids"],
        "opponent_lineup_ids": stints["away_lineup_ids"],
        "lineup_names": stints["home_lineup_names"],
        "opponent_lineup_names": stints["away_lineup_names"],

        "score_for_start": stints["score_home_start"],
        "score_against_start": stints["score_away_start"],
        "score_for_end": stints["score_home_end"],
        "score_against_end": stints["score_away_end"],

        "points_for": stints["home_points_in_stint"],
        "points_against": stints["away_points_in_stint"],
        "net_points": stints["net_points_home"],

        "ended_by_substitution": stints["ended_by_substitution"],
    })

    # -----------------------------
    # Away team perspective
    # -----------------------------
    away = pd.DataFrame({
        "stint_id": stints["stint_id"],
        "game_id": stints["game_id"],
        "stint_number": stints["stint_number"],
        "period": stints["period"],
        "is_overtime": stints["is_overtime"],
        "start_clock": stints["start_clock"],
        "end_clock": stints["end_clock"],
        "start_seconds_remaining": stints["start_seconds_remaining"],
        "end_seconds_remaining": stints["end_seconds_remaining"],
        "duration_seconds": stints["duration_seconds"],
        "duration_minutes": stints["duration_minutes"],

        "team_id": stints["away_team_id"],
        "opponent_team_id": stints["home_team_id"],
        "is_home": 0,

        "lineup_ids": stints["away_lineup_ids"],
        "opponent_lineup_ids": stints["home_lineup_ids"],
        "lineup_names": stints["away_lineup_names"],
        "opponent_lineup_names": stints["home_lineup_names"],

        "score_for_start": stints["score_away_start"],
        "score_against_start": stints["score_home_start"],
        "score_for_end": stints["score_away_end"],
        "score_against_end": stints["score_home_end"],

        "points_for": stints["away_points_in_stint"],
        "points_against": stints["home_points_in_stint"],
        "net_points": -stints["net_points_home"],

        "ended_by_substitution": stints["ended_by_substitution"],
    })

    team_stints = pd.concat([home, away], ignore_index=True)

    # -----------------------------
    # Derived features
    # -----------------------------
    team_stints["lineup_key"] = team_stints["lineup_ids"].apply(lineup_key)
    team_stints["opponent_lineup_key"] = team_stints["opponent_lineup_ids"].apply(lineup_key)

    team_stints["score_margin_start"] = (
        team_stints["score_for_start"] - team_stints["score_against_start"]
    )

    team_stints["score_margin_end"] = (
        team_stints["score_for_end"] - team_stints["score_against_end"]
    )

    team_stints["score_margin_change"] = (
        team_stints["score_margin_end"] - team_stints["score_margin_start"]
    )

    team_stints["points_for_per_min"] = team_stints.apply(
        lambda r: safe_rate(r["points_for"], r["duration_minutes"]),
        axis=1,
    )

    team_stints["points_against_per_min"] = team_stints.apply(
        lambda r: safe_rate(r["points_against"], r["duration_minutes"]),
        axis=1,
    )

    team_stints["net_points_per_min"] = team_stints.apply(
        lambda r: safe_rate(r["net_points"], r["duration_minutes"]),
        axis=1,
    )

    # Per-48 is rough but useful before possession estimates
    team_stints["net_points_per_48"] = team_stints["net_points_per_min"] * 48

    # Avoid modeling zero-duration rows directly
    team_stints["is_zero_duration"] = team_stints["duration_seconds"] == 0

    return team_stints


def main():
    print(f"Reading stints from: {INPUT_PATH}")

    stints = pd.read_parquet(INPUT_PATH)

    print("Input shape:", stints.shape)
    print("Unique games:", stints["game_id"].nunique())

    team_stints = build_team_perspective_dataset(stints)

    print("Output shape:", team_stints.shape)
    print("Unique games:", team_stints["game_id"].nunique())
    print("Unique lineups:", team_stints["lineup_key"].nunique())

    print("\nZero-duration rows:")
    print(team_stints["is_zero_duration"].value_counts())

    team_stints.to_parquet(OUTPUT_PATH, index=False)
    team_stints.head(1000).to_csv(SAMPLE_PATH, index=False)

    print(f"\nSaved feature dataset to: {OUTPUT_PATH}")
    print(f"Saved sample CSV to: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()