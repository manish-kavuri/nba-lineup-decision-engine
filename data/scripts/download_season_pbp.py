import json
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import playbyplay as live_playbyplay


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path("/Users/manishkavuri/Desktop/nba-lineup-decision-engine")

# Change this if you want a different season
SEASON = "2024-25"
SEASON_TYPE = "Regular Season"

RAW_PBP_DIR = PROJECT_ROOT / "data/raw/pbp"
METADATA_DIR = PROJECT_ROOT / "data/metadata"

RAW_PBP_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_SECONDS = 1.5


# -----------------------------
# Get all game IDs
# -----------------------------
def get_season_games(season: str, season_type: str) -> pd.DataFrame:
    """
    Gets all NBA games for a given season and season type.

    LeagueGameFinder returns one row per team per game, so each game appears twice.
    We drop duplicates by GAME_ID.
    """
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type
    )

    games = finder.get_data_frames()[0].copy()

    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    # Keep useful columns if they exist
    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "WL",
        "PTS",
    ]

    keep_cols = [col for col in keep_cols if col in games.columns]

    games = games[keep_cols].copy()

    # One row per game
    one_row_per_game = (
        games.sort_values("GAME_DATE")
        .drop_duplicates(subset=["GAME_ID"])
        .reset_index(drop=True)
    )

    return one_row_per_game


# -----------------------------
# Download play-by-play
# -----------------------------
def download_pbp_json(game_id: str, force_refresh: bool = False):
    """
    Downloads and caches raw play-by-play JSON.
    If the file already exists, it skips unless force_refresh=True.

    Returns:
        local_path, downloaded
    """
    game_id = str(game_id).zfill(10)
    local_path = RAW_PBP_DIR / f"{game_id}.json"

    if local_path.exists() and not force_refresh:
        print(f"Already exists, skipping: {game_id}")
        return local_path, False

    print(f"Downloading play-by-play for {game_id}...")

    pbp = live_playbyplay.PlayByPlay(game_id=game_id)
    raw = pbp.get_dict()

    actions = raw.get("game", {}).get("actions", [])

    if not actions:
        raise ValueError(f"No play-by-play actions found for game {game_id}")

    with open(local_path, "w") as f:
        json.dump(raw, f)

    print(f"Saved: {local_path}")

    return local_path, True

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    print(f"Getting games for {SEASON} - {SEASON_TYPE}")

    games_df = get_season_games(
        season=SEASON,
        season_type=SEASON_TYPE
    )

    metadata_path = METADATA_DIR / f"nba_games_{SEASON.replace('-', '_')}_{SEASON_TYPE.replace(' ', '_')}.csv"
    games_df.to_csv(metadata_path, index=False)

    game_ids = games_df["GAME_ID"].tolist()

    print(f"\nFound {len(game_ids)} unique games")
    print(f"Saved game metadata to: {metadata_path}")

    success = []
    errors = []

    for i, game_id in enumerate(game_ids, start=1):
        print(f"\n[{i}/{len(game_ids)}] {game_id}")

        try:
            local_path = download_pbp_json(game_id)

            success.append({
                "game_id": game_id,
                "local_path": str(local_path),
                "status": "success"
            })

        except Exception as e:
            print(f"FAILED {game_id}: {e}")

            errors.append({
                "game_id": game_id,
                "error": str(e),
                "status": "failed"
            })

        time.sleep(SLEEP_SECONDS)

    success_df = pd.DataFrame(success)
    error_df = pd.DataFrame(errors)

    success_path = METADATA_DIR / f"pbp_download_success_{SEASON.replace('-', '_')}.csv"
    error_path = METADATA_DIR / f"pbp_download_errors_{SEASON.replace('-', '_')}.csv"

    success_df.to_csv(success_path, index=False)
    error_df.to_csv(error_path, index=False)

    print("\nDONE")
    print(f"Successful downloads: {len(success_df)}")
    print(f"Failed downloads: {len(error_df)}")
    print(f"Success log: {success_path}")
    print(f"Error log: {error_path}")


if __name__ == "__main__":
    main()