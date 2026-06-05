import argparse
import traceback
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path("/Users/manishkavuri/Desktop/nba-lineup-decision-engine")
sys.path.append(str(PROJECT_ROOT))

from src.processing.lineup_stints_v2 import build_lineup_stints_for_game


RAW_PBP_DIR = PROJECT_ROOT / "data/raw/pbp"
BY_GAME_DIR = PROJECT_ROOT / "data/processed/lineup_stints_by_game_v2"
FINAL_DIR = PROJECT_ROOT / "data/processed/final"
METADATA_DIR = PROJECT_ROOT / "data/metadata"

BY_GAME_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_downloaded_game_ids(limit=None):
    metadata_path = METADATA_DIR / "nba_games_2024_25_Regular_Season.csv"

    games_df = pd.read_csv(metadata_path)

    game_ids = (
        games_df["GAME_ID"]
        .astype(str)
        .str.zfill(10)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if limit is not None:
        game_ids = game_ids[:limit]

    return game_ids


def save_game_outputs(game_id, stints_df, quality_df, debug_df):
    game_dir = BY_GAME_DIR / f"game_id={game_id}"
    game_dir.mkdir(parents=True, exist_ok=True)

    stints_path = game_dir / "stints.parquet"
    quality_path = game_dir / "quality.parquet"
    debug_path = game_dir / "period_start_debug.parquet"

    stints_df.to_parquet(stints_path, index=False)
    quality_df.to_parquet(quality_path, index=False)
    debug_df.to_parquet(debug_path, index=False)

    return stints_path, quality_path, debug_path


def build_season(limit=None, combine=True):
    game_ids = get_downloaded_game_ids(limit=limit)

    print(f"Found {len(game_ids)} games to process.")

    all_quality = []
    errors = []
    successful_stint_paths = []

    for i, game_id in enumerate(game_ids, start=1):
        print(f"\n[{i}/{len(game_ids)}] Processing {game_id}")

        try:
            stints_df, quality_df, debug_df = build_lineup_stints_for_game(game_id)

            quality_row = quality_df.iloc[0].to_dict()
            all_quality.append(quality_row)

            bad_game = (
                quality_row.get("negative_durations", 0) > 0
                or quality_row.get("bad_home_lineup_size", 0) > 0
                or quality_row.get("bad_away_lineup_size", 0) > 0
                or quality_row.get("boundary_issues", 0) > 0
            )

            if bad_game:
                print("Validation warning:")
                print(quality_df.to_string(index=False))

            stints_path, quality_path, debug_path = save_game_outputs(
                game_id,
                stints_df,
                quality_df,
                debug_df
            )

            successful_stint_paths.append(stints_path)

            print(f"Saved stints: {stints_path}")
            print(quality_df.to_string(index=False))

        except Exception as e:
            print(f"FAILED {game_id}: {e}")

            errors.append({
                "game_id": game_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    quality_summary_df = pd.DataFrame(all_quality)
    error_df = pd.DataFrame(errors)

    quality_summary_path = METADATA_DIR / "lineup_stints_quality_v2_2024_25.csv"
    error_path = METADATA_DIR / "lineup_stints_errors_v2_2024_25.csv"

    quality_summary_df.to_csv(quality_summary_path, index=False)
    error_df.to_csv(error_path, index=False)

    print("\nFinished processing.")
    print(f"Successful games: {len(successful_stint_paths)}")
    print(f"Failed games: {len(error_df)}")
    print(f"Quality summary: {quality_summary_path}")
    print(f"Error log: {error_path}")

    if combine and successful_stint_paths:
        print("\nCombining game-level stint files...")

        dfs = [pd.read_parquet(path) for path in successful_stint_paths]
        final_df = pd.concat(dfs, ignore_index=True)

        final_path = FINAL_DIR / "fact_lineup_stints_v2_2024_25.parquet"
        sample_path = FINAL_DIR / "fact_lineup_stints_v2_2024_25_sample.csv"

        final_df.to_parquet(final_path, index=False)
        final_df.head(1000).to_csv(sample_path, index=False)

        print(f"Final rows: {len(final_df)}")
        print(f"Unique games: {final_df['game_id'].nunique()}")
        print(f"Final dataset saved to: {final_path}")
        print(f"Sample CSV saved to: {sample_path}")

    return quality_summary_df, error_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of games to process. Use 1, 5, or 50 for testing."
    )

    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Do not combine all game-level files into final dataset."
    )

    args = parser.parse_args()

    build_season(
        limit=args.limit,
        combine=not args.no_combine
    )