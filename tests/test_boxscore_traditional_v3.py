"""Integration checks for BoxScoreTraditionalV3 ingestion (requires network)."""

from __future__ import annotations

import os
import unittest

# Fixture game used in the project notebook; known starter sets from API list order.
GAME_ID = "0042500111"
EXPECTED_BOS = {1627759, 1628369, 1628401, 1629674, 1630573}
EXPECTED_PHI = {202331, 1626162, 1630178, 1641737, 1642845}
TEAM_BOS = 1610612738
TEAM_PHI = 1610612755


@unittest.skipUnless(
    os.environ.get("RUN_NBA_INTEGRATION") == "1",
    "Set RUN_NBA_INTEGRATION=1 to run live NBA Stats API tests.",
)
class TestBoxScoreTraditionalV3(unittest.TestCase):
    def test_opening_lineups(self) -> None:
        from src.ingestion.boxscore_traditional import opening_lineups_by_team_id_traditional_v3

        lineups = opening_lineups_by_team_id_traditional_v3(GAME_ID)
        self.assertEqual(lineups[TEAM_BOS], EXPECTED_BOS)
        self.assertEqual(lineups[TEAM_PHI], EXPECTED_PHI)

    def test_load_player_stats_columns(self) -> None:
        from src.ingestion.boxscore_traditional import load_player_stats_traditional_v3

        df = load_player_stats_traditional_v3(GAME_ID)
        self.assertGreater(len(df), 0)
        for col in ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "personId", "teamId"):
            self.assertIn(col, df.columns)
        self.assertEqual(df["PLAYER_ID"].dtype, "int64")
        # Every roster row should map id to name
        row = df.loc[df["PLAYER_ID"] == 1627759].iloc[0]
        self.assertIn("Brown", row["PLAYER_NAME"])


if __name__ == "__main__":
    unittest.main()
