"""End-to-end stint build (cached PBP + live box score)."""

from __future__ import annotations

import os
import unittest


@unittest.skipUnless(
    os.environ.get("RUN_NBA_INTEGRATION") == "1",
    "Set RUN_NBA_INTEGRATION=1 to run live NBA Stats API tests.",
)
class TestLineupStintsIntegration(unittest.TestCase):
    def test_fixture_game_returns_table(self) -> None:
        from src.processing.lineup_stints import build_lineup_stints_for_game

        game_id = "0042500111"
        st, warnings = build_lineup_stints_for_game(game_id)
        self.assertIsInstance(warnings, list)
        self.assertGreater(len(st), 0)
        for col in (
            "game_id",
            "period",
            "duration_seconds",
            "home_player_ids",
            "away_player_ids",
            "ended_by_substitution",
        ):
            self.assertIn(col, st.columns)


if __name__ == "__main__":
    unittest.main()
