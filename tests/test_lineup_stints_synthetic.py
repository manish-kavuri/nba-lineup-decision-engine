"""Synthetic PBP → stint table (no API)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.processing.lineup_stints import build_lineup_stints


class TestLineupStintsSynthetic(unittest.TestCase):
    def test_two_subs_one_period(self) -> None:
        home_team_id = 10
        away_team_id = 20
        starters = {
            home_team_id: {1, 2, 3, 4, 5},
            away_team_id: {6, 7, 8, 9, 10},
        }
        players = pd.DataFrame(
            {
                "PLAYER_ID": [1, 2, 3, 4, 5, 11, 6, 7, 8, 9, 10, 12],
                "TEAM_ID": [home_team_id] * 6 + [away_team_id] * 6,
                "PLAYER_NAME": [
                    "Alpha",
                    "B",
                    "C",
                    "D",
                    "E",
                    "Ken Eleven",
                    "Foxtrot",
                    "G",
                    "H",
                    "I",
                    "J",
                    "Larry Twelve",
                ],
            }
        )
        pbp = pd.DataFrame(
            [
                {
                    "actionNumber": 1,
                    "period": 1,
                    "clock": "PT12M00.00S",
                    "teamId": 0,
                    "actionType": "period",
                    "subType": "start",
                    "description": "Start",
                    "scoreHome": "0",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 2,
                    "period": 1,
                    "clock": "PT11M00.00S",
                    "teamId": home_team_id,
                    "personId": 1,
                    "actionType": "Substitution",
                    "description": "SUB: Ken Eleven FOR Alpha",
                    "scoreHome": "0",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 3,
                    "period": 1,
                    "clock": "PT10M00.00S",
                    "teamId": away_team_id,
                    "personId": 6,
                    "actionType": "Substitution",
                    "description": "SUB: Larry Twelve FOR Foxtrot",
                    "scoreHome": "0",
                    "scoreAway": "0",
                },
            ]
        )

        st, warnings = build_lineup_stints(
            "GAME1",
            pbp,
            players,
            starters,
            home_team_id,
            away_team_id,
        )
        self.assertEqual(warnings, [])
        self.assertGreaterEqual(len(st), 1)
        last = st.iloc[-1]
        self.assertIn("11", last["home_player_ids"].split(","))
        self.assertNotIn("1", last["home_player_ids"].split(","))
        self.assertIn("12", last["away_player_ids"].split(","))
        self.assertNotIn("6", last["away_player_ids"].split(","))


if __name__ == "__main__":
    unittest.main()
