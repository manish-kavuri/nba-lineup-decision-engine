"""Name resolution with accented characters."""

from __future__ import annotations

import unittest

import pandas as pd

from src.processing.substitutions import resolve_name_to_player_id


class TestSubstitutionsUnicode(unittest.TestCase):
    def test_vucevic_ascii_vs_roster(self) -> None:
        team_id = 1610612738
        players = pd.DataFrame(
            {
                "PLAYER_ID": [202696],
                "TEAM_ID": [team_id],
                "PLAYER_NAME": ["Nikola Vučević"],
            }
        )
        pid = resolve_name_to_player_id("Vucevic", team_id, players)
        self.assertEqual(pid, 202696)


if __name__ == "__main__":
    unittest.main()
