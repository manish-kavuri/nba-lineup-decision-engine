"""Unit tests for period clock parsing."""

from __future__ import annotations

import unittest

from src.processing.clock import parse_clock_seconds_remaining, period_length_seconds


class TestClock(unittest.TestCase):
    def test_parse_regulation(self) -> None:
        self.assertAlmostEqual(parse_clock_seconds_remaining("PT12M00.00S"), 720.0)
        self.assertAlmostEqual(parse_clock_seconds_remaining("PT08M43.00S"), 523.0)
        self.assertAlmostEqual(parse_clock_seconds_remaining("PT00M00.00S"), 0.0)

    def test_period_length(self) -> None:
        self.assertEqual(period_length_seconds(1), 720.0)
        self.assertEqual(period_length_seconds(4), 720.0)
        self.assertEqual(period_length_seconds(5), 300.0)


if __name__ == "__main__":
    unittest.main()
