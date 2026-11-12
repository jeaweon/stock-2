"""Unit tests for history storage module."""
import unittest
from pathlib import Path
import pandas as pd
from src.storage import save_history_record, load_history_records, clear_history

TEST_CSV = Path("tests/test_history.csv")


class TestStorage(unittest.TestCase):
    def setUp(self):
        clear_history(TEST_CSV)

    def tearDown(self):
        clear_history(TEST_CSV)

    def test_save_and_load_records(self):
        save_history_record(
            ticker="AAPL",
            iteration_round=0,
            strategy_name="테스트 원본 전략",
            buy_rules="이평선 골든크로스",
            sell_rules="이평선 데드크로스",
            position_sizing_rules="전액 매수",
            win_rate_pct=60.5,
            total_return_pct=15.2,
            mdd_pct=8.4,
            total_trades=12,
            profit_factor=1.85,
            sharpe_ratio=1.42,
            critic_verdict="PASS",
            critic_feedback="양호한 성과",
            strategy_code="def generate_signals(df): pass",
            file_path=TEST_CSV,
        )

        df = load_history_records(TEST_CSV)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ticker"], "AAPL")
        self.assertEqual(df.iloc[0]["win_rate_pct"], 60.5)
        self.assertEqual(df.iloc[0]["critic_verdict"], "PASS")

        # Save second record (append)
        save_history_record(
            ticker="AAPL",
            iteration_round=1,
            strategy_name="테스트 1차 개선",
            buy_rules="RSI 필터 추가",
            sell_rules="데드크로스",
            position_sizing_rules="50% 비중",
            win_rate_pct=72.0,
            total_return_pct=24.5,
            mdd_pct=6.1,
            total_trades=15,
            profit_factor=2.30,
            sharpe_ratio=1.88,
            critic_verdict="PASS",
            critic_feedback="우수한 개선",
            strategy_code="def generate_signals(df): pass",
            file_path=TEST_CSV,
        )

        df2 = load_history_records(TEST_CSV)
        self.assertEqual(len(df2), 2)
        self.assertEqual(df2.iloc[1]["iteration_round"], 1)


if __name__ == "__main__":
    unittest.main()
