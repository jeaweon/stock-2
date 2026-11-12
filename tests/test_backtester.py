"""Unit tests for backtesting engine."""
import unittest
import numpy as np
import pandas as pd
from src.backtester import run_backtest, BacktestResult


class TestBacktester(unittest.TestCase):
    def setUp(self):
        # Create 100 bars of deterministic price data (rising then falling)
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        prices = [100.0 + (i * 0.5) if i < 50 else 125.0 - ((i - 50) * 1.5) for i in range(100)]
        self.df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 0.2 for p in prices],
                "Low": [p - 0.2 for p in prices],
                "Close": prices,
                "Volume": [1000] * 100,
            },
            index=dates,
        )

    def test_single_profitable_trade(self):
        # Strategy: Buy at bar 10, Sell at bar 40 (price rises from ~105 to ~120)
        def strat(df):
            d = df.copy()
            d["signal"] = 0
            d.iloc[10, d.columns.get_loc("signal")] = 1
            d.iloc[40, d.columns.get_loc("signal")] = -1
            d["position_size"] = 1.0
            return d

        res = run_backtest(self.df, signal_func=strat, initial_capital=10000.0, fee_rate=0.001)
        self.assertEqual(res.total_trades, 1)
        self.assertEqual(res.winning_trades, 1)
        self.assertEqual(res.win_rate_pct, 100.0)
        self.assertGreater(res.total_return_pct, 10.0)
        self.assertGreater(res.final_equity, 10000.0)

    def test_stop_loss_trigger(self):
        # Strategy: Buy at bar 45 (near peak ~122.5), stop loss 2%
        def strat(df):
            d = df.copy()
            d["signal"] = 0
            d.iloc[45, d.columns.get_loc("signal")] = 1
            d["position_size"] = 1.0
            d["stop_loss_pct"] = 0.02
            return d

        res = run_backtest(self.df, signal_func=strat, initial_capital=10000.0)
        self.assertGreater(res.total_trades, 0)
        first_trade = res.trades[0]
        self.assertIn("손절", first_trade.exit_reason)
        self.assertLess(first_trade.pnl_pct, 0.0)

    def test_dynamic_position_sizing(self):
        # Strategy: Buy at 50% sizing
        def strat(df):
            d = df.copy()
            d["signal"] = 0
            d.iloc[5, d.columns.get_loc("signal")] = 1
            d.iloc[20, d.columns.get_loc("signal")] = -1
            d["position_size"] = 0.5  # 50%
            return d

        res = run_backtest(self.df, signal_func=strat, initial_capital=10000.0)
        self.assertEqual(len(res.trades), 1)
        self.assertAlmostEqual(res.trades[0].cost, 5000.0, delta=100.0)


if __name__ == "__main__":
    unittest.main()
