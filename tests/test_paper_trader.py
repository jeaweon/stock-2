"""Unit tests for Paper Trading (모의 투자) Engine."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.paper_trader import (
    get_default_portfolio,
    load_portfolio,
    save_portfolio,
    reset_portfolio,
    execute_paper_buy,
    execute_paper_sell,
    update_market_prices,
    evaluate_live_champion_signals,
    scan_and_rebalance_portfolio,
)


class TestPaperTrader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.portfolio_file = Path(self.temp_dir) / "test_portfolio.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_portfolio_and_persistence(self):
        port = get_default_portfolio(initial_cash=20000.0, fee_rate=0.001)
        self.assertEqual(port["cash"], 20000.0)
        self.assertEqual(port["initial_capital"], 20000.0)
        self.assertEqual(port["fee_rate"], 0.001)
        self.assertEqual(len(port["positions"]), 0)

        save_portfolio(port, filepath=self.portfolio_file)
        self.assertTrue(self.portfolio_file.exists())

        loaded = load_portfolio(filepath=self.portfolio_file)
        self.assertEqual(loaded["cash"], 20000.0)
        self.assertEqual(loaded["initial_capital"], 20000.0)

    def test_execute_paper_buy_success(self):
        port = get_default_portfolio(initial_cash=10000.0, fee_rate=0.001)
        # Allocate $2000 to buy AAPL @ $100
        ok, msg = execute_paper_buy(port, ticker="AAPL", price=100.0, allocation_amount=2000.0)
        self.assertTrue(ok)
        self.assertIn("AAPL", port["positions"])
        
        pos = port["positions"]["AAPL"]
        # $2000 / (100 * 1.001) = 19.98 -> 19 shares
        self.assertEqual(pos["shares"], 19)
        self.assertEqual(pos["entry_price"], 100.0)
        # Trade cost = 1900, fee = 1.90, total = 1901.90
        self.assertAlmostEqual(port["cash"], 10000.0 - 1901.90, places=2)
        # Stop loss price = 100 * 0.88 = 88.0
        self.assertEqual(pos["stop_loss_price"], 88.0)

    def test_execute_paper_buy_rejections(self):
        port = get_default_portfolio(initial_cash=100.0, fee_rate=0.001)
        # 1. Price too high for available cash
        ok, msg = execute_paper_buy(port, "NVDA", price=500.0, allocation_amount=2000.0)
        self.assertFalse(ok)
        self.assertIn("자금이 부족합니다", msg)

        # 2. Buy valid position first
        ok, msg = execute_paper_buy(port, "AAPL", price=10.0, allocation_amount=50.0)
        self.assertTrue(ok)
        # Duplicate buy
        ok, msg = execute_paper_buy(port, "AAPL", price=10.0, allocation_amount=20.0)
        self.assertFalse(ok)
        self.assertIn("이미 보유 중인 종목", msg)

        # 3. Invalid price
        ok, msg = execute_paper_buy(port, "MSFT", price=0.0, allocation_amount=50.0)
        self.assertFalse(ok)

    def test_execute_paper_sell_success(self):
        port = get_default_portfolio(initial_cash=10000.0, fee_rate=0.001)
        execute_paper_buy(port, "AAPL", price=100.0, allocation_amount=2000.0)
        pos = port["positions"]["AAPL"]
        shares = pos["shares"] # 19 shares, entry cost 1900, entry fee 1.90

        # Sell AAPL at $120 (+20%)
        ok, msg = execute_paper_sell(port, "AAPL", price=120.0, reason="Take Profit Test")
        self.assertTrue(ok)
        self.assertNotIn("AAPL", port["positions"])
        self.assertEqual(len(port["trade_history"]), 1)

        th = port["trade_history"][0]
        self.assertEqual(th["ticker"], "AAPL")
        self.assertEqual(th["shares"], shares)
        self.assertEqual(th["exit_price"], 120.0)
        self.assertEqual(th["pnl_pct"], 20.0)
        # Gross = 19 * 120 = 2280, exit fee = 2.28, net proceeds = 2277.72
        # Realized PnL = 2280 - 1900 - (1.90 + 2.28) = 375.82
        self.assertAlmostEqual(th["net_pnl"], 375.82, places=2)
        self.assertAlmostEqual(port["cash"], (10000.0 - 1901.90) + 2277.72, places=2)

    def test_execute_paper_sell_non_held(self):
        port = get_default_portfolio(initial_cash=10000.0)
        ok, msg = execute_paper_sell(port, "TSLA", price=200.0)
        self.assertFalse(ok)
        self.assertIn("보유하고 있지 않은", msg)

    def test_update_market_prices(self):
        port = get_default_portfolio(initial_cash=10000.0, fee_rate=0.001)
        execute_paper_buy(port, "AAPL", price=100.0, allocation_amount=2000.0) # 19 shares

        # Price moves to $110
        update_market_prices(port, {"AAPL": 110.0})
        pos = port["positions"]["AAPL"]
        self.assertEqual(pos["current_price"], 110.0)
        self.assertEqual(pos["unrealized_pnl_pct"], 10.0)
        self.assertGreater(pos["unrealized_pnl"], 0.0)

    def test_scan_and_rebalance_hard_stop(self):
        port = get_default_portfolio(initial_cash=10000.0, fee_rate=0.001)
        execute_paper_buy(port, "AAPL", price=100.0, allocation_amount=2000.0) # Stop is at 88.0

        # Simulate price dropping to $85 (< $88)
        dates = pd.date_range("2024-01-01", periods=250, freq="h")
        df_aapl = pd.DataFrame({
            "Open": [85.0] * 250,
            "High": [86.0] * 250,
            "Low": [84.0] * 250,
            "Close": [85.0] * 250,
            "Volume": [1000] * 250
        }, index=dates)

        res = scan_and_rebalance_portfolio(port, {"AAPL": df_aapl}, max_positions=5, auto_trade=True)
        self.assertEqual(len(res["sell_executions"]), 1)
        self.assertIn("12% 와이드 하드스탑", res["sell_executions"][0])
        self.assertNotIn("AAPL", port["positions"])
        self.assertEqual(len(port["trade_history"]), 1)


if __name__ == "__main__":
    unittest.main()
