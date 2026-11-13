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
    get_default_krw_portfolio,
    load_krw_portfolio,
    execute_krw_buy,
    execute_krw_sell,
    update_krw_market_prices,
    run_live_scan_and_trade,
    compute_journal_metrics,
    update_trade_memo,
)
from src.data_loader import _extract_ticker_df_from_batch


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

        res = scan_and_rebalance_portfolio(port, {"AAPL": df_aapl}, max_positions=5, auto_trade=True, filepath=self.portfolio_file)
        self.assertEqual(len(res["sell_executions"]), 1)
        self.assertIn("12% 와이드 하드스탑", res["sell_executions"][0])
        self.assertNotIn("AAPL", port["positions"])
        self.assertEqual(len(port["trade_history"]), 1)


def _make_bullish_1h_df(base_price: float, atr_amp: float = 0.02, periods: int = 260) -> pd.DataFrame:
    """Create synthetic 1h OHLCV data satisfying Champion DNA 4-step buy conditions."""
    dates = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    closes = np.linspace(base_price * 0.72, base_price, periods)
    opens = closes * (1.0 - atr_amp * 0.3)
    highs = closes * (1.0 + atr_amp * 0.75)
    lows = closes * (1.0 - atr_amp * 0.75)
    # Keep latest close very near recent 60-bar high
    closes[-1] = highs[-1] * 0.995
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": np.full(periods, 100000),
    }, index=dates)


class TestKrwPaperTraderAndJournal(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.portfolio_file = Path(self.temp_dir) / "krw_portfolio.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_krw_default_10m_and_integer_buy_sell_journal(self):
        port = get_default_krw_portfolio(initial_capital_krw=10_000_000.0, fee_rate=0.001, max_positions=20, fx_rate=1400.0)
        self.assertEqual(port["initial_capital_krw"], 10_000_000.0)
        self.assertEqual(port["cash_krw"], 10_000_000.0)
        self.assertEqual(port["max_positions"], 20)

        # 500,000 KRW slot budget at $100 * 1400 * 1.001 = 140,140 KRW/share -> 3 shares
        ok, msg = execute_krw_buy(
            port,
            ticker="NVDA",
            price_usd=100.0,
            fx_rate=1400.0,
            allocation_krw=500_000.0,
            reason="🏆 1시간봉 돌파 테스트",
            indicators={"atr_pct": 2.1, "alpha_score": 90.0},
        )
        self.assertTrue(ok, msg)
        pos = port["positions"]["NVDA"]
        self.assertEqual(pos["shares"], 3)
        self.assertEqual(pos["entry_cost_krw"], 420_000.0)
        self.assertEqual(pos["entry_fee_krw"], 420.0)
        self.assertEqual(port["cash_krw"], 10_000_000.0 - 420_420.0)

        # Update price to $120 -> unrealized gain > 0
        update_krw_market_prices(port, {"NVDA": 120.0}, fx_rate=1400.0)
        self.assertGreater(port["positions"]["NVDA"]["unrealized_pnl_krw"], 0.0)

        # Sell at $120
        ok_sell, _ = execute_krw_sell(
            port,
            ticker="NVDA",
            price_usd=120.0,
            fx_rate=1400.0,
            reason="목표 수익 청산",
            exit_indicators={"atr_pct": 1.8},
        )
        self.assertTrue(ok_sell)
        self.assertEqual(len(port["positions"]), 0)
        self.assertEqual(len(port["trade_history"]), 1)

        rec = port["trade_history"][0]
        # Entry reason & indicators must be preserved in the journal!
        self.assertEqual(rec["entry_reason"], "🏆 1시간봉 돌파 테스트")
        self.assertEqual(rec["entry_indicators"]["atr_pct"], 2.1)
        self.assertEqual(rec["exit_reason"], "목표 수익 청산")
        self.assertGreater(rec["net_pnl_krw"], 80_000.0)

        # Reflection memo update & metrics
        self.assertTrue(update_trade_memo(port, rec["trade_id"], "추세 완벽 추종"))
        self.assertEqual(port["trade_history"][0]["memo"], "추세 완벽 추종")

        jm = compute_journal_metrics(port)
        self.assertEqual(jm["total_trades"], 1)
        self.assertEqual(jm["win_rate_pct"], 100.0)
        self.assertGreater(jm["total_realized_krw"], 80_000.0)

        save_portfolio(port, filepath=self.portfolio_file)
        reloaded = load_krw_portfolio(filepath=self.portfolio_file)
        self.assertEqual(reloaded["trade_history"][0]["memo"], "추세 완벽 추종")

    def test_krw_over_budget_skip_and_next_candidate_buy(self):
        port = get_default_krw_portfolio(initial_capital_krw=10_000_000.0, fee_rate=0.001, max_positions=20, fx_rate=1400.0)
        # HIGHPX ($500 ~= 700k KRW > 500k slot) vs AFFORD ($100 ~= 140k KRW <= 500k slot)
        df_high = _make_bullish_1h_df(base_price=500.0, atr_amp=0.025)
        df_afford = _make_bullish_1h_df(base_price=100.0, atr_amp=0.020)

        res = run_live_scan_and_trade(
            port,
            universe_data={"HIGHPX": df_high, "AFFORD": df_afford},
            fx_rate=1400.0,
            max_positions=20,
            auto_trade=True,
            filepath=self.portfolio_file,
        )
        self.assertNotIn("HIGHPX", port["positions"])
        self.assertIn("AFFORD", port["positions"])
        self.assertEqual(len(res["skipped_over_budget"]), 1)
        self.assertEqual(res["skipped_over_budget"][0]["ticker"], "HIGHPX")
        self.assertEqual(len(res["buy_executions"]), 1)
        self.assertEqual(port["positions"]["AFFORD"]["shares"], 3)

    def test_krw_hard_stop_gap_down_and_journal_preservation(self):
        port = get_default_krw_portfolio(initial_capital_krw=10_000_000.0, fee_rate=0.001, max_positions=20, fx_rate=1400.0)
        execute_krw_buy(
            port, "PLTR", price_usd=100.0, fx_rate=1400.0, allocation_krw=500_000.0,
            reason="신고가 돌파 진입", indicators={"atr_pct": 2.5},
        )
        # Gap down open at $82 (< stop loss $88)
        dates = pd.date_range("2026-02-01", periods=220, freq="h", tz="UTC")
        df_gap = pd.DataFrame({
            "Open": [95.0] * 219 + [82.0],
            "High": [96.0] * 219 + [84.0],
            "Low": [94.0] * 219 + [81.0],
            "Close": [95.0] * 219 + [83.0],
            "Volume": [50000] * 220,
        }, index=dates)

        res = run_live_scan_and_trade(port, {"PLTR": df_gap}, fx_rate=1400.0, max_positions=20, auto_trade=True, filepath=self.portfolio_file)
        self.assertEqual(len(res["sell_executions"]), 1)
        self.assertNotIn("PLTR", port["positions"])
        self.assertEqual(len(port["trade_history"]), 1)
        rec = port["trade_history"][0]
        self.assertEqual(rec["exit_price_usd"], 82.0)  # Filled at gap-down Open price
        self.assertEqual(rec["entry_reason"], "신고가 돌파 진입")
        self.assertIn("12% 하드스탑", rec["exit_reason"])

    def test_extract_ticker_df_from_batch_formats(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        cols = pd.MultiIndex.from_product([["AAPL", "NVDA"], ["Open", "High", "Low", "Close", "Volume"]])
        df_multi = pd.DataFrame(np.ones((3, 10)) * 100.0, index=idx, columns=cols)
        extracted = _extract_ticker_df_from_batch(df_multi, "NVDA", is_single_ticker=False)
        self.assertIsNotNone(extracted)
        self.assertEqual(list(extracted.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(len(extracted), 3)


if __name__ == "__main__":
    unittest.main()
