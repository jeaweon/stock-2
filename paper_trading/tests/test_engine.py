"""Comprehensive Unit Tests for Independent 1-Hour Live Paper Trading Engine."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Allow importing paper_trading/engine.py cleanly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (
    SimulatedTrader,
    apply_extended_hours_price,
    convert_usd_to_krw_1h,
    evaluate_champion_dna_1h,
    extract_ticker_from_batch,
    run_full_hourly_cycle,
)


def _make_1h_bullish_usd(
    price_usd: float,
    atr_amp: float = 0.025,
    periods: int = 260,
    overnight_gap: bool = True,
) -> pd.DataFrame:
    """Create 1h USD candle DataFrame passing all Champion DNA 4-step criteria."""
    # Use 30-min offsets like US regular market (:30)
    dates = pd.date_range("2026-01-05 14:30", periods=periods, freq="h", tz="UTC")
    closes = np.linspace(price_usd * 0.68, price_usd, periods)
    opens = closes * (1.0 - atr_amp * 0.25)
    if overnight_gap:
        # Inject >1% overnight-style jump every 7 bars to build institutional gap frequency
        for i in range(7, periods, 7):
            opens[i] = closes[i - 1] * 1.018
            closes[i] = max(closes[i], opens[i] * 1.005)
    highs = np.maximum(opens, closes) * (1.0 + atr_amp * 0.7)
    lows = np.minimum(opens, closes) * (1.0 - atr_amp * 0.7)
    closes[-1] = highs[-1] * 0.995
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(periods, 120000.0),
        },
        index=dates,
    )


def _make_fx_hourly(rate: float = 1400.0, periods: int = 300) -> pd.Series:
    # Use :00 top-of-hour timestamps like KRW=X
    idx = pd.date_range("2026-01-01 00:00", periods=periods, freq="h", tz="UTC")
    return pd.Series(np.full(periods, rate), index=idx, name="FX_Rate")


class TestIndependentPaperTradingEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.state_file = str(Path(self.tmp_dir) / "trader_state.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_usd_to_krw_time_alignment_different_minutes(self):
        df_usd = _make_1h_bullish_usd(price_usd=100.0)
        fx = _make_fx_hourly(rate=1400.0, periods=360)
        df_krw = convert_usd_to_krw_1h(df_usd, fx)
        self.assertFalse(df_krw["Close"].isna().any())
        self.assertFalse(df_krw["FX_Rate"].isna().any())
        # Last USD price (~100) * 1400 ~= 140,000 KRW
        self.assertGreater(df_krw["Close"].iloc[-1], 130_000.0)
        self.assertIn("EMA_10D", df_krw.columns)
        self.assertIn("EMA_30D", df_krw.columns)
        self.assertIn("EMA_200H", df_krw.columns)

    def test_initial_balance_10m_integer_buy_sell_journal_and_stats(self):
        trader = SimulatedTrader(
            initial_balance=10_000_000,
            save_file=self.state_file,
            max_positions=20,
            fee_rate=0.001,
        )
        self.assertEqual(trader.cash, 10_000_000.0)
        self.assertEqual(trader.max_positions, 20)

        # 500,000 KRW budget at 140,000 KRW/share * 1.001 = 140,140 -> 3 integer shares
        ok, msg = trader.buy_integer_shares(
            symbol="NVDA",
            price_krw=140_000.0,
            allocation_krw=500_000.0,
            strategy_name="유망3호 DNA (점수 95 / ATR 2.5%)",
            reason="🏆 유망3호+DNA 매수충족",
            bar_ts="2026-11-13 14:30",
        )
        self.assertTrue(ok, msg)
        pos = trader.positions["NVDA"]
        self.assertEqual(pos["qty"], 3)
        self.assertEqual(pos["cost_basis_krw"], 420_000.0)
        self.assertEqual(pos["entry_fee"], 420.0)
        self.assertEqual(trader.cash, 10_000_000.0 - 420_420.0)
        self.assertEqual(pos["stop_loss_price"], round(140_000.0 * 0.88, 0))

        # Sell at 168,000 KRW (+20%)
        ok_s, msg_s = trader.sell_all_shares(
            symbol="NVDA",
            price_krw=168_000.0,
            reason="📉 데드크로스 익절",
            bar_ts="2026-11-14 15:30",
        )
        self.assertTrue(ok_s, msg_s)
        self.assertNotIn("NVDA", trader.positions)
        self.assertEqual(len(trader.trade_logs), 2)

        buy_log, sell_log = trader.trade_logs
        self.assertEqual(buy_log["타입"], "매수")
        self.assertEqual(sell_log["타입"], "매도")
        self.assertEqual(sell_log["진입 전략"], "유망3호 DNA (점수 95 / ATR 2.5%)")
        self.assertEqual(sell_log["상세 사유"], "📉 데드크로스 익절")
        self.assertGreater(sell_log["실현손익(원)"], 80_000)
        self.assertGreater(sell_log["수익률(%)"], 19.0)

        perf = trader.get_performance_summary()
        self.assertEqual(perf["total_closed"], 1)
        self.assertEqual(perf["win_rate"], 100.0)
        self.assertEqual(perf["realized_profit_krw"], sell_log["실현손익(원)"])

        # Reload persistence verification
        reloaded = SimulatedTrader(initial_balance=10_000_000, save_file=self.state_file)
        self.assertEqual(len(reloaded.trade_logs), 2)
        self.assertEqual(reloaded.cash, trader.cash)

    def test_over_budget_skip_and_same_bar_duplicate_guard(self):
        trader = SimulatedTrader(initial_balance=10_000_000, save_file=self.state_file, max_positions=20)
        fx = _make_fx_hourly(rate=1400.0, periods=400)

        df_high_u = _make_1h_bullish_usd(price_usd=500.0, atr_amp=0.028)
        df_high_k = convert_usd_to_krw_1h(df_high_u, fx)
        ev_high = evaluate_champion_dna_1h(df_high_u, df_high_k, "HIGHPX")

        df_aff_u = _make_1h_bullish_usd(price_usd=100.0, atr_amp=0.022)
        df_aff_k = convert_usd_to_krw_1h(df_aff_u, fx)
        ev_aff = evaluate_champion_dna_1h(df_aff_u, df_aff_k, "AFFORD")

        self.assertTrue(ev_high.is_buy_signal)
        self.assertTrue(ev_aff.is_buy_signal)

        evals = {"HIGHPX": ev_high, "AFFORD": ev_aff}
        res1 = trader.run_api_scan_and_trade(evals, auto_trade=True)

        self.assertNotIn("HIGHPX", trader.positions)
        self.assertIn("AFFORD", trader.positions)
        self.assertEqual(len(res1["skipped_over_budget"]), 1)
        self.assertEqual(res1["skipped_over_budget"][0]["ticker"], "HIGHPX")
        self.assertEqual(trader.positions["AFFORD"]["qty"], 3)

        # Running scan again on the EXACT same 1h bar must not duplicate trades
        res2 = trader.run_api_scan_and_trade(evals, auto_trade=True)
        self.assertEqual(len(res2["buy_executions"]), 0)
        self.assertEqual(len(res2["sell_executions"]), 0)

    def test_hard_stop_gap_down_open_fill_and_no_same_bar_rebuy(self):
        trader = SimulatedTrader(initial_balance=10_000_000, save_file=self.state_file, max_positions=20)
        trader.buy_integer_shares(
            symbol="PLTR",
            price_krw=140_000.0,
            allocation_krw=500_000.0,
            strategy_name="유망3호 DNA",
            reason="신고가 돌파 매수",
            bar_ts="2026-01-10 14:30",
        )
        # Next bar opens gap-down at 112,000 (< stop 123,200)
        fx = _make_fx_hourly(rate=1400.0, periods=400)
        df_u = _make_1h_bullish_usd(price_usd=100.0)
        df_k = convert_usd_to_krw_1h(df_u, fx)
        df_k.iloc[-1, df_k.columns.get_loc("Open")] = 112_000.0
        df_k.iloc[-1, df_k.columns.get_loc("Close")] = 115_000.0

        ev = evaluate_champion_dna_1h(df_u, df_k, "PLTR")
        ev.open_krw = 112_000.0
        ev.price_krw = 115_000.0
        ev.bar_ts = "2026-01-15 14:30"

        res = trader.run_api_scan_and_trade({"PLTR": ev}, auto_trade=True)
        self.assertEqual(len(res["sell_executions"]), 1)
        self.assertIn("-12% 하드스탑", res["sell_executions"][0])
        self.assertNotIn("PLTR", trader.positions)
        # Filled at gap-down Open price (112,000), not Close (115,000)
        self.assertEqual(trader.trade_logs[-1]["가격(원)"], 112_000)
        # Did not immediately rebuy on the same bar
        self.assertEqual(len(res["buy_executions"]), 0)

    @patch("engine.requests.patch")
    @patch("engine.requests.get")
    def test_gist_cloud_load_save_and_check_mock(self, mock_get, mock_patch):
        fake_state = {
            "cash": 9_500_000.0,
            "initial_balance": 10_000_000.0,
            "max_positions": 20,
            "fee_rate": 0.001,
            "stop_loss_pct": 0.12,
            "positions": {},
            "trade_logs": [{"일자": "2026-01-01", "타입": "매수", "종목": "NVDA"}],
        }
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = {"files": {"trader_state.json": {"content": json.dumps(fake_state)}}}
        mock_get.return_value = get_resp

        patch_resp = MagicMock()
        patch_resp.status_code = 200
        mock_patch.return_value = patch_resp

        trader = SimulatedTrader(
            initial_balance=10_000_000,
            save_file=self.state_file,
            gist_id="fake_gist_id",
            github_token="fake_token",
        )
        self.assertEqual(trader.cash, 9_500_000.0)
        self.assertEqual(len(trader.trade_logs), 1)

        ok_check, msg_check = trader.check_gist_connection()
        self.assertTrue(ok_check)
        self.assertIn("완벽하게 설정", msg_check)

        trader.reset_account()
        self.assertTrue(mock_patch.called)
        self.assertEqual(trader.cash, 10_000_000.0)

    @patch("engine.yf.download")
    def test_extended_hours_overlay(self, mock_dl):
        df_u = _make_1h_bullish_usd(price_usd=100.0, periods=50)
        ext_idx = pd.date_range("2026-01-07 22:00", periods=3, freq="min", tz="UTC")
        mock_dl.return_value = pd.DataFrame(
            {"Open": [108.0, 109.0, 110.0], "High": [109.0, 110.0, 112.0], "Low": [107.0, 108.0, 109.0], "Close": [108.5, 109.5, 111.5], "Volume": [100, 100, 100]},
            index=ext_idx,
        )
        updated = apply_extended_hours_price("NVDA", df_u)
        self.assertEqual(float(updated["Close"].iloc[-1]), 111.5)
        self.assertGreaterEqual(float(updated["High"].iloc[-1]), 111.5)

    def test_extract_ticker_from_batch(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        cols = pd.MultiIndex.from_product([["AAPL", "NVDA"], ["Open", "High", "Low", "Close", "Volume"]])
        df_multi = pd.DataFrame(np.ones((3, 10)) * 50.0, index=idx, columns=cols)
        sub = extract_ticker_from_batch(df_multi, "NVDA", is_single_ticker=False)
        self.assertIsNotNone(sub)
        self.assertEqual(len(sub), 3)

    def test_legacy_stock_minus_gist_schema_normalization(self):
        # Legacy stock- schema lacks stop_loss_price, cost_basis_krw, and 실현손익(원)
        legacy_state = {
            "cash": 8_000_000,
            "positions": {
                "NVDA": {"qty": 10, "avg_price": 200_000, "entry_strategy": "단타"}
            },
            "trade_logs": [
                {"일자": "2026-10-01", "타입": "매도", "종목": "AMD", "가격": 150_000, "수량": 5, "수익률(%)": 5.2, "진입 전략": "스윙", "상세 사유": "익절"}
            ],
        }
        Path(self.state_file).write_text(json.dumps(legacy_state), encoding="utf-8")
        trader = SimulatedTrader(save_file=self.state_file)
        self.assertIn("NVDA", trader.positions)
        self.assertAlmostEqual(trader.positions["NVDA"]["stop_loss_price"], 176_000.0)
        self.assertEqual(trader.trade_logs[0]["실현손익(원)"], 0)
        perf = trader.get_performance_summary()
        self.assertEqual(perf["total_closed"], 1)

    @patch("engine.apply_extended_hours_price", side_effect=lambda sym, df: df)
    @patch("engine.fetch_live_1h_batch")
    @patch("engine.fetch_usd_krw_hourly")
    def test_run_full_hourly_cycle_executes_buy(self, mock_fx, mock_batch, _mock_ext):
        mock_fx.return_value = _make_fx_hourly(rate=1400.0, periods=360)
        df_bull = _make_1h_bullish_usd(price_usd=100.0, atr_amp=0.025)
        mock_batch.return_value = ({"NVDA": df_bull}, {})

        trader = SimulatedTrader(initial_balance=10_000_000, save_file=self.state_file)
        res = run_full_hourly_cycle(trader, scan_universe=["NVDA"], auto_trade=True)

        self.assertEqual(res["scanned_count"], 1)
        self.assertEqual(len(res["buy_executions"]), 1)
        self.assertIn("NVDA", trader.positions)


if __name__ == "__main__":
    unittest.main()
