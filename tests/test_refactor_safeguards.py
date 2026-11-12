"""Unit tests for the 6-point refactor (15-stock rule, EMA terminology, sizing formula,
overnight gap-down fills, critic anti-overfitting rules, yfinance rate-limit defense)."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import data_loader
from src.backtester import run_backtest, _compute_session_open_flags
from src.gemini_agents import DualGeminiAgents, MIN_UNIVERSE_SIZE, compile_sizing_code
from src.timeframes import SPAN_20DAY, SPAN_50DAY, SPAN_100DAY, day_to_hour_bars, ema_days, ema_hours


def _two_session_df(gap_open: float) -> pd.DataFrame:
    """Day 1: 7 hourly bars @100. Day 2 opens at `gap_open` (overnight gap)."""
    day1 = pd.date_range("2024-03-04 09:30", periods=7, freq="1h", tz="America/New_York")
    day2 = pd.date_range("2024-03-05 09:30", periods=7, freq="1h", tz="America/New_York")
    idx = day1.append(day2).tz_convert("UTC")
    opens = [100.0] * 7 + [gap_open] + [gap_open] * 6
    closes = [100.0] * 7 + [gap_open] * 7
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) + 0.1 for o, c in zip(opens, closes)],
            "Low": [min(o, c) - 0.1 for o, c in zip(opens, closes)],
            "Close": closes,
            "Volume": [1000] * 14,
        },
        index=idx,
    )


def _buy_first_bar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["signal"] = 0
    d.iloc[0, d.columns.get_loc("signal")] = 1
    d["position_size"] = 1.0
    d["stop_loss_pct"] = 0.05  # stop @ 95
    return d


class TestTimeframes(unittest.TestCase):
    def test_day_to_hour_conversion(self):
        self.assertEqual(day_to_hour_bars(20), 130)
        self.assertEqual((SPAN_20DAY, SPAN_50DAY, SPAN_100DAY), (130, 325, 650))

    def test_day_vs_hour_ema_differs(self):
        close = pd.Series(np.linspace(100, 200, 400))
        pd.testing.assert_series_equal(ema_days(close, 20), close.ewm(span=130, adjust=False).mean())
        pd.testing.assert_series_equal(ema_hours(close, 20), close.ewm(span=20, adjust=False).mean())
        self.assertFalse(np.allclose(ema_days(close, 20), ema_hours(close, 20)))


class TestGapDownFill(unittest.TestCase):
    def test_session_open_flags(self):
        flags = _compute_session_open_flags(_two_session_df(100.0).index)
        self.assertEqual(int(flags.sum()), 1)
        self.assertTrue(flags[7])

    def test_gap_down_fills_at_open_not_stop(self):
        df = _two_session_df(gap_open=90.0)  # opens 10% lower, below 95 stop
        res = run_backtest(df, signal_func=_buy_first_bar, initial_capital=10000.0, fee_rate=0.0)
        t = res.trades[0]
        self.assertIn("갭하락", t.exit_reason)
        self.assertAlmostEqual(t.exit_price, 90.0, places=6)
        self.assertAlmostEqual(t.pnl_pct, -10.0, places=4)

    def test_gap_down_slippage_applied(self):
        df = _two_session_df(gap_open=90.0)
        res = run_backtest(df, signal_func=_buy_first_bar, fee_rate=0.0, slippage_rate=0.01)
        self.assertAlmostEqual(res.trades[0].exit_price, 89.1, places=6)

    def test_no_gap_uses_stop_price(self):
        df = _two_session_df(gap_open=97.0)  # opens above stop; intraday low later hits 95
        df.iloc[9, df.columns.get_loc("Low")] = 94.0
        res = run_backtest(df, signal_func=_buy_first_bar, fee_rate=0.0)
        t = res.trades[0]
        self.assertNotIn("갭하락", t.exit_reason)
        self.assertAlmostEqual(t.exit_price, 95.0, places=6)


class TestSizingFormula(unittest.TestCase):
    def test_sizing_func_evaluated_once_locally(self):
        df = _two_session_df(100.0)
        calls = {"n": 0}

        def sizing(d):
            calls["n"] += 1
            return pd.Series(0.4, index=d.index)

        res = run_backtest(df, signal_func=_buy_first_bar, fee_rate=0.0, position_sizing_func=sizing)
        self.assertEqual(calls["n"], 1)  # no per-bar calls
        self.assertAlmostEqual(res.trades[0].cost, 4000.0, places=4)

    def test_compile_sizing_code(self):
        code = (
            "import pandas as pd\n"
            "def compute_position_size(df):\n"
            "    return pd.Series(0.7, index=df.index)\n"
        )
        self.assertTrue(callable(compile_sizing_code(code)))
        self.assertIsNone(compile_sizing_code("def generate_signals(df):\n    return df\n"))

    def test_mock_optimizer_returns_sizing_func(self):
        opt = DualGeminiAgents(api_key=None).propose_improved_strategy("x", {}, None, 1)
        self.assertTrue(callable(opt["sizing_func"]))


class TestCriticUniverseRule(unittest.TestCase):
    def test_min_universe_is_15(self):
        self.assertEqual(MIN_UNIVERSE_SIZE, 15)

    def test_fail_when_below_bh_on_15_plus(self):
        critic = DualGeminiAgents(api_key=None).evaluate_with_adversarial_critic(
            "s", "code",
            {"total_return_pct": 50, "max_drawdown_pct": 10, "total_trades": 30,
             "universe_count": 16, "universe_avg_return_pct": 20.0, "universe_bh_avg_return_pct": 40.0},
            {"total_return_pct": 0},
        )
        self.assertEqual(critic["verdict"], "FAIL")


class TestRateLimitDefense(unittest.TestCase):
    def test_retry_with_backoff_then_success(self):
        sleeps = []
        attempts = {"n": 0}
        good = pd.DataFrame({"Close": [1.0]})

        def fetch():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("429 Too Many Requests")
            return good

        out = data_loader._fetch_with_retry(fetch, max_retries=3, sleep_fn=sleeps.append)
        self.assertFalse(out.empty)
        self.assertEqual(attempts["n"], 3)
        self.assertIn(data_loader.BACKOFF_BASE_SEC, sleeps)

    def test_batch_loader_uses_cache_and_pauses_between_batches(self):
        sleeps = []
        cached_df = pd.DataFrame({"Close": [1.0]})
        with mock.patch.object(data_loader, "_read_cache", side_effect=lambda p: cached_df if "AAA" in str(p) else None), \
             mock.patch.object(data_loader, "load_stock_data", return_value=cached_df) as m_load:
            loaded, failed = data_loader.load_many_stock_data(
                ["AAA", "B1", "B2", "B3"], batch_size=2, batch_pause_sec=7.0, sleep_fn=sleeps.append
            )
        self.assertEqual(set(loaded), {"AAA", "B1", "B2", "B3"})
        self.assertEqual(m_load.call_count, 3)  # cache hit skipped
        self.assertEqual(sleeps, [7.0])        # one pause between 2 batches
        self.assertEqual(failed, {})


if __name__ == "__main__":
    unittest.main()
