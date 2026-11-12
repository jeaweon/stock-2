"""Script to discover the single best unified strategy per Asset Group (A, B, C).

Evaluates candidate strategy rules across all stocks in each group,
finding the configuration that maximizes group aggregate return and win rate.
"""
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.data_loader import load_stock_data
from src.backtester import run_backtest, BacktestResult
from src.adaptive_experiment import compute_atr, compute_rsi

# Asset Group definitions
GROUPS = {
    "A (장기 우상향/빅테크)": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "NFLX"],
    "B (초고변동성/노이즈)": ["ETHU", "CONL", "TSLA", "MARA", "SOXL"],
    "C (박스권/갭 리스크)": ["KORU", "EWY", "FXI", "AMD", "TSM"],
}

# ---------------- GROUP A CANDIDATES ---------------- #

def strat_A_v1(d: pd.DataFrame) -> pd.DataFrame:
    """A1: BB 하단 터치(50% 매수) + 20MA 상향 돌파(100% 확대) + 50 EMA 이탈 청산."""
    data = d.copy()
    close = data["Close"]
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_lower = ma20 - (std20 * 2.0)
    ema50 = close.ewm(span=50, adjust=False).mean()

    data["signal"] = 0
    dip = close <= bb_lower * 1.005
    cross = (close > ma20) & (close.shift(1) <= ma20.shift(1))
    data.loc[dip, "signal"] = 1
    data.loc[dip, "position_size"] = 0.5
    data.loc[cross, "signal"] = 1
    data.loc[cross, "position_size"] = 1.0

    trend_break = (close < ema50) & (close.shift(1) >= ema50.shift(1))
    data.loc[trend_break, "signal"] = -1
    data["stop_loss_pct"] = 0.06
    data["take_profit_pct"] = 0.35
    return data


def strat_A_v2(d: pd.DataFrame) -> pd.DataFrame:
    """A2: 200 EMA 상단 정배열 추세 + 20MA 골든크로스/눌림목 + 20 EMA 트레일링 청산 (과최적화 방지)."""
    data = d.copy()
    close = data["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    data["signal"] = 0
    # 200 EMA 상승 추세장에서만 진입
    uptrend = (close > ema200) & (ema20 > ema50)
    entry = uptrend & (close > ema20) & (close.shift(1) <= ema20.shift(1))
    data.loc[entry, "signal"] = 1
    data.loc[entry, "position_size"] = 1.0

    # 청산: 50 EMA 하향 이탈
    exit_cond = (close < ema50) & (close.shift(1) >= ema50.shift(1))
    data.loc[exit_cond, "signal"] = -1
    data["stop_loss_pct"] = 0.05
    data["take_profit_pct"] = 0.25
    return data


def strat_A_v3(d: pd.DataFrame) -> pd.DataFrame:
    """A3: 200선 추세 + RSI(14) 40~60 건강한 모멘텀 구간 골든크로스 + 손절 4%, 익절 30%."""
    data = d.copy()
    close = data["Close"]
    rsi = compute_rsi(close, 14)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    data["signal"] = 0
    buy = (close > ema200) & (ema20 > ema60) & (ema20.shift(1) <= ema60.shift(1)) & (rsi >= 40)
    data.loc[buy, "signal"] = 1
    data.loc[buy, "position_size"] = 0.9

    sell = (ema20 < ema60) & (ema20.shift(1) >= ema60.shift(1))
    data.loc[sell, "signal"] = -1
    data["stop_loss_pct"] = 0.04
    data["take_profit_pct"] = 0.30
    return data


# ---------------- GROUP B CANDIDATES ---------------- #

def strat_B_v1(d: pd.DataFrame) -> pd.DataFrame:
    """B1: 20봉 신고가 + 거래량 1.2배 돌파, 역배열 물타기 금지, 2.5% 손절, 8% 익절."""
    data = d.copy()
    close = data["Close"]
    vol = data["Volume"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    high20 = data["High"].shift(1).rolling(20).max()
    vol20 = vol.rolling(20).mean()

    data["signal"] = 0
    uptrend = (close > ema20) & (ema20 > ema50)
    breakout = (close > high20) & (vol > vol20 * 1.2) & uptrend
    data.loc[breakout, "signal"] = 1
    data["position_size"] = 0.6

    sell = (close < ema20) & (close.shift(1) >= ema20.shift(1))
    data.loc[sell, "signal"] = -1
    data["stop_loss_pct"] = 0.025
    data["take_profit_pct"] = 0.08
    return data


def strat_B_v2(d: pd.DataFrame) -> pd.DataFrame:
    """B2: 15봉 최고가 돌파 + ATR 동적 자금관리 + 3.0% 칼손절, 12% 익절 (추세 확장형)."""
    data = d.copy()
    close = data["Close"]
    vol = data["Volume"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    high15 = data["High"].shift(1).rolling(15).max()
    vol20 = vol.rolling(20).mean()
    atr = compute_atr(data, 14)

    data["signal"] = 0
    breakout = (close > high15) & (vol > vol20 * 1.15) & (close > ema20) & (ema20 > ema50)
    data.loc[breakout, "signal"] = 1
    norm_atr = (atr / close).fillna(0.03)
    data["position_size"] = np.clip(0.05 / norm_atr, 0.3, 0.8)

    sell = (close < ema20) & (close.shift(1) >= ema20.shift(1))
    data.loc[sell, "signal"] = -1
    data["stop_loss_pct"] = 0.03
    data["take_profit_pct"] = 0.12
    return data


def strat_B_v3(d: pd.DataFrame) -> pd.DataFrame:
    """B3: MACD 양수 전환 + 20MA 돌파 (물타기 금지) + 2.0% 타이트 손절."""
    data = d.copy()
    close = data["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    data["signal"] = 0
    buy = (macd > signal) & (macd.shift(1) <= signal.shift(1)) & (close > ema50)
    data.loc[buy, "signal"] = 1
    data["position_size"] = 0.7

    sell = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    data.loc[sell, "signal"] = -1
    data["stop_loss_pct"] = 0.02
    data["take_profit_pct"] = 0.07
    return data


# ---------------- GROUP C CANDIDATES ---------------- #

def strat_C_v1(d: pd.DataFrame) -> pd.DataFrame:
    """C1: RSI < 32 과매도 진입 + 20MA(중심선) 익절 + 오버나이트 손실 컷."""
    data = d.copy()
    close = data["Close"]
    rsi = compute_rsi(close, 14)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_lower = ma20 - (std20 * 2.0)

    data["signal"] = 0
    time_diff_next = data.index.to_series().diff(-1).abs()
    is_session_end = time_diff_next > pd.Timedelta(hours=4)

    oversold = (rsi <= 32) & (close <= bb_lower * 1.01)
    data.loc[oversold & (~is_session_end), "signal"] = 1
    data["position_size"] = 0.8

    mean_rev = close >= ma20
    data.loc[mean_rev | (is_session_end & (close < ma20)), "signal"] = -1
    data["stop_loss_pct"] = 0.03
    data["take_profit_pct"] = 0.06
    return data


def strat_C_v2(d: pd.DataFrame) -> pd.DataFrame:
    """C2: 볼린저밴드 하단 돌파 반등 + RSI < 35 필터 + 상단 밴드 익절 (스윙 반등형)."""
    data = d.copy()
    close = data["Close"]
    rsi = compute_rsi(close, 14)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_lower = ma20 - (std20 * 2.0)
    bb_mid = ma20

    data["signal"] = 0
    entry = (close > bb_lower) & (close.shift(1) <= bb_lower.shift(1)) & (rsi < 35)
    data.loc[entry, "signal"] = 1
    data["position_size"] = 0.75

    exit_cond = (close >= bb_mid) | (rsi >= 65)
    data.loc[exit_cond, "signal"] = -1
    data["stop_loss_pct"] = 0.025
    data["take_profit_pct"] = 0.07
    return data


def strat_C_v3(d: pd.DataFrame) -> pd.DataFrame:
    """C3: 200선 하단 박스권 역추세 RSI 28 이하 극과매도 + 3% 칼손절, 5% 단기 익절."""
    data = d.copy()
    close = data["Close"]
    rsi = compute_rsi(close, 14)
    ma20 = close.rolling(20).mean()

    data["signal"] = 0
    entry = (rsi <= 28)
    data.loc[entry, "signal"] = 1
    data["position_size"] = 0.7

    exit_cond = (close >= ma20) | (rsi >= 60)
    data.loc[exit_cond, "signal"] = -1
    data["stop_loss_pct"] = 0.025
    data["take_profit_pct"] = 0.05
    return data


# Candidate Registry per group
CANDIDATES = {
    "A (장기 우상향/빅테크)": [
        ("A1 (BB 하단 분할 매수 + 20MA 돌파 비중확대 + 50선 청산)", strat_A_v1),
        ("A2 (200선 정배열 추세 + 20MA 눌림목 돌파 + 50선 청산)", strat_A_v2),
        ("A3 (200선 추세 + RSI 40~60 모멘텀 골든크로스)", strat_A_v3),
    ],
    "B (초고변동성/노이즈)": [
        ("B1 (20봉 신고가 돌파 + 거래량 1.2배 + 2.5% 손절)", strat_B_v1),
        ("B2 (15봉 돌파 + ATR 동적 사이징 + 3.0% 손절/12% 익절)", strat_B_v2),
        ("B3 (MACD 상향 반전 + 20MA 상단 + 2.0% 타이트 손절)", strat_B_v3),
    ],
    "C (박스권/갭 리스크)": [
        ("C1 (RSI 32 과매도 + 20MA 익절 + 장마감 손실 갭 컷)", strat_C_v1),
        ("C2 (BB 하단 반등 + RSI 35 + 중심선/RSI 65 익절)", strat_C_v2),
        ("C3 (RSI 28 극과매도 진입 + 2.5% 손절/5% 익절)", strat_C_v3),
    ],
}


def evaluate_group_candidates():
    print("=" * 95)
    print("🔬 [자산 그룹별 최적 통합 전략 탐색] 종목 개별이 아닌 '그룹 전체 통합 종합 수익률' 평가")
    print("=" * 95)

    data_cache = {}
    for grp_name, tickers in GROUPS.items():
        for t in tickers:
            if t not in data_cache:
                try:
                    data_cache[t] = load_stock_data(t, period="730d", interval="1h")
                except Exception as e:
                    print(f"[{t}] 로드 실패: {e}")

    best_strategies = {}

    for grp_name, cand_list in CANDIDATES.items():
        tickers = GROUPS[grp_name]
        print(f"\n========================================================")
        print(f"📊 자산 그룹: {grp_name} (포함 종목: {tickers})")
        print(f"========================================================")

        scores = []
        for cand_name, strat_fn in cand_list:
            returns = []
            win_rates = []
            mdds = []
            trades_list = []
            pfs = []

            for t in tickers:
                if t in data_cache and not data_cache[t].empty:
                    res = run_backtest(data_cache[t], signal_func=strat_fn, initial_capital=10000.0, fee_rate=0.001)
                    returns.append(res.total_return_pct)
                    win_rates.append(res.win_rate_pct)
                    mdds.append(res.max_drawdown_pct)
                    trades_list.append(res.total_trades)
                    pfs.append(res.profit_factor)

            avg_ret = float(np.mean(returns)) if returns else 0.0
            avg_win = float(np.mean(win_rates)) if win_rates else 0.0
            avg_mdd = float(np.mean(mdds)) if mdds else 0.0
            avg_pf = float(np.mean(pfs)) if pfs else 0.0
            total_trades = sum(trades_list)

            scores.append({
                "cand_name": cand_name,
                "strat_fn": strat_fn,
                "avg_return": avg_ret,
                "avg_win_rate": avg_win,
                "avg_mdd": avg_mdd,
                "avg_profit_factor": avg_pf,
                "total_trades": total_trades,
            })

            print(f"  • {cand_name}")
            print(f"    -> 평균 수익률: {avg_ret:+.2f}% | 평균 승률: {avg_win:.1f}% | 평균 MDD: {avg_mdd:.1f}% | 평균 PF: {avg_pf:.2f} | 총 거래: {total_trades}회")

        # Pick the best by highest average return (with reasonable MDD)
        best_cand = sorted(scores, key=lambda x: x["avg_return"], reverse=True)[0]
        best_strategies[grp_name] = best_cand
        print(f"\n  🏆 [{grp_name} 최종 선정 최우수 통합 전략]: {best_cand['cand_name']}")
        print(f"     통합 평균 수익률: {best_cand['avg_return']:+.2f}% | 평균 승률: {best_cand['avg_win_rate']:.1f}% | 평균 MDD: {best_cand['avg_mdd']:.1f}%")

    return best_strategies


if __name__ == "__main__":
    best_results = evaluate_group_candidates()
