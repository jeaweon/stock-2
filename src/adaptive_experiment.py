"""Regime-Adaptive Asset Classification & Multi-Strategy Experimentation Module.

Analyzes 2-year 1h data (trend shape, ATR volatility, overnight gap frequency),
classifies assets into 3 distinct regimes (A: Trend, B: Breakout/Anti-Martingale, C: Mean Reversion/Gap Control),
and evaluates performance with cross-testing and objective critique.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data_loader import load_stock_data
from src.backtester import run_backtest, BacktestResult
from src.storage import save_history_record
from src.timeframes import ema_days, ema_hours


@dataclass
class AssetProfile:
    ticker: str
    total_return_pct: float
    cagr_pct: float
    mean_normalized_atr_pct: float
    overnight_gap_freq_pct: float  # Frequency of gaps > 1.0%
    mean_overnight_gap_pct: float
    above_ema200_ratio: float
    assigned_group: str
    group_name: str
    rationale: str


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def profile_asset(df: pd.DataFrame, ticker: str) -> AssetProfile:
    """Analyze 2-year 1h data (trend shape, ATR volatility, overnight gap frequency)

    and classify into Group A, Group B, or Group C.
    """
    sym = ticker.upper().strip()
    close = df["Close"]
    total_bars = len(df)

    # 1. 2-Year Total Return & Trend
    total_ret = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100.0
    approx_years = total_bars / (252.0 * 6.5)
    cagr = (
        ((close.iloc[-1] / close.iloc[0]) ** (1.0 / max(0.1, approx_years)) - 1.0) * 100.0
        if close.iloc[-1] > 0
        else total_ret
    )

    # 2. Daily/Hourly Volatility (Normalized ATR %)
    atr = compute_atr(df, period=14)
    norm_atr_pct = float((atr / close * 100.0).mean())

    # 3. Overnight Gap Analysis
    time_diff = df.index.to_series().diff()
    is_overnight = time_diff > pd.Timedelta(hours=4)

    overnight_gap = (df["Open"] - df["Close"].shift(1)).abs() / df["Close"].shift(1) * 100.0
    valid_gaps = overnight_gap[is_overnight].dropna()
    mean_gap_pct = float(valid_gaps.mean()) if not valid_gaps.empty else 0.0
    gap_freq_pct = float((valid_gaps > 1.0).mean() * 100.0) if not valid_gaps.empty else 0.0

    # 4. Moving Averages for Trend Persistence
    ema_200h = ema_hours(close, 200)  # 200시간봉 EMA (약 30.8거래일)
    above_ema200_ratio = float((close > ema_200h).mean())

    # Pure Dynamic Statistical Classification (No hardcoded tickers)
    if norm_atr_pct >= 1.8 or (total_ret < -15.0 and norm_atr_pct >= 1.2):
        group = "B"  # Ultra-high volatility / noise / severe drawdown
    elif (gap_freq_pct >= 45.0 and mean_gap_pct >= 0.9) or (total_ret < 25.0 and above_ema200_ratio < 0.50):
        group = "C"  # Box-range / mean reversion & gap risk
    elif total_ret >= 25.0 and above_ema200_ratio >= 0.50:
        group = "A"  # Long-term uptrend & trend persistence
    elif norm_atr_pct >= 1.3:
        group = "B"
    else:
        group = "C"

    group_meta = {
        "A": (
            "그룹 A: 장기 우상향 및 빅테크 자산",
            "글로벌 자본 집중 & 강한 추세 지속성. 하락은 일시적 눌림목이므로 모아가고 상승 시 비중 확대",
        ),
        "B": (
            "그룹 B: 초고변동성 및 노이즈 자산",
            "하락장 바닥 없는 추락 위험. 역배열 물타기 절대 금지, 거래량 동반 돌파 시에만 진입 & 칼손절",
        ),
        "C": (
            "그룹 C: 박스권 횡보 및 갭 리스크 자산",
            "박스권 가두리 장세 & 높은 시차 갭 발생 빈도. RSI 과매도 역추세 진입 & 장중 익절, 오버나이트 갭 통제",
        ),
    }

    group_name, rationale = group_meta[group]

    return AssetProfile(
        ticker=sym,
        total_return_pct=round(total_ret, 2),
        cagr_pct=round(cagr, 2),
        mean_normalized_atr_pct=round(norm_atr_pct, 2),
        overnight_gap_freq_pct=round(gap_freq_pct, 1),
        mean_overnight_gap_pct=round(mean_gap_pct, 2),
        above_ema200_ratio=round(above_ema200_ratio * 100, 1),
        assigned_group=group,
        group_name=group_name,
        rationale=rationale,
    )


# ---------------- 3 Strategies ---------------- #

def generate_strategy_A_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy A (장기 우상향/성장주 - 거시 추세 라이딩 & 무제한 홀딩형):

    [기존 조건 전면 폐기 사유]
    - 15개 이상 조사 시 B&H 평균 미달로 영구 폐기됨.
    - 1시간봉 단기 이평선(20시간봉 EMA) 청산과 잦은 손절은 2년 대세 상승장에서
      수십 회의 휩쏘와 수수료를 유발하고 랠리 초기에 하차하여 B&H에 패배함.

    [새로운 매수/매도 규칙]
    - 매수: 20일 EMA(130봉) > 50일 EMA(325봉) 거시 정배열 상태에서 주가가 200시간봉 EMA 위에 있을 때 100% 진입.
    - 홀딩: 단기 잔파동(일일/시간봉 노이즈)에 흔들리지 않고 거시 상승 추세를 무제한 라이딩.
    - 매도: 20일 EMA가 50일 EMA를 하향 이탈(거시 데드크로스)하거나 200시간봉 EMA 5% 하회 시 전량 청산.
    - 손절: 12% 하드 스탑으로 단기 흔들기 휩쏘 원천 차단.
    """
    d = df.copy()
    close = d["Close"]
    ema_20day_130h = ema_days(close, 20)   # 20일 거시 EMA = 130시간봉
    ema_50day_325h = ema_days(close, 50)   # 50일 거시 EMA = 325시간봉
    ema_200h = ema_hours(close, 200)       # 200시간봉 EMA (약 30.8거래일)

    d["signal"] = 0
    macro_uptrend = (ema_20day_130h > ema_50day_325h) & (close > ema_200h)
    d.loc[macro_uptrend, "signal"] = 1
    d["position_size"] = 1.0

    macro_exit = (ema_20day_130h <= ema_50day_325h) | (close < ema_200h * 0.95)
    d.loc[macro_exit, "signal"] = -1
    d["stop_loss_pct"] = 0.12  # 12% 하드 스탑
    return d


def generate_strategy_B_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy B (초고변동성/레버리지 - 슈퍼서핑 & 폭락 차단형):

    [기존 조건 전면 폐기 사유]
    - 15개 이상 조사 시 B&H 평균 미달로 영구 폐기됨.
    - 2.5% 칼손절과 20시간봉 EMA 이탈은 초고변동성 자산(SOXL, ETHU, KORU 등)의 자연스러운 
      장중 변동성(ATR 3~5%)을 견디지 못하고 계좌를 갉아먹음.

    [새로운 매수/매도 규칙]
    - 매수: 20일 EMA(130봉) > 100일 EMA(650봉) 골든크로스 상태에서 직전 20시간봉 고가 돌파 시 진입.
    - 홀딩: 고정 익절 없이 슈퍼 랠리를 끝까지 홀딩.
    - 매도: 20일 EMA가 100일 EMA를 하향 이탈(데드크로스)할 때 즉시 청산 (크립토/레버리지 폭락장 100% 현금 회피).
    - 손절: 8% 손절로 폭락장 갭하락 방어.
    """
    d = df.copy()
    close = d["Close"]
    high = d["High"]

    ema_20day_130h = ema_days(close, 20)     # 20일 거시 EMA = 130시간봉
    ema_100day_650h = ema_days(close, 100)   # 100일 거시 EMA = 650시간봉
    high_20h = high.shift(1).rolling(20).max()  # 직전 20시간봉 최고가

    d["signal"] = 0
    super_trend = (ema_20day_130h > ema_100day_650h) & (close > high_20h)
    d.loc[super_trend, "signal"] = 1
    d["position_size"] = 1.0

    crash_exit = (ema_20day_130h <= ema_100day_650h)
    d.loc[crash_exit, "signal"] = -1
    d["stop_loss_pct"] = 0.08  # 8% 손절
    return d


def generate_strategy_C_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy C (박스권/사이클/가치주 - 거시 추세 필터 & 하락장 전면 방어형):

    [기존 조건 전면 폐기 사유]
    - 15개 이상 조사 시 B&H 평균 미달로 영구 폐기됨.
    - RSI 역추세 진입 + 7% 쥐꼬리 익절 + 3% 잦은 손절은 하락 종목의 하락 파동을 막지 못함.

    [새로운 매수/매도 규칙]
    - 매수: 20일 EMA(130봉) > 50일 EMA(325봉) 거시 골든크로스 상태에서 주가가 50일선 위에 안착 시 진입.
    - 홀딩: 장기 반등 추세 유지 시 지속 보유.
    - 매도: 20일 EMA가 50일 EMA를 하향 이탈하거나 50일선을 3% 이상 붕괴 시 전량 청산 (하락장 -50% 폭락 방어).
    - 손절: 6% 손절.
    """
    d = df.copy()
    close = d["Close"]

    ema_20day_130h = ema_days(close, 20)   # 20일 거시 EMA = 130시간봉
    ema_50day_325h = ema_days(close, 50)   # 50일 거시 EMA = 325시간봉

    d["signal"] = 0
    entry_cond = (ema_20day_130h > ema_50day_325h) & (close > ema_50day_325h)
    d.loc[entry_cond, "signal"] = 1
    d["position_size"] = 1.0

    exit_cond = (ema_20day_130h <= ema_50day_325h) | (close < ema_50day_325h * 0.97)
    d.loc[exit_cond, "signal"] = -1
    d["stop_loss_pct"] = 0.06  # 6% 손절
    return d


STRATEGY_MAP: Dict[str, Tuple[Callable[[pd.DataFrame], pd.DataFrame], str, str]] = {
    "A": (
        generate_strategy_A_signals,
        "전략 A (거시 추세 라이딩 & 무제한 홀딩형)",
        "20일/50일 EMA 거시 정배열 100% 진입, 무제한 라이딩, 거시 데드크로스 청산, 12% 하드스탑",
    ),
    "B": (
        generate_strategy_B_signals,
        "전략 B (슈퍼서핑 & 폭락 차단형)",
        "20일/100일 EMA 골든크로스 + 20봉 돌파 진입, 데드크로스 청산, 8% 손절, 레버리지 폭락 방어",
    ),
    "C": (
        generate_strategy_C_signals,
        "전략 C (거시 추세 필터 & 하락장 전면 방어형)",
        "20일/50일 EMA 골든크로스 진입, 데드크로스/50일선 붕괴 청산, 6% 손절, 사이클 하락 방어",
    ),
}


def run_single_asset_experiment(
    ticker: str,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict[str, Any]:
    """Load asset, profile it into Group A/B/C, and run matched & unmatched cross-testing."""
    df = load_stock_data(ticker=ticker, period="730d", interval="1h")
    profile = profile_asset(df, ticker)

    # 1. Run Assigned (Matched) Strategy
    matched_strat_fn, matched_name, matched_desc = STRATEGY_MAP[profile.assigned_group]
    matched_res = run_backtest(df, signal_func=matched_strat_fn, initial_capital=initial_capital, fee_rate=fee_rate)

    # 2. Run Cross-Testing with all 3 strategies for comparison
    all_strategy_results = {}
    for grp_key, (strat_fn, s_name, _) in STRATEGY_MAP.items():
        if grp_key == profile.assigned_group:
            all_strategy_results[grp_key] = (s_name, matched_res, True)
        else:
            cross_res = run_backtest(df, signal_func=strat_fn, initial_capital=initial_capital, fee_rate=fee_rate)
            all_strategy_results[grp_key] = (s_name, cross_res, False)

    # 3. Buy & Hold benchmark calculation
    bh_ret = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100.0

    # Save matched strategy result to history CSV
    save_history_record(
        ticker=ticker,
        iteration_round=200 + ord(profile.assigned_group),
        strategy_name=f"[맞춤 전략 배정] {matched_name}",
        buy_rules=f"{profile.group_name} 맞춤 매수: {matched_desc}",
        sell_rules="맞춤 청산 및 리스크 관리",
        position_sizing_rules="동적 자산별 차등 비중",
        win_rate_pct=matched_res.win_rate_pct,
        total_return_pct=matched_res.total_return_pct,
        mdd_pct=matched_res.max_drawdown_pct,
        total_trades=matched_res.total_trades,
        profit_factor=matched_res.profit_factor,
        sharpe_ratio=matched_res.sharpe_ratio,
        critic_verdict="PASS" if matched_res.total_return_pct > 0 and matched_res.max_drawdown_pct < 35 else "EVALUATED",
        critic_feedback=f"자산 프로파일링: {profile.rationale} (2Y수익: {profile.total_return_pct}%, ATR: {profile.mean_normalized_atr_pct}%, 갭빈도: {profile.overnight_gap_freq_pct}%)",
    )

    return {
        "ticker": ticker,
        "profile": profile,
        "df": df,
        "matched_result": matched_res,
        "matched_name": matched_name,
        "all_strategy_results": all_strategy_results,
        "buy_and_hold_return_pct": round(bh_ret, 2),
    }


def run_multi_asset_experiment_suite(
    tickers: List[str],
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict[str, Any]:
    """Run full experiment across multiple random/selected assets and compile comparative results."""
    results = {}
    summary_rows = []

    for t in tickers:
        try:
            res = run_single_asset_experiment(t, initial_capital=initial_capital, fee_rate=fee_rate)
            results[t] = res
            prof = res["profile"]
            m_res = res["matched_result"]
            summary_rows.append({
                "ticker": t,
                "group": prof.assigned_group,
                "group_name": prof.group_name,
                "2y_buy_hold_ret": res["buy_and_hold_return_pct"],
                "matched_strategy": res["matched_name"],
                "strategy_return": m_res.total_return_pct,
                "win_rate": m_res.win_rate_pct,
                "mdd": m_res.max_drawdown_pct,
                "trades": m_res.total_trades,
                "profit_factor": m_res.profit_factor,
                "sharpe_ratio": m_res.sharpe_ratio,
                "alpha_vs_hold": round(m_res.total_return_pct - res["buy_and_hold_return_pct"], 2),
                "atr_pct": prof.mean_normalized_atr_pct,
                "gap_freq_pct": prof.overnight_gap_freq_pct,
            })
        except Exception as e:
            print(f"Error testing {t}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    return {
        "results": results,
        "summary_df": summary_df,
    }
