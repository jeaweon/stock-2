"""Independent 1-Hour Live Paper Trading Engine (KRW 10,000,000 Base).

Completely standalone module modeled after the execution & Gist persistence style
of https://github.com/jeaweon/stock- while running:
- Stock Selection: 4-Step Champion DNA 1h Screener (ATR, 200h EMA, 10/30d EMA, Gap Freq, High Proximity)
- Trading Strategy: Promising Strategy #3 (10d > 30d Golden Cross + >200h + ATR>=1.1% + 60-bar High -6%,
  exit on 10d <= 30d Dead Cross or -12% Wide Hard Stop with overnight gap-open fill, 0.1% one-way fee)
- Money Management: Default 10,000,000 KRW, max 20 equal-weight integer-share slots, live FX conversion
  plus extended-hours (pre/post-market) 1m price injection on the latest 1-hour candle.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import yfinance as yf


HOURS_PER_TRADING_DAY: float = 6.5
DEFAULT_INITIAL_BALANCE_KRW: int = 10_000_000
DEFAULT_FEE_RATE: float = 0.001
DEFAULT_MAX_POSITIONS: int = 20
DEFAULT_STOP_LOSS_PCT: float = 0.12
DEFAULT_FX_FALLBACK: float = 1400.0
MIN_CASH_ORDER_KRW: float = 10_000.0

DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "trader_state.json"


# Curated Liquid US Equities & ETFs for API Screener Scanning (157 tickers)
LIVE_SCAN_UNIVERSE_157: List[str] = [
    # Mega Tech & Cloud / AI / Platforms
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "PLTR", "CRM", "ORCL",
    "ADBE", "INTU", "NOW", "IBM", "CSCO", "FTNT", "DDOG", "NET", "SNOW", "PANW",
    "CRWD", "ZS", "MDB", "WDAY", "APP", "RBLX", "SPOT", "UBER", "ABNB", "DASH",
    # Semiconductors & Data Center Hardware
    "AMD", "AVGO", "TSM", "MU", "QCOM", "INTC", "ARM", "SMCI", "MRVL", "LRCX",
    "AMAT", "ADI", "NXPI", "MCHP", "ON", "STM", "TER", "WDC", "STX", "SWKS",
    "QRVO", "ANET", "DELL", "HPQ", "VRT", "ALAB", "SOXL",
    # Nuclear / Clean Energy / Space / Quantum / High-Beta Growth
    "OKLO", "SMR", "NNE", "BE", "CEG", "TEM", "IONQ", "RGTI", "QBTS", "RKLB",
    "ASTS", "LUNR", "JOBY", "ACHR", "FSLR", "ENPH", "CCJ", "UEC", "NXE",
    # Fintech / Crypto Proxies / Financials
    "HOOD", "SOFI", "COIN", "MARA", "RIOT", "CLSK", "MSTR", "BITO", "PYPL", "UPST",
    "AFRM", "TOST", "NU", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "V", "MA",
    # Healthcare / Consumer / Mobility / Asia ADR
    "MRNA", "LLY", "PFE", "JNJ", "ABBV", "GILD", "BMY", "MRK", "CVS", "BSX",
    "EW", "DXCM", "PODD", "NKE", "SBUX", "TGT", "WMT", "DIS", "GM", "F",
    "RIVN", "LCID", "CCL", "RCL", "DAL", "UAL", "LUV", "GRAB", "SE", "BABA", "PDD",
    # Industrials / Defense / Energy / Materials / Core ETFs
    "GE", "BA", "LMT", "RTX", "CAT", "DE", "ETN", "EMR", "PWR", "XOM",
    "CVX", "OXY", "SLB", "HAL", "KMI", "FCX", "NEM", "AA", "CLF",
    "SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "TLT", "GLD", "SLV", "EWY",
]

AFFORDABLE_20SLOT_UNIVERSE: List[str] = [
    "NVDA", "PLTR", "AMD", "MU", "INTC", "ARM", "SMCI", "MRVL", "WDC", "STX",
    "ON", "STM", "TER", "SWKS", "QRVO", "HPQ", "DELL", "VRT", "ALAB", "SOXL",
    "OKLO", "SMR", "NNE", "BE", "TEM", "IONQ", "RGTI", "QBTS", "RKLB", "ASTS",
    "LUNR", "JOBY", "ACHR", "FSLR", "ENPH", "CCJ", "UEC", "NXE", "DDOG", "NET",
    "SNOW", "ZS", "MDB", "RBLX", "UBER", "ABNB", "CRM", "ORCL", "FTNT", "CSCO",
    "HOOD", "SOFI", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PYPL", "UPST", "AFRM",
    "TOST", "NU", "BAC", "WFC", "C", "SCHW", "MRNA", "PFE", "BMY", "GILD",
    "CVS", "BSX", "EW", "DXCM", "NKE", "SBUX", "TGT", "WMT", "DIS", "GM",
    "F", "RIVN", "LCID", "CCL", "DAL", "UAL", "LUV", "GRAB", "SE", "BABA",
    "PDD", "XOM", "CVX", "OXY", "SLB", "HAL", "KMI", "FCX", "NEM", "AA",
    "CLF", "TQQQ", "SQQQ", "IWM", "TLT", "SLV", "EWY",
]

HIGH_GROWTH_THEME_UNIVERSE: List[str] = [
    "NVDA", "PLTR", "APP", "ALAB", "VRT", "OKLO", "SMR", "NNE", "BE", "TEM",
    "IONQ", "RGTI", "QBTS", "RKLB", "ASTS", "LUNR", "JOBY", "ACHR", "HOOD", "SOFI",
    "COIN", "MSTR", "MARA", "CLSK", "MU", "AMD", "AVGO", "ARM", "MRVL", "LRCX",
    "AMAT", "TSLA", "SNOW", "NET", "CRWD", "DDOG", "RBLX", "TOST", "SOXL", "TQQQ",
]

SCAN_UNIVERSES: Dict[str, List[str]] = {
    "🌐 전체 대량 API 검색 풀 (157종목: S&P/나스닥/고성장 통합)": LIVE_SCAN_UNIVERSE_157,
    "💵 20분할 정수매수 최적화 풀 (107종목: 소액 분산 적합주)": AFFORDABLE_20SLOT_UNIVERSE,
    "🚀 AI · 반도체 · 원전 · 우주 고성장 테마 (40종목)": HIGH_GROWTH_THEME_UNIVERSE,
}


# ---------------------------------------------------------------------------
# Helpers: Timeframes, Indicators & Data Normalization
# ---------------------------------------------------------------------------

def normalize_tickers(tickers: Iterable[str]) -> List[str]:
    """Deduplicate and uppercase ticker symbols cleanly."""
    out: List[str] = []
    seen = set()
    for t in tickers:
        sym = str(t).upper().strip()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def day_to_hour_bars(days: float) -> int:
    """Convert trading days (6.5h/day) to 1h candle count (e.g., 10d -> 65, 30d -> 195)."""
    return int(round(days * HOURS_PER_TRADING_DAY))


def ema_days(close: pd.Series, days: float) -> pd.Series:
    return close.ewm(span=day_to_hour_bars(days), adjust=False).mean()


def ema_hours(close: pd.Series, hours: int) -> pd.Series:
    return close.ewm(span=int(hours), adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def standardize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance OHLCV columns, UTC DatetimeIndex, and ascending order."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [col[0] for col in out.columns]

    col_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = out.rename(columns={c: col_map[str(c).lower()] for c in out.columns if str(c).lower() in col_map})

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"필수 컬럼({col})이 누락되었습니다: {out.columns.tolist()}")

    out = out[required].dropna(subset=["Close"]).ffill()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "Datetime"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def extract_ticker_from_batch(
    raw_df: pd.DataFrame, sym: str, is_single_ticker: bool
) -> Optional[pd.DataFrame]:
    """Safely extract single symbol OHLCV from yf.download group_by='ticker' output."""
    if raw_df is None or raw_df.empty:
        return None
    if isinstance(raw_df.columns, pd.MultiIndex):
        lvl0 = raw_df.columns.get_level_values(0)
        lvl1 = raw_df.columns.get_level_values(1)
        if sym in lvl0:
            sub = raw_df[sym]
        elif sym in lvl1:
            sub = raw_df.xs(sym, axis=1, level=1)
        else:
            return None
    elif is_single_ticker:
        sub = raw_df
    else:
        return None

    try:
        std = standardize_ohlcv_df(sub)
        return std if not std.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live Market Data Fetchers (FX, Indices, 1h Batch + Pre/Post 1m Overlay)
# ---------------------------------------------------------------------------

def fetch_usd_krw_hourly(period: str = "100d", fallback: float = DEFAULT_FX_FALLBACK) -> pd.Series:
    """Fetch 1-hour historical USD/KRW exchange rate series (UTC-indexed)."""
    try:
        fx_df = yf.Ticker("KRW=X").history(period=period, interval="1h")
        if fx_df is None or fx_df.empty:
            fx_df = yf.Ticker("KRW=X").history(period=period, interval="1d")
        if fx_df is not None and not fx_df.empty and "Close" in fx_df.columns:
            s = fx_df["Close"].dropna().copy()
            s.index = pd.to_datetime(s.index, utc=True)
            s = s[(s >= 800.0) & (s <= 3000.0)].sort_index()
            if not s.empty:
                return s
    except Exception:
        pass
    now_idx = pd.DatetimeIndex([datetime.now(timezone.utc)], name="Datetime")
    return pd.Series([float(fallback)], index=now_idx, name="FX_Rate")


def fetch_market_indices() -> Dict[str, Dict[str, float]]:
    """Fetch representative market indices matching the stock- reference dashboard."""
    tickers = {
        "나스닥": "^IXIC",
        "S&P500": "^GSPC",
        "코스피": "^KS11",
        "이더리움": "ETH-USD",
        "원/달러 환율": "KRW=X",
    }
    data: Dict[str, Dict[str, float]] = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if hist is not None and len(hist) >= 2:
                close = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                chg = ((close - prev) / prev) * 100.0 if prev != 0 else 0.0
                data[name] = {"price": close, "change": chg}
            else:
                data[name] = {"price": 0.0, "change": 0.0}
        except Exception:
            data[name] = {"price": 0.0, "change": 0.0}
    return data


def apply_extended_hours_price(ticker: str, df_usd: pd.DataFrame) -> pd.DataFrame:
    """Overlay latest pre/post/after-hours 1m execution price onto the latest 1h candle."""
    if df_usd is None or df_usd.empty:
        return df_usd
    sym = ticker.upper().strip()
    if sym.endswith("-USD"):
        return df_usd

    out = df_usd.copy()
    try:
        ext_df = yf.download(sym, period="1d", interval="1m", prepost=True, progress=False, auto_adjust=True)
        if ext_df is not None and not ext_df.empty:
            if isinstance(ext_df.columns, pd.MultiIndex):
                ext_df.columns = [col[0] for col in ext_df.columns]
            closes = ext_df["Close"].dropna()
            if not closes.empty:
                latest_ext = float(closes.iloc[-1])
                if latest_ext > 0:
                    last_idx = out.index[-1]
                    out.loc[last_idx, "Close"] = latest_ext
                    if latest_ext > float(out.loc[last_idx, "High"]):
                        out.loc[last_idx, "High"] = latest_ext
                    if latest_ext < float(out.loc[last_idx, "Low"]):
                        out.loc[last_idx, "Low"] = latest_ext
    except Exception:
        pass
    return out


def fetch_live_1h_batch(
    tickers: Iterable[str],
    period: str = "90d",
    chunk_size: int = 25,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Download fresh 1-hour USD candle data directly from Yahoo Finance API."""
    syms = normalize_tickers(tickers)
    total = len(syms)
    loaded: Dict[str, pd.DataFrame] = {}
    failed: Dict[str, str] = {}
    if total == 0:
        return loaded, failed

    step = max(1, chunk_size)
    done = 0
    for c_start in range(0, total, step):
        chunk = syms[c_start:c_start + step]
        if progress_callback:
            progress_callback(done, total, f"{chunk[0]} 등 {len(chunk)}종목 1시간봉 수신 중...")

        raw_chunk = pd.DataFrame()
        try:
            raw_chunk = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval="1h",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            raw_chunk = pd.DataFrame()

        missed: List[str] = []
        is_single = len(chunk) == 1
        for sym in chunk:
            sub = extract_ticker_from_batch(raw_chunk, sym, is_single)
            if sub is not None and not sub.empty:
                loaded[sym] = sub
            else:
                missed.append(sym)

        for sym in missed:
            try:
                single = yf.Ticker(sym).history(period=period, interval="1h", auto_adjust=True)
                if single is not None and not single.empty:
                    loaded[sym] = standardize_ohlcv_df(single)
                else:
                    failed[sym] = "API 1시간봉 데이터 없음"
            except Exception as e:
                failed[sym] = str(e)

        done += len(chunk)
        if progress_callback:
            progress_callback(done, total, f"API 수신 완료 ({done}/{total})")

    return loaded, failed


def convert_usd_to_krw_1h(df_usd: pd.DataFrame, fx_series: pd.Series) -> pd.DataFrame:
    """Convert a 1-hour USD DataFrame into KRW prices using time-aligned USD/KRW FX rates
    and compute 1h chart indicators (EMA 10d/30d/200h, ATR%).
    """
    if df_usd is None or df_usd.empty:
        return pd.DataFrame()

    df = df_usd.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    fx_clean = fx_series.dropna().copy()
    fx_clean.index = pd.to_datetime(fx_clean.index, utc=True)
    fx_df = fx_clean.sort_index().rename("FX_Rate").to_frame()

    # Align hourly stock timestamps (e.g. :30) with nearest prior hourly FX timestamps (:00)
    merged = pd.merge_asof(
        df.reset_index().rename(columns={df.index.name or "index": "Datetime"}),
        fx_df.reset_index().rename(columns={fx_df.index.name or "index": "Datetime"}),
        on="Datetime",
        direction="backward",
    )
    merged["FX_Rate"] = merged["FX_Rate"].ffill().bfill().fillna(DEFAULT_FX_FALLBACK)
    merged = merged.set_index("Datetime")

    for col in ["Open", "High", "Low", "Close"]:
        merged[col] = merged[col] * merged["FX_Rate"]

    merged["EMA_10D"] = ema_days(merged["Close"], 10)
    merged["EMA_30D"] = ema_days(merged["Close"], 30)
    merged["EMA_200H"] = ema_hours(merged["Close"], 200)
    merged["Vol_Avg_90"] = merged["Volume"].rolling(window=90, min_periods=1).mean()
    atr = compute_atr(merged, period=14)
    merged["ATR_Pct"] = (atr / merged["Close"].replace(0, np.nan)) * 100.0
    return merged


# ---------------------------------------------------------------------------
# Stock Selection (Champion DNA 4-Step Screener) & Signal (Promising Strategy #3)
# ---------------------------------------------------------------------------

@dataclass
class DnaEvaluation:
    ticker: str
    status: str  # 'BUY_READY', 'WATCHLIST', 'DISQUALIFIED'
    is_qualified: bool
    is_buy_signal: bool
    is_exit_signal: bool
    alpha_score: float
    atr_pct: float
    gap_freq_pct: float
    dist_from_high_pct: float
    ema200_slope_pct: float
    above_200h: bool
    golden_aligned: bool
    price_usd: float
    price_krw: float
    open_krw: float
    fx_rate: float
    ema_10d_krw: float
    ema_30d_krw: float
    ema_200h_krw: float
    bar_ts: str
    rationale: str


def evaluate_champion_dna_1h(
    df_usd: pd.DataFrame,
    df_krw: pd.DataFrame,
    ticker: str,
) -> DnaEvaluation:
    """Evaluate 4-step Champion DNA Screener & Promising Strategy #3 signals on 1-hour bars.

    Technical criteria are evaluated strictly on raw USD bars to prevent FX noise distortion;
    execution/display prices and EMAs are reported in KRW from df_krw.
    """
    sym = ticker.upper().strip()
    if df_usd is None or df_krw is None or len(df_usd) < 200 or len(df_krw) < 200:
        px_usd = float(df_usd["Close"].iloc[-1]) if df_usd is not None and not df_usd.empty else 0.0
        px_krw = float(df_krw["Close"].iloc[-1]) if df_krw is not None and not df_krw.empty else 0.0
        op_krw = float(df_krw["Open"].iloc[-1]) if df_krw is not None and not df_krw.empty else 0.0
        fx = float(df_krw["FX_Rate"].iloc[-1]) if df_krw is not None and "FX_Rate" in df_krw.columns and not df_krw.empty else DEFAULT_FX_FALLBACK
        bar_ts = str(df_krw.index[-1])[:16] if df_krw is not None and not df_krw.empty else ""
        return DnaEvaluation(
            ticker=sym,
            status="DISQUALIFIED",
            is_qualified=False,
            is_buy_signal=False,
            is_exit_signal=False,
            alpha_score=0.0,
            atr_pct=0.0,
            gap_freq_pct=0.0,
            dist_from_high_pct=999.0,
            ema200_slope_pct=0.0,
            above_200h=False,
            golden_aligned=False,
            price_usd=round(px_usd, 4),
            price_krw=round(px_krw, 0),
            open_krw=round(op_krw, 0),
            fx_rate=round(fx, 2),
            ema_10d_krw=0.0,
            ema_30d_krw=0.0,
            ema_200h_krw=0.0,
            bar_ts=bar_ts,
            rationale=f"데이터 부족 ({0 if df_usd is None else len(df_usd)}/200봉)",
        )

    close = df_usd["Close"]
    high = df_usd["High"]

    ema_10d = ema_days(close, 10)     # 65h
    ema_30d = ema_days(close, 30)     # 195h
    ema_200h = ema_hours(close, 200)  # 200h
    atr = compute_atr(df_usd, period=14)

    latest_close = float(close.iloc[-1])
    norm_atr_pct = float(atr.iloc[-1] / latest_close * 100.0) if latest_close > 0 else 0.0

    latest_10d = float(ema_10d.iloc[-1])
    latest_30d = float(ema_30d.iloc[-1])
    latest_200h = float(ema_200h.iloc[-1])

    above_200h = latest_close > latest_200h
    golden_aligned = latest_10d > latest_30d

    # Overnight gap (>1%) frequency over recent 390 bars (~60 trading days)
    lookback = min(len(df_usd), 390)
    sub = df_usd.iloc[-lookback:]
    time_diff = sub.index.to_series().diff()
    is_overnight = time_diff > pd.Timedelta(hours=4)
    gaps = (sub["Open"] - sub["Close"].shift(1)).abs() / sub["Close"].shift(1) * 100.0
    valid_gaps = gaps[is_overnight].dropna()
    gap_freq_pct = float((valid_gaps > 1.0).mean() * 100.0) if not valid_gaps.empty else 0.0

    # Distance from 60-bar high
    recent_high_60 = float(high.iloc[-60:].max())
    dist_from_high_pct = float((recent_high_60 - latest_close) / recent_high_60 * 100.0) if recent_high_60 > 0 else 999.0

    # 200h EMA 20-bar slope
    ema200_prev = float(ema_200h.iloc[-20])
    ema200_slope = ((latest_200h - ema200_prev) / ema200_prev * 100.0) if ema200_prev > 0 else 0.0

    # 100-Point Champion DNA Scoring (same rubric as src/screener.py)
    score = 0.0
    if above_200h:
        score += 20.0
    if golden_aligned:
        score += 15.0

    if 1.2 <= norm_atr_pct <= 3.5:
        score += 25.0
    elif norm_atr_pct > 3.5:
        score += 18.0
    elif 0.9 <= norm_atr_pct < 1.2:
        score += 10.0

    if gap_freq_pct >= 40.0:
        score += 15.0
    elif gap_freq_pct >= 25.0:
        score += 10.0

    if dist_from_high_pct <= 5.0:
        score += 10.0
    elif dist_from_high_pct <= 10.0:
        score += 5.0

    if ema200_slope > 1.0:
        score += 15.0
    elif ema200_slope > 0.0:
        score += 8.0
    score = min(100.0, max(0.0, score))

    is_qualified = above_200h and golden_aligned and (norm_atr_pct >= 1.1) and (score >= 65.0)
    is_buy_signal = is_qualified and (dist_from_high_pct <= 6.0)
    is_exit_signal = not golden_aligned  # 10일 거시선 <= 30일 거시선

    if is_buy_signal:
        status = "BUY_READY"
        rationale = (
            f"🏆 유망3호+DNA 매수충족 (점수 {score:.0f}, ATR {norm_atr_pct:.2f}%, "
            f"갭빈도 {gap_freq_pct:.0f}%, 고점대비 -{dist_from_high_pct:.1f}%)"
        )
    elif is_qualified:
        status = "WATCHLIST"
        rationale = f"👀 DNA 적격 관심대기 (점수 {score:.0f}, 신고가 -{dist_from_high_pct:.1f}% 눌림)"
    else:
        status = "DISQUALIFIED"
        reasons = []
        if not above_200h:
            reasons.append("200h선 하회")
        if not golden_aligned:
            reasons.append("10/30일 역배열")
        if norm_atr_pct < 1.1:
            reasons.append(f"저변동(ATR {norm_atr_pct:.2f}%)")
        if not reasons:
            reasons.append(f"DNA 점수 미달({score:.0f}/65)")
        rationale = "❌ 탈락: " + ", ".join(reasons)

    px_krw = float(df_krw["Close"].iloc[-1])
    op_krw = float(df_krw["Open"].iloc[-1])
    fx = float(df_krw["FX_Rate"].iloc[-1]) if "FX_Rate" in df_krw.columns else DEFAULT_FX_FALLBACK
    bar_ts = pd.to_datetime(df_krw.index[-1]).strftime("%Y-%m-%d %H:%M")

    return DnaEvaluation(
        ticker=sym,
        status=status,
        is_qualified=is_qualified,
        is_buy_signal=is_buy_signal,
        is_exit_signal=is_exit_signal,
        alpha_score=round(score, 1),
        atr_pct=round(norm_atr_pct, 2),
        gap_freq_pct=round(gap_freq_pct, 1),
        dist_from_high_pct=round(dist_from_high_pct, 1),
        ema200_slope_pct=round(ema200_slope, 2),
        above_200h=above_200h,
        golden_aligned=golden_aligned,
        price_usd=round(latest_close, 4),
        price_krw=round(px_krw, 0),
        open_krw=round(op_krw, 0),
        fx_rate=round(fx, 2),
        ema_10d_krw=round(float(df_krw["EMA_10D"].iloc[-1]), 0),
        ema_30d_krw=round(float(df_krw["EMA_30D"].iloc[-1]), 0),
        ema_200h_krw=round(float(df_krw["EMA_200H"].iloc[-1]), 0),
        bar_ts=bar_ts,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# SimulatedTrader (Modeled after jeaweon/stock- Gist Persistence Engine)
# ---------------------------------------------------------------------------

class SimulatedTrader:
    """Simulated KRW Paper Trading Engine with GitHub Gist Cloud Persistence."""

    def __init__(
        self,
        initial_balance: float = DEFAULT_INITIAL_BALANCE_KRW,
        save_file: str = str(DEFAULT_SAVE_PATH),
        max_positions: int = DEFAULT_MAX_POSITIONS,
        fee_rate: float = DEFAULT_FEE_RATE,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        gist_id: str = "",
        github_token: str = "",
        gist_filename: str = "trader_state.json",
    ):
        self.save_file = str(save_file)
        self.gist_filename = gist_filename
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.max_positions = int(max_positions)
        self.fee_rate = float(fee_rate)
        self.stop_loss_pct = float(stop_loss_pct)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_logs: List[Dict[str, Any]] = []
        self.gist_id = str(gist_id or "").strip()
        self.github_token = str(github_token or "").strip()

        self._load_from_file()

    def _build_state_dict(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "initial_balance": self.initial_balance,
            "max_positions": self.max_positions,
            "fee_rate": self.fee_rate,
            "stop_loss_pct": self.stop_loss_pct,
            "positions": self.positions,
            "trade_logs": self.trade_logs,
        }

    def _save_to_file(self) -> Optional[str]:
        state = self._build_state_dict()
        save_path = Path(self.save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

        if self.gist_id and self.github_token:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            payload = {
                "files": {
                    self.gist_filename: {
                        "content": json.dumps(state, ensure_ascii=False, indent=4)
                    }
                }
            }
            try:
                res = requests.patch(
                    f"https://api.github.com/gists/{self.gist_id}",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                if res.status_code not in (200, 201):
                    return f"Gist 백업 응답 코드 {res.status_code}"
            except Exception as e:
                return f"Gist 백업 오류: {e}"
        return None

    def _apply_loaded_state(self, state: Dict[str, Any]) -> None:
        """Apply and normalize loaded JSON state (safe with legacy stock- Gist schemas)."""
        self.initial_balance = float(state.get("initial_balance", self.initial_balance))
        self.cash = float(state.get("cash", self.initial_balance))
        self.max_positions = int(state.get("max_positions", self.max_positions))
        self.fee_rate = float(state.get("fee_rate", self.fee_rate))
        self.stop_loss_pct = float(state.get("stop_loss_pct", self.stop_loss_pct))

        raw_positions = state.get("positions", {})
        norm_positions: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_positions, dict):
            for sym, pos in raw_positions.items():
                if not isinstance(pos, dict):
                    continue
                qty = int(pos.get("qty", 0))
                if qty <= 0:
                    continue
                avg_px = float(pos.get("avg_price", 0.0))
                cost_basis = float(pos.get("cost_basis_krw", avg_px * qty))
                entry_fee = float(pos.get("entry_fee", 0.0))
                stop_px = float(pos.get("stop_loss_price", avg_px * (1.0 - self.stop_loss_pct)))
                norm_positions[str(sym)] = {
                    **pos,
                    "qty": qty,
                    "avg_price": avg_px,
                    "cost_basis_krw": cost_basis,
                    "entry_fee": entry_fee,
                    "stop_loss_price": stop_px,
                    "entry_strategy": pos.get("entry_strategy", "기존 Gist 포지션"),
                    "entry_reason": pos.get("entry_reason", pos.get("entry_strategy", "-")),
                    "entry_bar_ts": pos.get("entry_bar_ts", "-"),
                    "last_trade_bar_ts": pos.get("last_trade_bar_ts", ""),
                }
        self.positions = norm_positions

        raw_logs = state.get("trade_logs", [])
        norm_logs: List[Dict[str, Any]] = []
        if isinstance(raw_logs, list):
            for row in raw_logs:
                if not isinstance(row, dict):
                    continue
                norm_logs.append({
                    "일자/시각": row.get("일자/시각", row.get("일자", "-")),
                    "타입": row.get("타입", "-"),
                    "종목": row.get("종목", "-"),
                    "가격(원)": row.get("가격(원)", row.get("가격", 0)),
                    "수량(주)": row.get("수량(주)", row.get("수량", 0)),
                    "수수료(원)": row.get("수수료(원)", 0),
                    "실현손익(원)": row.get("실현손익(원)", row.get("실현손익", 0)),
                    "수익률(%)": float(row.get("수익률(%)", 0.0) or 0.0),
                    "진입 전략": row.get("진입 전략", "-"),
                    "상세 사유": row.get("상세 사유", "-"),
                })
        self.trade_logs = norm_logs

    def _load_from_file(self) -> None:
        state = None

        # 1. Load from GitHub Gist first if configured
        if self.gist_id and self.github_token:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            try:
                res = requests.get(
                    f"https://api.github.com/gists/{self.gist_id}",
                    headers=headers,
                    timeout=10,
                )
                if res.status_code == 200:
                    gist_data = res.json()
                    content = gist_data.get("files", {}).get(self.gist_filename, {}).get("content")
                    if content:
                        state = json.loads(content)
            except Exception:
                pass

        # 2. Local file fallback
        if state is None and os.path.exists(self.save_file):
            try:
                with open(self.save_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                state = None

        if isinstance(state, dict):
            self._apply_loaded_state(state)

    def reset_account(self, new_initial_balance: Optional[float] = None) -> Optional[str]:
        """Reset cash, positions, and trade logs cleanly on both local & Gist."""
        if new_initial_balance is not None:
            self.initial_balance = float(new_initial_balance)
        self.cash = float(self.initial_balance)
        self.positions = {}
        self.trade_logs = []
        return self._save_to_file()

    def restore_from_json_bytes(self, raw_bytes: bytes) -> Tuple[bool, str]:
        """Restore account state from uploaded backup JSON bytes and sync to Gist."""
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(parsed, dict) or "cash" not in parsed:
                return False, "유효한 계좌 백업 JSON 형식이 아닙니다."
            self._apply_loaded_state(parsed)
            err = self._save_to_file()
            return True, "복원 완료" + (f" (Gist 경고: {err})" if err else " (로컬 + 클라우드 동기화 완료)")
        except Exception as e:
            return False, f"복원 실패: {e}"

    def check_gist_connection(self) -> Tuple[bool, str]:
        """Verify GitHub Gist connectivity mirroring stock-'s diagnostics."""
        if not self.gist_id or not self.github_token:
            return False, "❌ Secrets에 GIST_ID 또는 GITHUB_TOKEN 값이 빠져 있습니다!"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        try:
            res = requests.get(
                f"https://api.github.com/gists/{self.gist_id}",
                headers=headers,
                timeout=10,
            )
            if res.status_code == 200:
                return True, "✅ 클라우드(Gist) 연동이 완벽하게 설정되었습니다!"
            if res.status_code == 404:
                return False, "❌ 연결 실패: Gist 아이디(URL)가 잘못되었거나, 토큰 권한이 부족합니다."
            return False, f"❌ 오류 코드 {res.status_code}: 설정값을 다시 확인해주세요."
        except Exception as e:
            return False, f"❌ 네트워크 오류: {e}"

    def buy_integer_shares(
        self,
        symbol: str,
        price_krw: float,
        allocation_krw: float,
        strategy_name: str,
        reason: str,
        bar_ts: str,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Execute an integer-share KRW buy order with one-way 0.1% fee deduction."""
        sym = symbol.upper().strip()
        if price_krw <= 0:
            return False, f"[{sym}] 유효하지 않은 원화 가격입니다: {price_krw}"
        if sym in self.positions:
            return False, f"[{sym}] 이미 보유 중인 종목입니다."

        budget = min(self.cash, float(allocation_krw))
        if budget < MIN_CASH_ORDER_KRW:
            return False, f"[{sym}] 가용 매수 예산 부족 (₩{budget:,.0f})"

        unit_with_fee = price_krw * (1.0 + self.fee_rate)
        qty = int(budget // unit_with_fee)
        if qty <= 0:
            return False, f"예산초과(1주 필요액 ₩{unit_with_fee:,.0f} > 슬롯 배분금 ₩{budget:,.0f})"

        while qty > 0:
            cost_krw = qty * price_krw
            fee_krw = cost_krw * self.fee_rate
            total_deduct = cost_krw + fee_krw
            if total_deduct <= self.cash + 1e-6:
                break
            qty -= 1

        if qty <= 0:
            return False, f"[{sym}] 수수료 포함 잔고 부족"

        self.cash = round(max(0.0, self.cash - total_deduct), 0)
        stop_price_krw = round(price_krw * (1.0 - self.stop_loss_pct), 0)

        self.positions[sym] = {
            "qty": int(qty),
            "avg_price": round(float(price_krw), 0),
            "cost_basis_krw": round(cost_krw, 0),
            "entry_fee": round(fee_krw, 0),
            "stop_loss_price": stop_price_krw,
            "entry_strategy": strategy_name,
            "entry_reason": reason,
            "entry_indicators": indicators or {},
            "entry_bar_ts": bar_ts,
            "last_trade_bar_ts": bar_ts,
        }

        self.trade_logs.append({
            "일자": bar_ts,
            "타입": "매수",
            "종목": sym,
            "가격(원)": int(round(price_krw)),
            "수량": int(qty),
            "수수료(원)": int(round(fee_krw)),
            "실현손익(원)": 0,
            "수익률(%)": 0.0,
            "진입 전략": strategy_name,
            "상세 사유": reason,
        })
        self._save_to_file()
        return True, f"✅ [{sym}] {qty}주 매수 (@{int(round(price_krw)):,}원, 수수료 {int(round(fee_krw)):,}원) - {reason}"

    def sell_all_shares(
        self,
        symbol: str,
        price_krw: float,
        reason: str,
        bar_ts: str,
    ) -> Tuple[bool, str]:
        """Sell entire integer position with 0.1% exit fee and record complete journal entry."""
        sym = symbol.upper().strip()
        pos = self.positions.get(sym)
        if not pos or int(pos.get("qty", 0)) <= 0:
            return False, f"[{sym}] 보유 중인 수량이 없습니다."
        if price_krw <= 0:
            return False, f"[{sym}] 유효하지 않은 매도 가격입니다."

        qty = int(pos["qty"])
        cost_basis_krw = float(pos.get("cost_basis_krw", qty * float(pos["avg_price"])))
        entry_fee = float(pos.get("entry_fee", 0.0))
        invested_total = cost_basis_krw + entry_fee

        gross_val = qty * float(price_krw)
        exit_fee = gross_val * self.fee_rate
        net_proceeds = gross_val - exit_fee

        net_profit = net_proceeds - invested_total
        roi = (net_profit / invested_total) * 100.0 if invested_total > 0 else 0.0
        total_fees = entry_fee + exit_fee

        self.cash = round(self.cash + net_proceeds, 0)
        entry_strategy = pos.get("entry_strategy", "유망3호 챔피언 DNA")
        del self.positions[sym]

        self.trade_logs.append({
            "일자": bar_ts,
            "타입": "매도",
            "종목": sym,
            "가격(원)": int(round(price_krw)),
            "수량": int(qty),
            "수수료(원)": int(round(total_fees)),
            "실현손익(원)": int(round(net_profit)),
            "수익률(%)": round(roi, 2),
            "진입 전략": entry_strategy,
            "상세 사유": reason,
        })
        self._save_to_file()
        return True, f"🛑 [{sym}] {qty}주 매도 (@{int(round(price_krw)):,}원, 순손익 {int(round(net_profit)):+,}원 / {roi:+.2f}%) - {reason}"

    def run_api_scan_and_trade(
        self,
        evaluations: Dict[str, DnaEvaluation],
        auto_trade: bool = True,
    ) -> Dict[str, Any]:
        """Execute Promising Strategy #3 exits & Screener BUY_READY integer-share entries."""
        sell_msgs: List[str] = []
        buy_msgs: List[str] = []
        skipped_over_budget: List[Dict[str, Any]] = []
        sold_this_bar = set()

        # 1. Check Exits on Held Positions First (Hard Stop & Macro Dead Cross)
        for sym in list(self.positions.keys()):
            ev = evaluations.get(sym)
            if ev is None or ev.price_krw <= 0:
                continue
            pos = self.positions[sym]
            # Same-bar duplicate guard
            if pos.get("last_trade_bar_ts") == ev.bar_ts:
                continue

            stop_krw = float(pos["stop_loss_price"])
            if ev.open_krw <= stop_krw or ev.price_krw <= stop_krw:
                exec_krw = ev.open_krw if ev.open_krw <= stop_krw else ev.price_krw
                reason = f"🚨 -12% 하드스탑 손절 (기준 {int(stop_krw):,}원 / 체결 {int(exec_krw):,}원)"
                if auto_trade:
                    ok, msg = self.sell_all_shares(sym, exec_krw, reason, ev.bar_ts)
                    if ok:
                        sell_msgs.append(msg)
                        sold_this_bar.add(sym)
                continue

            if ev.is_exit_signal:
                reason = f"📉 10일 거시선({int(ev.ema_10d_krw):,}원) ≤ 30일 거시선({int(ev.ema_30d_krw):,}원) 데드크로스 매도"
                if auto_trade:
                    ok, msg = self.sell_all_shares(sym, ev.price_krw, reason, ev.bar_ts)
                    if ok:
                        sell_msgs.append(msg)
                        sold_this_bar.add(sym)

        # 2. Rank Screener BUY_READY Candidates
        candidates = [
            ev for sym, ev in evaluations.items()
            if ev.is_buy_signal and sym not in self.positions and sym not in sold_this_bar
        ]
        candidates.sort(key=lambda e: (e.alpha_score, e.atr_pct), reverse=True)

        # 3. Execute Auto-Buys (up to max_positions, integer shares only, skip over-budget tickers)
        bought_syms = set()
        over_budget_syms = set()
        available_slots = max(0, self.max_positions - len(self.positions))

        if auto_trade and available_slots > 0 and candidates:
            for cand in candidates:
                if available_slots <= 0 or self.cash < MIN_CASH_ORDER_KRW:
                    break
                alloc_krw = self.cash / available_slots
                unit_need = cand.price_krw * (1.0 + self.fee_rate)
                if int(min(self.cash, alloc_krw) // unit_need) <= 0:
                    over_budget_syms.add(cand.ticker)
                    skipped_over_budget.append({
                        "ticker": cand.ticker,
                        "unit_krw": round(unit_need, 0),
                        "slot_budget_krw": round(alloc_krw, 0),
                    })
                    continue

                strat_label = f"유망3호 DNA (점수 {cand.alpha_score:.0f} / ATR {cand.atr_pct:.1f}%)"
                indicators = {
                    "alpha_score": cand.alpha_score,
                    "atr_pct": cand.atr_pct,
                    "gap_freq_pct": cand.gap_freq_pct,
                    "dist_from_high_pct": cand.dist_from_high_pct,
                    "ema200_slope_pct": cand.ema200_slope_pct,
                    "fx_rate": cand.fx_rate,
                }
                ok, msg = self.buy_integer_shares(
                    symbol=cand.ticker,
                    price_krw=cand.price_krw,
                    allocation_krw=alloc_krw,
                    strategy_name=strat_label,
                    reason=cand.rationale,
                    bar_ts=cand.bar_ts,
                    indicators=indicators,
                )
                if ok:
                    buy_msgs.append(msg)
                    bought_syms.add(cand.ticker)
                    available_slots -= 1

        # 4. Build Screener Inspection Table
        scan_rows: List[Dict[str, Any]] = []
        for sym, ev in evaluations.items():
            if sym in bought_syms:
                disp_status = "✅ 매수체결"
            elif sym in self.positions:
                disp_status = "📌 보유중"
            elif sym in over_budget_syms:
                disp_status = "💸 예산초과 스킵"
            elif ev.status == "BUY_READY":
                disp_status = "🎯 BUY_READY"
            elif ev.status == "WATCHLIST":
                disp_status = "👀 WATCHLIST"
            else:
                disp_status = "⚪ 조건미달"

            scan_rows.append({
                "종목": sym,
                "상태": disp_status,
                "원화현재가(원)": int(round(ev.price_krw)),
                "달러가($)": round(ev.price_usd, 2),
                "DNA 점수": ev.alpha_score,
                "시간당 ATR(%)": ev.atr_pct,
                "갭빈도(%)": ev.gap_freq_pct,
                "고점대비(%)": -ev.dist_from_high_pct if ev.dist_from_high_pct < 900 else None,
                "기준봉 시각": ev.bar_ts,
                "검색 진단 사유": ev.rationale,
            })

        prio = {
            "✅ 매수체결": 0,
            "🎯 BUY_READY": 1,
            "💸 예산초과 스킵": 2,
            "📌 보유중": 3,
            "👀 WATCHLIST": 4,
            "⚪ 조건미달": 5,
        }
        scan_rows.sort(key=lambda r: (prio.get(r["상태"], 9), -r["DNA 점수"], -r["시간당 ATR(%)"]))

        return {
            "sell_executions": sell_msgs,
            "buy_executions": buy_msgs,
            "buy_ready_count": len(candidates),
            "skipped_over_budget": skipped_over_budget,
            "scan_df": pd.DataFrame(scan_rows),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Compute overall trading journal KPIs (win rate, ROI, profit factor, total fees)."""
        sell_logs = [log for log in self.trade_logs if log.get("타입") == "매도"]
        total_closed = len(sell_logs)
        if total_closed == 0:
            total_fees = sum(float(log.get("수수료(원)", 0)) for log in self.trade_logs if log.get("타입") == "매수")
            return {
                "total_closed": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_roi": 0.0,
                "profit_factor": 0.0,
                "realized_profit_krw": 0,
                "total_fees_krw": int(round(total_fees)),
            }

        rois = [float(l.get("수익률(%)", 0.0)) for l in sell_logs]
        profits = [float(l.get("실현손익(원)", 0.0)) for l in sell_logs]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        gross_win = sum(wins)
        gross_loss = abs(sum(p for p in profits if p < 0))
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        # Sum fees across closed sells (entry+exit already combined on sell rows) + open position entry fees
        closed_fees = sum(float(l.get("수수료(원)", 0)) for l in sell_logs)
        open_fees = sum(float(p.get("entry_fee", 0)) for p in self.positions.values())

        return {
            "total_closed": total_closed,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / total_closed * 100.0, 1),
            "avg_roi": round(float(np.mean(rois)), 2),
            "profit_factor": pf,
            "realized_profit_krw": int(round(sum(profits))),
            "total_fees_krw": int(round(closed_fees + open_fees)),
        }


def run_full_hourly_cycle(
    trader: SimulatedTrader,
    scan_universe: Iterable[str],
    auto_trade: bool = True,
    refresh_state_first: bool = False,
    apply_ext_to_candidates: bool = True,
    fx_series: Optional[pd.Series] = None,
    stage_callback: Optional[Callable[[float, str], None]] = None,
    batch_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Execute one complete 1-hour API scan, FX conversion, extended-hours overlay, and trade cycle.

    Shared single pipeline used by both interactive Streamlit scans (manual button & hourly
    auto-refresh fragment) and headless GitHub Actions hourly cron runs (`auto_trade.py`).
    """
    if refresh_state_first:
        trader._load_from_file()

    if stage_callback:
        stage_callback(0.05, "실시간 환율(KRW=X) 조회 중...")
    if fx_series is None or fx_series.empty:
        fx_series = fetch_usd_krw_hourly(period="100d")

    targets = normalize_tickers(list(trader.positions.keys()) + list(scan_universe))
    raw_batch, failed_map = fetch_live_1h_batch(
        targets,
        period="90d",
        chunk_size=25,
        progress_callback=batch_progress_callback,
    )

    if stage_callback:
        stage_callback(0.88, "챔피언 DNA 4단계 스크리너 분석 및 장외 실시간 가격(prepost) 반영 중...")

    usd_map: Dict[str, pd.DataFrame] = {}
    krw_map: Dict[str, pd.DataFrame] = {}
    eval_map: Dict[str, DnaEvaluation] = {}

    for sym, df_u in raw_batch.items():
        if sym in trader.positions:
            df_u = apply_extended_hours_price(sym, df_u)
        df_k = convert_usd_to_krw_1h(df_u, fx_series)
        ev = evaluate_champion_dna_1h(df_u, df_k, sym)
        usd_map[sym] = df_u
        krw_map[sym] = df_k
        eval_map[sym] = ev

    if apply_ext_to_candidates:
        for sym, ev in list(eval_map.items()):
            if ev.is_buy_signal and sym not in trader.positions:
                df_u_ext = apply_extended_hours_price(sym, usd_map[sym])
                df_k_ext = convert_usd_to_krw_1h(df_u_ext, fx_series)
                usd_map[sym] = df_u_ext
                krw_map[sym] = df_k_ext
                eval_map[sym] = evaluate_champion_dna_1h(df_u_ext, df_k_ext, sym)

    if stage_callback:
        stage_callback(0.96, "유망 매매 방법 3호 매수/매도 체결 엔진 실행 중...")

    scan_res = trader.run_api_scan_and_trade(eval_map, auto_trade=auto_trade)
    scan_res["scanned_count"] = len(eval_map)
    scan_res["failed_count"] = len(failed_map)
    scan_res["fx_series"] = fx_series
    scan_res["usd_map"] = usd_map
    scan_res["krw_map"] = krw_map
    scan_res["eval_map"] = eval_map
    scan_res["executed_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if stage_callback:
        stage_callback(1.0, "검색 및 자동 매매 완료!")

    return scan_res

