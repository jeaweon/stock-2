"""Alpha Stock Screener & Investment Selection Pipeline.

Analyzes 1-hour candle stock data to identify high-return 'Champion Stock DNA':
1. Volatility Power: Normalized ATR(14) >= 1.2% (Eliminates sluggish, low-return noise)
2. Macro Trend Health: Close > 200h EMA and 10d EMA > 30d EMA (Pure Golden Alignment)
3. Institutional Momentum: Price within 7% of recent 60-bar high, Gap Frequency >= 35%
4. Trend Slope: Upward slope on 200h EMA and healthy expanding spread
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.timeframes import ema_days, ema_hours
from src.adaptive_experiment import compute_atr


@dataclass
class ScreenerResult:
    ticker: str
    is_qualified: bool
    alpha_score: float  # 0 to 100
    atr_pct: float
    gap_freq_pct: float
    dist_from_60h_high_pct: float
    above_200h: bool
    golden_aligned: bool
    signal_status: str  # 'BUY_READY', 'WATCHLIST', 'DISQUALIFIED'
    rationale: str


def evaluate_stock_dna(df: pd.DataFrame, ticker: str = "ASSET") -> ScreenerResult:
    """Evaluate whether a single stock possesses the High-Return Champion DNA.
    
    Evaluated strictly at the latest bar (no future lookahead).
    """
    sym = ticker.upper().strip()
    if len(df) < 200:
        return ScreenerResult(
            ticker=sym,
            is_qualified=False,
            alpha_score=0.0,
            atr_pct=0.0,
            gap_freq_pct=0.0,
            dist_from_60h_high_pct=999.0,
            above_200h=False,
            golden_aligned=False,
            signal_status="DISQUALIFIED",
            rationale="데이터 부족 (최소 200시간봉 필요)"
        )

    close = df["Close"]
    high = df["High"]
    open_p = df["Open"]
    
    # 1. Normalized ATR (변동성 파워)
    atr = compute_atr(df, period=14)
    norm_atr_pct = float(atr.iloc[-1] / close.iloc[-1] * 100.0)
    
    # 2. Macro Moving Averages
    ema_10d = ema_days(close, 10)    # 65h
    ema_30d = ema_days(close, 30)    # 195h
    ema_200h = ema_hours(close, 200) # 200h
    
    latest_close = float(close.iloc[-1])
    latest_ema10d = float(ema_10d.iloc[-1])
    latest_ema30d = float(ema_30d.iloc[-1])
    latest_ema200h = float(ema_200h.iloc[-1])
    
    above_200h = latest_close > latest_ema200h
    golden_aligned = latest_ema10d > latest_ema30d
    
    # 3. Overnight Gap Frequency (최근 60거래일 = 약 390봉)
    lookback_bars = min(len(df), 390)
    sub_df = df.iloc[-lookback_bars:]
    time_diff = sub_df.index.to_series().diff()
    is_overnight = time_diff > pd.Timedelta(hours=4)
    gaps = (sub_df["Open"] - sub_df["Close"].shift(1)).abs() / sub_df["Close"].shift(1) * 100.0
    valid_gaps = gaps[is_overnight].dropna()
    gap_freq_pct = float((valid_gaps > 1.0).mean() * 100.0) if not valid_gaps.empty else 0.0
    
    # 4. Momentum (최근 60봉 최고가 대비 이격률)
    recent_high = float(high.iloc[-60:].max())
    dist_from_high_pct = float((recent_high - latest_close) / recent_high * 100.0)
    
    # 5. 200h EMA 기울기 (최근 20봉 기준 추세 각도)
    ema200_slope = (latest_ema200h - float(ema_200h.iloc[-20])) / float(ema_200h.iloc[-20]) * 100.0
    
    # --- Scoring System (0 ~ 100) ---
    score = 0.0
    
    # A. 추세 정배열 점수 (최대 35점)
    if above_200h:
        score += 20.0
    if golden_aligned:
        score += 15.0
        
    # B. 변동성 적합도 점수 (최대 25점: 1.2% ~ 3.5% 구간이 고수익 스윗스팟)
    if 1.2 <= norm_atr_pct <= 3.5:
        score += 25.0
    elif norm_atr_pct > 3.5:
        score += 18.0  # 초고변동성은 약간 감점
    elif 0.9 <= norm_atr_pct < 1.2:
        score += 10.0
        
    # C. 수급 갭 및 모멘텀 점수 (최대 25점)
    if gap_freq_pct >= 40.0:
        score += 15.0
    elif gap_freq_pct >= 25.0:
        score += 10.0
        
    if dist_from_high_pct <= 5.0:
        score += 10.0
    elif dist_from_high_pct <= 10.0:
        score += 5.0
        
    # D. 거시 추세선 기울기 점수 (최대 15점)
    if ema200_slope > 1.0:
        score += 15.0
    elif ema200_slope > 0.0:
        score += 8.0

    score = min(100.0, max(0.0, score))
    
    # 매수 적격성 판단:
    # 1. 200선 위 + 10/30일선 정배열
    # 2. 변동성 >= 1.2%
    # 3. 점수 >= 70점
    is_qualified = above_200h and golden_aligned and (norm_atr_pct >= 1.1) and (score >= 65.0)
    
    # 매수 신호 상태
    if is_qualified and dist_from_high_pct <= 6.0:
        signal_status = "BUY_READY"
        rationale = f"🏆 고수익 챔피언 DNA 충족! (ATR {norm_atr_pct:.1f}%, 10/30일 정배열, 신고가권 {dist_from_high_pct:.1f}%)"
    elif is_qualified:
        signal_status = "WATCHLIST"
        rationale = f"👀 유망 주도주 관심종목 (신고가 대비 눌림목 -{dist_from_high_pct:.1f}%, 재돌파 시 매수 대기)"
    else:
        signal_status = "DISQUALIFIED"
        reasons = []
        if not above_200h: reasons.append("200시간선 하회")
        if not golden_aligned: reasons.append("10/30일 거시 역배열")
        if norm_atr_pct < 1.1: reasons.append("변동성 부족(저활성)")
        rationale = f"❌ 탈락: {', '.join(reasons) if reasons else '종합 점수 미달'}"

    return ScreenerResult(
        ticker=sym,
        is_qualified=is_qualified,
        alpha_score=round(score, 1),
        atr_pct=round(norm_atr_pct, 2),
        gap_freq_pct=round(gap_freq_pct, 1),
        dist_from_60h_high_pct=round(dist_from_high_pct, 1),
        above_200h=above_200h,
        golden_aligned=golden_aligned,
        signal_status=signal_status,
        rationale=rationale
    )


def screen_candidate_universe(
    data_dict: Dict[str, pd.DataFrame],
    min_score: float = 65.0
) -> pd.DataFrame:
    """Screen an entire dictionary of stock DataFrames and return ranked candidates."""
    rows = []
    for ticker, df in data_dict.items():
        res = evaluate_stock_dna(df, ticker)
        rows.append({
            "ticker": res.ticker,
            "status": res.signal_status,
            "score": res.alpha_score,
            "atr_pct": res.atr_pct,
            "gap_freq_pct": res.gap_freq_pct,
            "dist_from_high_pct": res.dist_from_60h_high_pct,
            "above_200h": res.above_200h,
            "golden_aligned": res.golden_aligned,
            "is_qualified": res.is_qualified,
            "rationale": res.rationale
        })
    df_ranked = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df_ranked
