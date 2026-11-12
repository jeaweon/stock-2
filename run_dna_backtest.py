"""Historical Backtesting Engine for Stocks Reaching the High-Return Champion DNA.

Simulates 2-year 1-hour backtest for stocks that meet the 4-step Champion DNA:
1. Macro Golden Cross: 10d EMA (65h) > 30d EMA (195h)
2. Long-term Trend Support: Close > 200h EMA
3. Volatility Power Energy: Hourly ATR(14) >= 1.1%
4. Momentum Breakout Window: Price within 6% of recent 60-bar High
5. Overnight Gap Down & 0.1% real trading fees included.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_stock_data, CACHE_DIR
from src.backtester import run_backtest
from src.timeframes import ema_days, ema_hours
from src.adaptive_experiment import compute_atr


def generate_champion_dna_signals(df: pd.DataFrame, stop_loss_pct: float = 0.12) -> pd.DataFrame:
    """Generate trading signals strictly based on Champion Stock DNA."""
    d = df.copy()
    close = d["Close"]
    high = d["High"]
    
    # 1. 10d EMA (65h) & 30d EMA (195h)
    ema_10d = ema_days(close, 10)
    ema_30d = ema_days(close, 30)
    # 2. 200h EMA
    ema_200h = ema_hours(close, 200)
    # 3. ATR % (14 period)
    atr = compute_atr(d, period=14)
    atr_pct = (atr / close) * 100.0
    # 4. Recent 60h High
    recent_high_60 = high.rolling(window=60, min_periods=20).max()
    dist_from_high_pct = (recent_high_60 - close) / recent_high_60 * 100.0
    
    d["signal"] = 0
    # Entry Condition:
    entry_cond = (
        (ema_10d > ema_30d) & 
        (close > ema_200h) & 
        (atr_pct >= 1.1) & 
        (dist_from_high_pct <= 6.0)
    )
    d.loc[entry_cond, "signal"] = 1
    
    # Exit Condition: Macro Dead Cross
    exit_cond = (ema_10d <= ema_30d)
    d.loc[exit_cond, "signal"] = -1
    
    d["position_size"] = 1.0
    d["stop_loss_pct"] = stop_loss_pct  # 12% 와이드 하드스탑
    return d


def main():
    parser = argparse.ArgumentParser(description="고수익 챔피언 DNA 조건 도달 종목 과거 데이터 백테스트")
    parser.add_argument("--group", choices=["qualified", "top_winners", "all"], default="qualified",
                        help="백테스트 대상 종목군 (qualified: BUY_READY & WATCHLIST, top_winners: 100%%+ 초고수익주, all: 전체)")
    parser.add_argument("--fee", type=float, default=0.001, help="편도 거래 수수료 (기본: 0.001 = 0.1%%)")
    args = parser.parse_args()

    print("=" * 110)
    print("🚀 [고수익 챔피언 DNA 조건 도달 종목 과거 2년 1시간봉 실증 백테스트]")
    print("   매수 진입 조건: 10일>30일 EMA 정배열 + 200시간선 상단 + ATR 변동성(≥1.1%) + 신고가권(-6% 이내)")
    print("   청산/손절 조건: 10일선<=30일선 데드크로스 전량청산 + 12% 와이드 하드스탑 (갭하락 시가체결) + 편도 0.1% 수수료")
    print("=" * 110)

    # 1. 대상 종목 선정
    sc_file = Path("screener_qualified_candidates.csv")
    scan_file = Path("scanner_results.csv")
    
    target_tickers = []
    if args.group == "qualified" and sc_file.exists():
        sc = pd.read_csv(sc_file)
        # BUY_READY and WATCHLIST
        target_tickers = sc[sc["status"].isin(["BUY_READY", "WATCHLIST"])]["ticker"].tolist()
        print(f"🎯 스크리너 적격 종목군 (BUY_READY 22개 + WATCHLIST 15개) 총 {len(target_tickers)}개 선정")
    elif args.group == "top_winners" and scan_file.exists():
        scan = pd.read_csv(scan_file)
        target_tickers = scan[scan["strategy_return"] >= 100.0]["ticker"].tolist()
        print(f"🏆 초고수익 종목군 (전략 수익률 ≥ 100%) 총 {len(target_tickers)}개 선정")
    else:
        # Load from all cached files
        cache_files = list(CACHE_DIR.glob("*_1h_730d.csv"))
        target_tickers = [f.name.split("_")[0] for f in cache_files]
        print(f"🌐 전체 로컬 캐시 유니버스 총 {len(target_tickers)}개 선정")

    # Additional hot growth / momentum tickers if cached
    extra_hot = ["TEM", "ALAB", "VRT", "OKLO", "SMR", "NNE", "BE", "PLTR", "NVDA", "MSTR"]
    for t in extra_hot:
        if (CACHE_DIR / f"{t}_1h_730d.csv").exists() and t not in target_tickers:
            target_tickers.append(t)

    results = []
    print(f"📊 {len(target_tickers)}개 종목에 대해 2년 과거 데이터 백테스트 시뮬레이션 시작...\n")

    for i, sym in enumerate(target_tickers, 1):
        try:
            df = load_stock_data(sym)
            if len(df) < 200:
                continue
            res = run_backtest(df, signal_func=generate_champion_dna_signals, fee_rate=args.fee)
            bnh = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100.0
            alpha = res.total_return_pct - bnh
            
            results.append({
                "ticker": sym,
                "dna_strategy_return": round(res.total_return_pct, 2),
                "bnh_return": round(bnh, 2),
                "alpha_vs_bnh": round(alpha, 2),
                "win_rate_pct": round(res.win_rate_pct, 2),
                "trades": res.total_trades,
                "mdd_pct": round(res.max_drawdown_pct, 2),
                "profit_factor": round(res.profit_factor, 2) if res.profit_factor < 900 else 99.0
            })
            print(f"[{i:02d}/{len(target_tickers):02d}] {sym:<5}: 전략 수익 = {res.total_return_pct:+7.2f}%, B&H = {bnh:+7.2f}%, 알파 = {alpha:+7.2f}%p, 승률 = {res.win_rate_pct:4.1f}%, 매매 = {res.total_trades:2d}회")
        except Exception as e:
            print(f"[{i:02d}/{len(target_tickers):02d}] {sym:<5}: 처리 실패 ({e})")

    if not results:
        print("❌ 테스트 결과가 없습니다.")
        return

    res_df = pd.DataFrame(results).sort_values("dna_strategy_return", ascending=False)
    
    print("\n" + "=" * 110)
    print("📈 [고수익 챔피언 DNA 백테스트 상위 종목 랭킹]")
    print("=" * 110)
    print(res_df.head(25).to_string(index=False))

    # Summary Statistics
    total_count = len(res_df)
    pos_returns = (res_df["dna_strategy_return"] > 0).sum()
    bnh_wins = (res_df["alpha_vs_bnh"] > 0).sum()
    avg_strat = res_df["dna_strategy_return"].mean()
    avg_bnh = res_df["bnh_return"].mean()
    avg_alpha = res_df["alpha_vs_bnh"].mean()
    avg_win = res_df["win_rate_pct"].mean()
    avg_trades = res_df["trades"].mean()
    avg_mdd = res_df["mdd_pct"].mean()

    print("\n" + "=" * 110)
    print("📊 [고수익 챔피언 DNA 백테스트 종합 성과 요약]")
    print(f" • 총 검증 종목: {total_count}개")
    print(f" • 📈 플러스(+) 수익 종목 수: {pos_returns}개 / {total_count}개 ({pos_returns/total_count*100:.1f}%)")
    print(f" • 🏆 B&H 초과 승리 종목 수: {bnh_wins}개 / {total_count}개 ({bnh_wins/total_count*100:.1f}%)")
    print(f" • 💰 평균 2년 전략 수익률: {avg_strat:+.2f}%")
    print(f" • 📦 평균 2년 B&H 수익률: {avg_bnh:+.2f}%")
    print(f" • 🚀 평균 초과 알파: {avg_alpha:+.2f}%p")
    print(f" • 🎯 평균 승률: {avg_win:.1f}%")
    print(f" • 🔄 평균 매매 횟수: {avg_trades:.1f}회 (2년간 저빈도 장기 추세 라이딩)")
    print(f" • 🛡️ 평균 최대낙폭 (MDD): {avg_mdd:.1f}%")
    print("=" * 110)

    out_file = "dna_backtest_results.csv"
    res_df.to_csv(out_file, index=False)
    print(f"💾 백테스트 결과가 '{out_file}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
