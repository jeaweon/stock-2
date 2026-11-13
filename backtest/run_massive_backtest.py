"""Massive multi-stock backtesting runner using the locked-in 3-Group Unified Strategies."""
import time
from typing import Dict, List
import pandas as pd
import numpy as np

from src.data_loader import load_stock_data
from src.backtester import run_backtest
from src.adaptive_experiment import profile_asset, STRATEGY_MAP
from src.storage import save_history_record

# 80+ Diverse Market Assets across Tech, Semi, Crypto, Finance, Energy, Defense, Biotech, Consumer, Commodities, and Global ETFs
EXPANDED_TICKERS = [
    # 1. Mega Tech & AI Software (11)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "PLTR", "CRM", "ORCL",
    # 2. Semiconductors & Hardware (10)
    "AMD", "TSM", "AVGO", "QCOM", "MU", "INTC", "SOXL", "ARM", "SMCI", "TXN",
    # 3. Crypto & High-Beta / Leveraged Bull & Bear (10)
    "ETHU", "CONL", "MARA", "COIN", "BITO", "TQQQ", "UVXY", "LABU", "SQQQ", "RIVN",
    # 4. Financials & Payments / FinTech (9)
    "JPM", "V", "MA", "BAC", "GS", "PYPL", "HOOD", "SOFI", "BABA",
    # 5. Aerospace, Defense & Industrial (5)
    "CAT", "BA", "LMT", "GE", "RTX",
    # 6. Traditional & Clean Energy, Utilities (6)
    "XOM", "CVX", "OXY", "ENPH", "FSLR", "NEE",
    # 7. Healthcare, Biotech & Pharma (7)
    "LLY", "UNH", "JNJ", "PFE", "MRNA", "ABBV", "BIIB",
    # 8. Consumer, Retail & Entertainment (9)
    "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "DIS", "SPOT",
    # 9. Commodities & Global / Index ETFs (11)
    "SPY", "QQQ", "IWM", "DIA", "EWY", "FXI", "KORU", "GLD", "SLV", "USO", "EEM", "INDA"
]


def run_massive_backtests(tickers: List[str], initial_capital: float = 10000.0, fee_rate: float = 0.001):
    print("=" * 105)
    print(f"🚀 [대규모 80+ 다변화 전방위 시장 백테스트 실행] 총 {len(tickers)}개 종목 | 초기자본: ${initial_capital:,.0f} | 수수료: {fee_rate*100:.2f}%")
    print("   자산 그룹별 최우수 신규 매수/매도 전략(A, B, C) 전방위 교차 검증")
    print("=" * 105)

    results = []
    start_time = time.time()

    for idx, sym in enumerate(tickers, 1):
        print(f"[{idx:02d}/{len(tickers):02d}] 🔍 {sym:5s} -> ", end="", flush=True)
        try:
            df = load_stock_data(sym, period="730d", interval="1h")
        except Exception as e:
            print(f"데이터 로드 실패 ({e})")
            continue

        if len(df) < 100:
            print(f"데이터 부족 ({len(df)}개 봉)")
            continue

        # 1. Profile Asset into Group A, B, or C
        prof = profile_asset(df, sym)

        # 2. Run Assigned Strategy
        strat_fn, s_name, s_desc = STRATEGY_MAP[prof.assigned_group]
        res = run_backtest(df, signal_func=strat_fn, initial_capital=initial_capital, fee_rate=fee_rate)

        # 3. Buy & Hold Benchmark
        bh_ret = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100.0
        alpha = res.total_return_pct - bh_ret

        # Save to history CSV
        save_history_record(
            ticker=sym,
            iteration_round=300 + ord(prof.assigned_group),
            strategy_name=f"[통합 확정] {s_name}",
            buy_rules=f"{prof.group_name}: {s_desc}",
            sell_rules="통합 리스크 관리 룰",
            position_sizing_rules="자산군별 동적 비중",
            win_rate_pct=res.win_rate_pct,
            total_return_pct=res.total_return_pct,
            mdd_pct=res.max_drawdown_pct,
            total_trades=res.total_trades,
            profit_factor=res.profit_factor,
            sharpe_ratio=res.sharpe_ratio,
            critic_verdict="PASS" if res.total_return_pct > 0 and res.max_drawdown_pct < 35 else "EVALUATED",
            critic_feedback=f"그룹 {prof.assigned_group} 확정 통합 전략 백테스트 (ATR: {prof.mean_normalized_atr_pct}%, 갭: {prof.overnight_gap_freq_pct}%)",
        )

        results.append({
            "ticker": sym,
            "group": prof.assigned_group,
            "group_name": prof.group_name.split(":")[0],
            "strategy_name": s_name,
            "2y_buy_hold_ret": round(bh_ret, 2),
            "strategy_return": round(res.total_return_pct, 2),
            "alpha_vs_hold": round(alpha, 2),
            "win_rate": round(res.win_rate_pct, 2),
            "mdd": round(res.max_drawdown_pct, 2),
            "trades": res.total_trades,
            "profit_factor": round(res.profit_factor, 2),
            "sharpe_ratio": round(res.sharpe_ratio, 2),
            "atr_pct": prof.mean_normalized_atr_pct,
            "gap_freq_pct": prof.overnight_gap_freq_pct,
            # Before vs After compatibility format
            "before_return_pct": round(bh_ret, 2),
            "after_return_pct": round(res.total_return_pct, 2),
            "return_improvement_pctp": round(alpha, 2),
            "before_win_rate_pct": round(res.win_rate_pct, 2),
            "after_win_rate_pct": round(res.win_rate_pct, 2),
            "win_rate_improvement_pctp": 0.0,
            "before_mdd_pct": round(res.max_drawdown_pct, 2),
            "after_mdd_pct": round(res.max_drawdown_pct, 2),
            "mdd_change_pctp": 0.0,
            "after_profit_factor": round(res.profit_factor, 2),
            "critic_verdict": "PASS" if res.total_return_pct > 0 else "EVALUATED",
            "critic_score": 85 if res.total_return_pct > 0 else 55,
        })

        print(f"그룹 {prof.assigned_group} | 전략수익률: {res.total_return_pct:+6.2f}% | 승률: {res.win_rate_pct:4.1f}% | MDD: {res.max_drawdown_pct:4.1f}% | 거래: {res.total_trades:3d}회 (B&H: {bh_ret:+6.1f}%)")

    elapsed = time.time() - start_time
    df_res = pd.DataFrame(results)

    # Save to scanner_results.csv
    df_res.to_csv("scanner_results.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 105)
    print(f"📊 [대규모 백테스트 완료] 총 {len(df_res)}개 종목 완료 (소요시간: {elapsed:.1f}초)")
    print("=" * 105)

    # Sorted by strategy return descending
    df_sorted = df_res.sort_values("strategy_return", ascending=False)
    display_cols = ["ticker", "group", "strategy_name", "strategy_return", "2y_buy_hold_ret", "alpha_vs_hold", "win_rate", "mdd", "trades", "profit_factor"]
    
    print("\n[🏆 전체 종목 수익률 랭킹 TOP 15]")
    print(df_sorted[display_cols].head(15).to_string(index=False))

    print("\n[📉 전체 종목 수익률 랭킹 하위 10]")
    print(df_sorted[display_cols].tail(10).to_string(index=False))

    # Group Statistics
    print("\n" + "=" * 105)
    print("📈 [자산 그룹별 평균 성과 통계]")
    print("=" * 105)
    group_summary = df_res.groupby("group").agg({
        "ticker": "count",
        "strategy_return": "mean",
        "2y_buy_hold_ret": "mean",
        "alpha_vs_hold": "mean",
        "win_rate": "mean",
        "mdd": "mean",
        "trades": "mean",
        "profit_factor": "mean",
    }).rename(columns={"ticker": "종목수", "strategy_return": "전략평균수익률", "2y_buy_hold_ret": "단순보유평균수익", "alpha_vs_hold": "초과수익(알파)", "win_rate": "평균승률", "mdd": "평균MDD", "trades": "평균거래수", "profit_factor": "평균PF"})

    print(group_summary.round(2).to_string())

    print(f"\n • 전체 {len(df_res)}개 종목 평균 전략 수익률: {df_res['strategy_return'].mean():+.2f}%")
    print(f" • 전체 평균 승률: {df_res['win_rate'].mean():.1f}%")
    print(f" • 전체 평균 MDD: {df_res['mdd'].mean():.1f}%")
    print(f" • 승률 50% 이상 종목 수: {(df_res['win_rate'] >= 50.0).sum()}개 ({(df_res['win_rate'] >= 50.0).mean()*100:.1f}%)")
    print(f" • 플러스(+) 수익률 달성 종목 수: {(df_res['strategy_return'] > 0).sum()}개 / {len(df_res)}개 ({(df_res['strategy_return'] > 0).mean()*100:.1f}%)")

    print("\n💾 모든 결과가 'scanner_results.csv' 및 'backtest_history.csv'에 업데이트되었습니다.")
    print("🌐 Streamlit UI(http://localhost:8502)에서 40여 개 종목의 인터랙티브 차트 및 리더보드를 확인하세요!\n")
    return df_res


if __name__ == "__main__":
    run_massive_backtests(EXPANDED_TICKERS)
