"""CLI Runner for '유망 매매 방법 1' (Promising Strategy 1).

Usage:
    python3 run_promising_strategy_1.py               # Runs on default 16 diverse assets
    python3 run_promising_strategy_1.py AAPL NVDA TSLA # Runs on specific tickers
"""
import sys
import os
import pandas as pd
from typing import List

from src.data_loader import load_stock_data
from src.backtester import run_backtest
from src.strategy_presets import PRESETS

DEFAULT_16_TICKERS = [
    "INTC", "ORCL", "UNH", "BA", "NKE", "BIIB", "CRM", "NFLX",
    "SOXL", "MRNA", "ENPH", "BITO", "ETHU", "SQQQ", "UVXY", "KORU"
]


def execute_promising_1(tickers: List[str]):
    preset = PRESETS["promising_1"]
    print("=" * 95)
    print(f"🌟 [{preset.name}] 실전 백테스트 실행")
    print(f"   규칙 요약:")
    for line in preset.summary_rules.split("\n"):
        print(f"    • {line}")
    print("=" * 95)

    rows = []
    for idx, sym in enumerate(tickers, 1):
        sym = sym.upper().strip()
        print(f"[{idx:02d}/{len(tickers):02d}] 🔍 {sym:5s} -> ", end="", flush=True)
        try:
            df = load_stock_data(sym, period="730d", interval="1h")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            continue

        if len(df) < 100:
            print(f"데이터 부족 ({len(df)}개 봉)")
            continue

        sig_df = preset.signal_func(df, sym)
        res = run_backtest(df, df_with_signals=sig_df, initial_capital=10000.0, fee_rate=0.001)
        bh_ret = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100.0
        alpha = res.total_return_pct - bh_ret

        verdict = "PASS (B&H 초과)" if alpha > 0 else "미달"
        print(f"수익률: {res.total_return_pct:+7.2f}% | B&H: {bh_ret:+7.2f}% | 알파: {alpha:+7.2f}%p | 승률: {res.win_rate_pct:4.1f}% | 거래: {res.total_trades:2d}회 | [{verdict}]")

        rows.append({
            "ticker": sym,
            "strategy_return": round(res.total_return_pct, 2),
            "bh_return": round(bh_ret, 2),
            "alpha": round(alpha, 2),
            "win_rate": round(res.win_rate_pct, 1),
            "mdd": round(res.max_drawdown_pct, 1),
            "trades": res.total_trades,
            "profit_factor": round(res.profit_factor, 2)
        })

    df_res = pd.DataFrame(rows)
    print("\n" + "=" * 95)
    print("📊 [유망 매매 방법 1 종합 성과 보고서]")
    print("=" * 95)
    print(f" • 총 검증 종목 수: {len(df_res)}개")
    print(f" • 전략 평균 수익률: {df_res['strategy_return'].mean():+.2f}%")
    print(f" • 단순 보유(B&H) 평균 수익률: {df_res['bh_return'].mean():+.2f}%")
    print(f" • 평균 초과 수익 (알파): {df_res['strategy_return'].mean() - df_res['bh_return'].mean():+.2f}%p")
    win_cnt = (df_res['alpha'] > 0).sum()
    print(f" • B&H 초과 승리 종목: {win_cnt}개 / {len(df_res)}개 ({win_cnt / len(df_res) * 100:.1f}%)")
    print(f" • 평균 거래 횟수: {df_res['trades'].mean():.1f}회 (수수료 극소화)")
    print("=" * 95)


if __name__ == "__main__":
    target_tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_16_TICKERS
    execute_promising_1(target_tickers)
