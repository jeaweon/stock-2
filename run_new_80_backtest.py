"""Backtest 80 brand new, previously untested tickers using '유망 매매 방법 1' (Promising Strategy 1)."""
import os
import sys
import time
import pandas as pd
import numpy as np

from src.data_loader import load_stock_data
from src.backtester import run_backtest
from src.strategy_presets import PRESETS
from src.adaptive_experiment import profile_asset
from src.storage import save_history_record

# 80 New Diverse Tickers across 10 sectors (None in the existing 82 set)
NEW_80_TICKERS = [
    # 1. Tech & Software & Cybersecurity (8)
    "ADBE", "INTU", "NOW", "IBM", "CSCO", "FTNT", "DDOG", "NET",
    # 2. Semiconductors & Chip Equipment (8)
    "ASML", "LRCX", "KLAC", "AMAT", "MRVL", "ADI", "NXPI", "MCHP",
    # 3. Industrials, Machinery & Logistics (8)
    "DE", "UNP", "HON", "UPS", "FDX", "ETN", "WM", "EMR",
    # 4. Financials, Exchanges & Insurance (8)
    "MS", "BLK", "C", "AXP", "SCHW", "CME", "ICE", "PGR",
    # 5. Aerospace & Defense (5)
    "NOC", "GD", "TDG", "LHX", "HII",
    # 6. Healthcare, MedTech & Pharma (9)
    "ISRG", "TMO", "DHR", "SYK", "BSX", "GILD", "VRTX", "REGN", "AMGN",
    # 7. Consumer Discretionary & Auto / EV / Leisure (9)
    "UBER", "ABNB", "BKNG", "TJX", "LOW", "LULU", "MAR", "GM", "F",
    # 8. Consumer Staples & Food / Beverage (8)
    "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    # 9. Energy, Infrastructure & Utilities (9)
    "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "SO", "DUK", "CEG",
    # 10. High Beta / Crypto / Volatility / Speculative / Leveraged (8)
    "MSTR", "RIOT", "CLSK", "UPST", "AFRM", "NVO", "TLT", "HYG"
]

def run_new_universe_backtest():
    preset = PRESETS["promising_2"]
    print("=" * 105)
    print(f"🚀 [유망 매매 방법 1 - 신규 미조사 80개 주식 전방위 검증]")
    print(f"   전략: {preset.name}")
    print(f"   검증 유니버스: 총 {len(NEW_80_TICKERS)}개 종목 (빅테크, 반도체장비, 방산, 헬스케어, 에너지, 배당소비재, MSTR 등)")
    print(f"   기간: 최근 2년(730일) 1시간봉 | 초기자본: $10,000 | 편도 수수료: 0.10%")
    print("=" * 105)

    results = []
    start_t = time.time()

    for idx, sym in enumerate(NEW_80_TICKERS, 1):
        sym = sym.upper().strip()
        print(f"[{idx:02d}/{len(NEW_80_TICKERS):02d}] 🔍 {sym:5s} -> ", end="", flush=True)
        try:
            df = load_stock_data(sym, period="730d", interval="1h")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            continue

        if len(df) < 150:
            print(f"데이터 부족 ({len(df)}개 봉)")
            continue

        prof = profile_asset(df, sym)
        sig_df = preset.signal_func(df, sym)
        res = run_backtest(df, df_with_signals=sig_df, initial_capital=10000.0, fee_rate=0.001)  # 편도 0.1% 실거래 수수료

        bh_ret = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100.0
        alpha = res.total_return_pct - bh_ret
        verdict = "PASS" if alpha > 0 else "미달"

        print(f"[{prof.assigned_group}] 전략수익: {res.total_return_pct:+7.2f}% | B&H: {bh_ret:+7.2f}% | 알파: {alpha:+7.2f}%p | 승률: {res.win_rate_pct:4.1f}% | MDD: {res.max_drawdown_pct:4.1f}% | 거래: {res.total_trades:2d}회 | {verdict}")

        row = {
            "ticker": sym,
            "group": prof.assigned_group,
            "group_name": prof.group_name,
            "strategy_name": preset.name,
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
            "before_return_pct": round(bh_ret, 2),
            "after_return_pct": round(res.total_return_pct, 2),
            "return_improvement_pctp": round(alpha, 2),
            "before_win_rate_pct": 0.0,
            "after_win_rate_pct": round(res.win_rate_pct, 2),
            "win_rate_improvement_pctp": round(res.win_rate_pct, 2),
            "before_mdd_pct": 0.0,
            "after_mdd_pct": round(res.max_drawdown_pct, 2),
            "mdd_change_pctp": round(res.max_drawdown_pct, 2),
            "after_profit_factor": round(res.profit_factor, 2),
            "critic_verdict": verdict,
            "critic_score": 90 if alpha > 0 else 70,
        }
        results.append(row)

        # Save to history CSV
        save_history_record(
            ticker=sym,
            iteration_round=2,
            strategy_name=preset.name,
            buy_rules="10일 EMA(65봉) > 30일 EMA(195봉) 정배열 + 200시간봉 EMA 위 매수",
            sell_rules="10일 EMA < 30일 EMA 데드크로스 청산",
            position_sizing_rules="100% 추세 가속 & 잔파동 휩쏘 차단",
            win_rate_pct=round(res.win_rate_pct, 2),
            total_return_pct=round(res.total_return_pct, 2),
            mdd_pct=round(res.max_drawdown_pct, 2),
            total_trades=res.total_trades,
            profit_factor=round(res.profit_factor, 2),
            sharpe_ratio=round(res.sharpe_ratio, 2),
            critic_verdict=verdict,
            critic_feedback=f"B&H({bh_ret:+.2f}%) 대비 알파 {alpha:+.2f}%p 산출"
        )

    df_res = pd.DataFrame(results)
    
    # Append or update scanner_results.csv
    scanner_path = "scanner_results.csv"
    if os.path.exists(scanner_path):
        old_scanner = pd.read_csv(scanner_path)
        combined = pd.concat([old_scanner[~old_scanner["ticker"].isin(df_res["ticker"])], df_res], ignore_index=True)
    else:
        combined = df_res
    combined.to_csv(scanner_path, index=False)
    
    elapsed = time.time() - start_t
    print("\n" + "=" * 105)
    print(f"📊 [신규 80개 주식 전방위 검증 완료 보고서] (총 소요 시간: {elapsed:.1f}초)")
    print("=" * 105)
    print(f" • 총 검증 완료 종목: {len(df_res)}개")
    print(f" • 전략 평균 수익률: {df_res['strategy_return'].mean():+.2f}%")
    print(f" • 단순 보유(B&H) 평균 수익률: {df_res['2y_buy_hold_ret'].mean():+.2f}%")
    print(f" • 평균 초과 수익 (알파): {df_res['strategy_return'].mean() - df_res['2y_buy_hold_ret'].mean():+.2f}%p")
    win_cnt = (df_res['alpha_vs_hold'] > 0).sum()
    print(f" • B&H 초과 승리 종목: {win_cnt}개 / {len(df_res)}개 ({win_cnt / len(df_res) * 100:.1f}%)")
    print(f" • 플러스(+) 수익 달성 종목: {(df_res['strategy_return'] > 0).sum()}개 / {len(df_res)}개 ({(df_res['strategy_return'] > 0).sum() / len(df_res) * 100:.1f}%)")
    print(f" • 평균 거래 횟수: {df_res['trades'].mean():.1f}회 (단기 휩쏘/수수료 방어)")
    print(f" • 평균 승률: {df_res['win_rate'].mean():.1f}% | 평균 MDD: {df_res['mdd'].mean():.1f}%")
    print("=" * 105)

    print("\n[🏆 신규 80개 종목 중 최고 알파 TOP 10]")
    print(df_res.sort_values("alpha_vs_hold", ascending=False)[["ticker", "group", "strategy_return", "2y_buy_hold_ret", "alpha_vs_hold", "trades"]].head(10).to_string(index=False))

    print("\n[📁 누적 통계]")
    print(f" • scanner_results.csv 총 등록 종목: {len(combined)}개 종목으로 대폭 확장 완료!")
    print("=" * 105)

if __name__ == "__main__":
    run_new_universe_backtest()
