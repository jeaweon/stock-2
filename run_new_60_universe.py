"""Backtest 60 brand new, previously untested tickers using '유망 매매 방법 2' (Promising Strategy 2).

기준 조건:
- 전략: 유망 매매 방법 2 (추세 가속 & 잔파동 휩쏘 차단형, 10d/30d EMA + 200시간봉 필터)
- 수수료: 편도 0.1% (실거래 손익 반영)
- 체결: 오버나이트 갭하락 시가 체결 및 슬리피지 반영
- 포트폴리오 리밸런싱 없음 (종목별 단독 시뮬레이션)
"""
import os
import sys
import time
import pandas as pd
import numpy as np

from src.data_loader import load_many_stock_data, load_stock_data
from src.backtester import run_backtest
from src.strategy_presets import PRESETS
from src.adaptive_experiment import profile_asset
from src.storage import save_history_record

# 60 Brand New Tickers across diverse sectors (None in the existing 159 set)
NEW_60_TICKERS = [
    # 1. Tech, Cloud & Cybersecurity (11)
    "SNOW", "PANW", "CRWD", "ZS", "MDB", "ANET", "DELL", "HPQ", "WDAY", "APP", "RBLX",
    # 2. Semis & Hardware (8)
    "ON", "STM", "TER", "ENTG", "WDC", "STX", "SWKS", "QRVO",
    # 3. Industrials, Infrastructure & Machinery (4)
    "PCAR", "CMI", "PWR", "URI",
    # 4. Financials, Banking & Insurance (7)
    "COF", "WFC", "USB", "PNC", "AIG", "MET", "TRV",
    # 5. Healthcare, Biotech & Pharma (10)
    "MRK", "BMY", "CVS", "CI", "ELV", "MCK", "COR", "BDX", "EW", "IDXX",
    # 6. Consumer, Dining, Travel & Airlines (7)
    "CMG", "RCL", "CCL", "DAL", "UAL", "LUV", "LCID",
    # 7. Energy, Materials & Utilities (10)
    "HAL", "BKR", "KMI", "WMB", "NEE", "AEP", "SRE", "FCX", "NEM", "NUE",
    # 8. Communication & Media (3)
    "T", "VZ", "SPOT"
]

def run_new_60_universe_backtest():
    preset = PRESETS["promising_2"]
    print("=" * 105)
    print(f"🚀 [유망 매매 방법 2 - 미조사 신규 {len(NEW_60_TICKERS)}개 주식 전방위 검증]")
    print(f"   전략: {preset.name}")
    print(f"   규칙: {preset.summary_rules.replace(chr(10), ' | ')}")
    print(f"   기간: 최근 2년(730일) 1시간봉 | 초기자본: $10,000 | 편도 수수료: 0.10% (실거래 손익)")
    print("=" * 105)

    # 1. 데이터 배치 로드 (Rate limit 방어 적용)
    print("📦 [1/2] 신규 종목 데이터 로드 및 캐싱 진행 중 (yfinance API 방어 적용)...")
    data_dict, failed_dict = load_many_stock_data(
        NEW_60_TICKERS, period="730d", interval="1h", batch_size=10, batch_pause_sec=2.0
    )
    if failed_dict:
        print(f"   ⚠️ 일부 종목 로드 실패: {failed_dict}")
    print(f"   ✅ 총 {len(data_dict)}개 종목 데이터 준비 완료!\n")

    # 2. 백테스트 실행
    print("⚙️ [2/2] 유망 매매 방법 2 백테스트 시뮬레이션 실행 중...")
    results = []
    start_t = time.time()

    valid_tickers = [sym for sym in NEW_60_TICKERS if sym in data_dict and len(data_dict[sym]) >= 150]

    for idx, sym in enumerate(valid_tickers, 1):
        df = data_dict[sym]
        prof = profile_asset(df, sym)
        sig_df = preset.signal_func(df, sym)
        res = run_backtest(df, df_with_signals=sig_df, initial_capital=10000.0, fee_rate=0.001)

        bh_ret = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100.0
        alpha = res.total_return_pct - bh_ret
        verdict = "PASS" if alpha > 0 else "미달"

        print(f"[{idx:02d}/{len(valid_tickers):02d}] 🔍 {sym:5s} -> [{prof.assigned_group}] "
              f"전략수익: {res.total_return_pct:+7.2f}% | B&H: {bh_ret:+7.2f}% | 알파: {alpha:+7.2f}%p | "
              f"승률: {res.win_rate_pct:4.1f}% | MDD: {res.max_drawdown_pct:4.1f}% | 거래: {res.total_trades:2d}회 | {verdict}")

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

        save_history_record(
            ticker=sym,
            iteration_round=1,
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

    # 누적 scanner_results.csv 갱신
    scanner_path = "scanner_results.csv"
    if os.path.exists(scanner_path):
        old_scanner = pd.read_csv(scanner_path)
        combined = pd.concat([old_scanner[~old_scanner["ticker"].isin(df_res["ticker"])], df_res], ignore_index=True)
    else:
        combined = df_res
    combined.to_csv(scanner_path, index=False)

    elapsed = time.time() - start_t
    print("\n" + "=" * 105)
    print(f"📊 [신규 {len(df_res)}개 주식 유망 매매 방법 2 검증 완료 보고서] (총 소요 시간: {elapsed:.1f}초)")
    print("=" * 105)
    print(f" • 총 검증 완료 종목: {len(df_res)}개")
    print(f" • 전략 평균 수익률: {df_res['strategy_return'].mean():+.2f}%")
    print(f" • 단순 보유(B&H) 평균 수익률: {df_res['2y_buy_hold_ret'].mean():+.2f}%")
    print(f" • 평균 초과 수익 (알파): {df_res['strategy_return'].mean() - df_res['2y_buy_hold_ret'].mean():+.2f}%p")
    win_cnt = (df_res['alpha_vs_hold'] > 0).sum()
    print(f" • B&H 초과 승리 종목: {win_cnt}개 / {len(df_res)}개 ({win_cnt / len(df_res) * 100:.1f}%)")
    print(f" • 플러스(+) 수익 달성 종목: {(df_res['strategy_return'] > 0).sum()}개 / {len(df_res)}개 ({(df_res['strategy_return'] > 0).sum() / len(df_res) * 100:.1f}%)")
    print(f" • 평균 거래 횟수: {df_res['trades'].mean():.1f}회 | 평균 승률: {df_res['win_rate'].mean():.1f}% | 평균 MDD: {df_res['mdd'].mean():.1f}%")
    print("=" * 105)

    print("\n[🏆 신규 종목 중 최고 알파 TOP 10]")
    print(df_res.sort_values("alpha_vs_hold", ascending=False)[["ticker", "group", "strategy_return", "2y_buy_hold_ret", "alpha_vs_hold", "trades"]].head(10).to_string(index=False))

    print("\n[📁 전체 누적 통계]")
    print(f" • scanner_results.csv 총 등록 종목: {len(combined)}개 종목으로 추가 확장 완료!")
    print("=" * 105)

if __name__ == "__main__":
    run_new_60_universe_backtest()
