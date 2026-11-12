"""Script to execute the 3-Regime Multi-Asset Experiment across 12 diverse tickers."""
import pandas as pd
from src.adaptive_experiment import run_multi_asset_experiment_suite

TARGET_TICKERS = [
    # Group A: Trend Following & Big Tech
    "AAPL", "NVDA", "TQQQ", "SOXL", "MSFT",
    # Group B: Ultra-High Volatility & Crypto/Beta
    "ETHU", "CONL", "TSLA", "MARA",
    # Group C: Box-Range & Gap Risk (Asia/Small Cap)
    "KORU", "EWY", "FXI"
]

print("=========================================================================================")
print("🔬 [3-Regime Multi-Asset Experiment] 2년치 1시간 봉 기반 자산 분류 및 맞춤 전략 실험 시작")
print("=========================================================================================")
print(f"테스트 대상 자산군 (총 {len(TARGET_TICKERS)}개 종목): {TARGET_TICKERS}\n")

suite = run_multi_asset_experiment_suite(TARGET_TICKERS)
summary_df = suite["summary_df"]

print("\n" + "=" * 90)
print("📊 [실험 결과 종합 요약 테이블]")
print("=" * 90)

display_cols = [
    "ticker", "group", "2y_buy_hold_ret", "strategy_return",
    "alpha_vs_hold", "win_rate", "mdd", "trades", "profit_factor", "atr_pct", "gap_freq_pct"
]
print(summary_df[display_cols].to_string(index=False))

print("\n" + "=" * 90)
print("🔍 [그룹별 성과 분석 & 교차 검증 (Cross-testing)]")
print("=" * 90)

for t in ["TQQQ", "ETHU", "KORU", "NVDA", "CONL", "EWY"]:
    if t in suite["results"]:
        res = suite["results"][t]
        prof = res["profile"]
        m_name = res["matched_name"]
        m_res = res["matched_result"]
        print(f"\n▶ [{t}] 분류: {prof.group_name}")
        print(f"   특성 지표: 2년 보유수익률: {prof.total_return_pct:+.1f}%, ATR: {prof.mean_normalized_atr_pct}%, 갭빈도: {prof.overnight_gap_freq_pct}%")
        print(f"   [맞춤 배정 전략: {m_name}]")
        print(f"   -> 수익률: {m_res.total_return_pct:+.2f}% | 승률: {m_res.win_rate_pct:.1f}% | MDD: {m_res.max_drawdown_pct:.2f}% | 거래수: {m_res.total_trades}회 | PF: {m_res.profit_factor:.2f}")
        
        print(f"   [타 전략 적용 시 교차 비교 (Cross-Testing)]")
        for g_k, (s_n, c_res, is_m) in res["all_strategy_results"].items():
            if not is_m:
                diff_ret = m_res.total_return_pct - c_res.total_return_pct
                print(f"    - {s_n}: 수익률 {c_res.total_return_pct:+.2f}%, MDD {c_res.max_drawdown_pct:.2f}% (맞춤 전략 대비 {diff_ret:+.2f}%p 차이)")

# Save full experiment summary table
summary_df.to_csv("regime_experiment_summary.csv", index=False, encoding="utf-8-sig")
print("\n💾 실험 요약 결과가 'regime_experiment_summary.csv' 및 'backtest_history.csv'에 모두 저장되었습니다.")
