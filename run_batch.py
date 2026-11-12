"""Command-line script to run mass stock investigations with Before vs After comparison."""
import argparse
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.batch_runner import run_batch_investigation, POPULAR_UNIVERSES


def print_progress(current: int, total: int, ticker: str, stage: str):
    print(f"[{current}/{total}] 🔍 {ticker:6s} -> {stage}")


def main():
    parser = argparse.ArgumentParser(description="대규모 종목 자동 조사 및 AI 피드백 전후 성과 비교기")
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="쉼표로 구분된 종목 티커 목록 (예: AAPL,NVDA,TSLA,AMD,SOXL,ETHU,KORU)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        choices=list(POPULAR_UNIVERSES.keys()) + [""],
        help="사전 정의된 시장 유니버스 (mega_tech, semiconductors, leverage_and_crypto, broad_and_global, sp500_sample_20)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="",
        help="사용자 정의 자연어 매수/매도 전략 (비워두면 AI 자산 적응형 전략 적용)",
    )
    parser.add_argument("--iterations", type=int, default=2, help="종목당 AI 개선 반복 횟수 (기본: 2)")
    parser.add_argument("--capital", type=float, default=10000.0, help="초기 자본 ($)")
    parser.add_argument("--fee", type=float, default=0.001, help="거래 수수료 비율 (0.001 = 0.1%%)")

    args = parser.parse_args()

    # Determine tickers list
    ticker_list = []
    if args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.universe and args.universe in POPULAR_UNIVERSES:
        ticker_list = POPULAR_UNIVERSES[args.universe]
    else:
        # Default: 12 diverse test assets
        ticker_list = ["AAPL", "NVDA", "TQQQ", "SOXL", "MSFT", "ETHU", "CONL", "TSLA", "MARA", "KORU", "EWY", "FXI"]

    print("=" * 95)
    print(f"🚀 [대규모 종목 조사 시작] 총 {len(ticker_list)}개 종목: {', '.join(ticker_list)}")
    print(f"   초기자본: ${args.capital:,.0f} | 편도수수료: {args.fee*100:.2f}% | AI 개선 반복: {args.iterations}회")
    print("=" * 95)

    res = run_batch_investigation(
        tickers=ticker_list,
        user_strategy_prompt=args.strategy if args.strategy else None,
        iterations_per_stock=args.iterations,
        initial_capital=args.capital,
        fee_rate=args.fee,
        progress_callback=print_progress,
    )

    df = res["comparison_df"]
    if df.empty:
        print("❌ 실행된 결과가 없습니다. 티커 목록을 확인해주세요.")
        sys.exit(1)

    print("\n" + "=" * 115)
    print("📊 [AI 피드백 전후 성과 비교 종합 테이블 (Before vs After Comparison)]")
    print("=" * 115)

    headers = [
        "종목", "그룹", "수익률(전)", "수익률(후)", "수익률 개선",
        "승률(전)", "승률(후)", "승률 개선", "MDD(전)", "MDD(후)", "거래수(전/후)", "심사"
    ]
    
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["ticker"],
            r["group"],
            f"{r['before_return_pct']:+.1f}%",
            f"{r['after_return_pct']:+.1f}%",
            f"{r['return_improvement_pctp']:+.1f}%p",
            f"{r['before_win_rate_pct']:.1f}%",
            f"{r['after_win_rate_pct']:.1f}%",
            f"{r['win_rate_improvement_pctp']:+.1f}%p",
            f"{r['before_mdd_pct']:.1f}%",
            f"{r['after_mdd_pct']:.1f}%",
            f"{r['before_trades']}/{r['after_trades']}",
            f"{r['critic_verdict']}({r['critic_score']})",
        ])

    comp_table = pd.DataFrame(rows, columns=headers)
    print(comp_table.to_string(index=False))

    print("\n" + "=" * 115)
    print("📈 [전체 요약 통계]")
    print("=" * 115)
    print(f" • 총 조사 완료 종목: {len(df)}개")
    print(f" • 평균 수익률 변화: {df['before_return_pct'].mean():+.2f}%  ->  {df['after_return_pct'].mean():+.2f}% ({df['return_improvement_pctp'].mean():+.2f}%p 개선)")
    print(f" • 평균 승률 변화: {df['before_win_rate_pct'].mean():.1f}%  ->  {df['after_win_rate_pct'].mean():.1f}% ({df['win_rate_improvement_pctp'].mean():+.1f}%p 개선)")
    print(f" • 평균 MDD 변화: {df['before_mdd_pct'].mean():.1f}%  ->  {df['after_mdd_pct'].mean():.1f}% ({df['mdd_change_pctp'].mean():+.1f}%p 변화)")
    print(f" • 최고 수익률 달성 종목: {df.loc[df['after_return_pct'].idxmax()]['ticker']} ({df['after_return_pct'].max():+.2f}%)")
    print(f" • 최대 수익률 개선 종목: {df.loc[df['return_improvement_pctp'].idxmax()]['ticker']} ({df['return_improvement_pctp'].max():+.2f}%p 개선)")

    print("\n💾 상세 결과는 'scanner_results.csv' 및 'backtest_history.csv'에 영구 저장되었습니다.")
    print("🌐 Streamlit UI에서 대시보드로 인터랙티브 차트 및 개별 타점을 확인하려면:")
    print("   streamlit run app.py (또는 http://localhost:8502 브라우저 새로고침)\n")


if __name__ == "__main__":
    main()
