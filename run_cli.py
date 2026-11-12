"""Command-line interface runner for stock backtester and dual-Gemini AI optimizer."""
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_stock_data
from src.backtester import run_backtest
from src.gemini_agents import DualGeminiAgents
from src.storage import save_history_record, load_history_records, DEFAULT_HISTORY_FILE


def main():
    parser = argparse.ArgumentParser(description="미국 주식 1시간 봉 백테스터 & 듀얼 제미나이 최적화")
    parser.add_argument("--ticker", type=str, default="AAPL", help="주식 티커 심볼 (예: AAPL, NVDA, TSLA, SPY)")
    parser.add_argument("--capital", type=float, default=10000.0, help="초기 자본 ($)")
    parser.add_argument("--fee", type=float, default=0.001, help="거래 수수료/슬리피지 비율 (0.001 = 0.1%%)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="단기 20시간 이평선이 장기 50시간 이평선을 상향 돌파 시 매수, 하향 돌파 시 매도. 3% 손절, 6% 익절. AI 변동성 분석 기반 동적 투자액 산출.",
        help="자연어 매수/매도 전략 설명",
    )
    parser.add_argument("--iterations", type=int, default=3, help="AI 개선 반복 최대 횟수 (1~5)")
    parser.add_argument("--force-reload", action="store_true", help="캐시 무시하고 데이터 새로고침")

    args = parser.parse_args()

    print(f"\n========================================================")
    print(f"📈 [1시간 봉 백테스터] 종목: {args.ticker} | 초기자본: ${args.capital:,.0f} | 수수료: {args.fee*100:.2f}%")
    print(f"========================================================")

    # 1. Load data
    print(f"📥 1시간 봉 최근 2년치(최대 730일) 데이터 로딩 중...")
    try:
        df = load_stock_data(ticker=args.ticker, period="730d", interval="1h", force_reload=args.force_reload)
        print(f"✅ 총 {len(df):,}개 1시간 봉 로드 완료 ({df.index[0]} ~ {df.index[-1]})")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        sys.exit(1)

    agents = DualGeminiAgents()

    # 2. Parse User Strategy
    print(f"\n🤖 사용자 전략 분석 및 코드 생성 중...\n입력 전략: {args.strategy}")
    try:
        parsed = agents.parse_user_strategy_to_code(args.strategy)
        baseline_res = run_backtest(df, signal_func=parsed["func"], initial_capital=args.capital, fee_rate=args.fee)
    except Exception as e:
        print(f"❌ 원본 전략 실행 실패: {e}")
        sys.exit(1)

    print(f"\n📌 [원본 전략 백테스트 성과]")
    print(f" - 총 수익률: {baseline_res.total_return_pct:+.2f}%")
    print(f" - 승률: {baseline_res.win_rate_pct:.1f}% ({baseline_res.winning_trades}승 / {baseline_res.losing_trades}패)")
    print(f" - 최대 낙폭 (MDD): {baseline_res.max_drawdown_pct:.2f}%")
    print(f" - 손익 팩터 (PF): {baseline_res.profit_factor:.2f}")
    print(f" - 총 거래 횟수: {baseline_res.total_trades}회")
    print(f" - 최종 자산: ${baseline_res.final_equity:,.2f}")

    # Save baseline to CSV
    save_history_record(
        ticker=args.ticker,
        iteration_round=0,
        strategy_name="사용자 원본 전략",
        buy_rules=args.strategy,
        sell_rules=args.strategy,
        position_sizing_rules="AI 동적 산출",
        win_rate_pct=baseline_res.win_rate_pct,
        total_return_pct=baseline_res.total_return_pct,
        mdd_pct=baseline_res.max_drawdown_pct,
        total_trades=baseline_res.total_trades,
        profit_factor=baseline_res.profit_factor,
        sharpe_ratio=baseline_res.sharpe_ratio,
        critic_verdict="BASELINE",
        critic_feedback="기준 원본 전략",
        strategy_code=parsed["code"],
    )

    # 3. Dual-Gemini Optimization Loop
    current_metrics = baseline_res.to_dict()
    last_feedback = None

    for r in range(1, args.iterations + 1):
        print(f"\n--------------------------------------------------------")
        print(f"🔄 [듀얼 제미나이 루프] 라운드 {r} / {args.iterations}")
        print(f"--------------------------------------------------------")

        # AI 1: Optimizer
        print(f"🤖 [AI 1: Optimizer] 개선 대안 및 전략 코드 설계 중...")
        opt = agents.propose_improved_strategy(
            original_strategy_desc=args.strategy,
            current_metrics=current_metrics,
            critic_feedback=last_feedback,
            iteration_round=r,
        )
        print(f"💡 개선 내용: {opt['improvements']}")

        # Backtest improved
        imp_res = run_backtest(df, signal_func=opt["func"], initial_capital=args.capital, fee_rate=args.fee)
        imp_metrics = imp_res.to_dict()
        print(f"📊 개선안 지표 -> 수익률: {imp_res.total_return_pct:+.2f}%, 승률: {imp_res.win_rate_pct:.1f}%, MDD: {imp_res.max_drawdown_pct:.2f}%, 거래수: {imp_res.total_trades}회")

        # AI 2: Critic
        print(f"🛡️ [AI 2: Critic] 객관적 리스크 심사 중 (과최적화/표본 검증)...")
        critic = agents.evaluate_with_adversarial_critic(
            strategy_desc=opt["improvements"],
            strategy_code=opt["code"],
            current_metrics=imp_metrics,
            baseline_metrics=baseline_res.to_dict(),
        )
        print(f"⚖️ 심사 판정: [{critic['verdict']}] (점수: {critic['score']}/100)")
        print(f"📝 심사위원 의견:\n{critic['full_critique']}")

        # Save to CSV
        save_history_record(
            ticker=args.ticker,
            iteration_round=r,
            strategy_name=f"AI {r}차 개선 대안",
            buy_rules=opt["improvements"],
            sell_rules="AI 동적 청산",
            position_sizing_rules="동적 변동성/신호 비중",
            win_rate_pct=imp_res.win_rate_pct,
            total_return_pct=imp_res.total_return_pct,
            mdd_pct=imp_res.max_drawdown_pct,
            total_trades=imp_res.total_trades,
            profit_factor=imp_res.profit_factor,
            sharpe_ratio=imp_res.sharpe_ratio,
            critic_verdict=critic["verdict"],
            critic_feedback=critic["full_critique"],
            strategy_code=opt["code"],
        )

        current_metrics = imp_metrics
        last_feedback = critic["full_critique"]

        if critic["verdict"] == "PASS":
            print(f"\n🎉 {r}차 개선안이 Critic 심사를 성공적으로 통과(PASS)했습니다!")
            break

    print(f"\n💾 모든 백테스트 및 평가 이력이 '{DEFAULT_HISTORY_FILE}'에 저장되었습니다.")
    print(f"🌐 웹 대시보드로 시각화 차트를 보려면: streamlit run app.py\n")


if __name__ == "__main__":
    main()
