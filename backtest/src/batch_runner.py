"""High-throughput multi-stock batch investigation engine.

Runs dynamic profiling, baseline backtesting, Dual-Gemini optimization,
and adversarial evaluation across any arbitrary list or universe of stocks.
"""
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from src.data_loader import load_stock_data
from src.backtester import run_backtest, BacktestResult
from src.gemini_agents import DualGeminiAgents
from src.adaptive_experiment import profile_asset, STRATEGY_MAP
from src.storage import save_history_record, load_history_records, DEFAULT_HISTORY_FILE


# Common Market Universes for instant mass scanning
POPULAR_UNIVERSES: Dict[str, List[str]] = {
    "mega_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "PLTR", "CRM", "ORCL"],
    "semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "MU", "SOXL", "ARM", "SMCI", "TXN"],
    "leverage_and_crypto": ["TQQQ", "SOXL", "CONL", "BITO", "MARA", "COIN", "ETHU", "UVXY", "LABU", "SQQQ"],
    "broad_and_global": ["SPY", "QQQ", "IWM", "DIA", "EWY", "FXI", "KORU", "GLD", "SLV", "USO", "EEM", "INDA"],
    "energy_and_defense": ["XOM", "CVX", "OXY", "CAT", "BA", "LMT", "GE", "RTX", "ENPH", "FSLR", "NEE"],
    "consumer_and_healthcare": ["COST", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "DIS", "SPOT", "LLY", "UNH", "JNJ", "PFE", "MRNA", "ABBV"],
    "financials_and_fintech": ["JPM", "V", "MA", "BAC", "GS", "PYPL", "HOOD", "SOFI", "BABA"],
    "all_80_universe": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "PLTR", "CRM", "ORCL",
        "AMD", "TSM", "AVGO", "QCOM", "MU", "INTC", "SOXL", "ARM", "SMCI", "TXN",
        "ETHU", "CONL", "MARA", "COIN", "BITO", "TQQQ", "UVXY", "LABU", "SQQQ", "RIVN",
        "JPM", "V", "MA", "BAC", "GS", "PYPL", "HOOD", "SOFI", "BABA",
        "CAT", "BA", "LMT", "GE", "RTX",
        "XOM", "CVX", "OXY", "ENPH", "FSLR", "NEE",
        "LLY", "UNH", "JNJ", "PFE", "MRNA", "ABBV", "BIIB",
        "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "DIS", "SPOT",
        "SPY", "QQQ", "IWM", "DIA", "EWY", "FXI", "KORU", "GLD", "SLV", "USO", "EEM", "INDA"
    ],
}


def run_batch_investigation(
    tickers: List[str],
    user_strategy_prompt: Optional[str] = None,
    use_adaptive_regime_strategy: bool = True,
    iterations_per_stock: int = 2,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-3.8-flash",
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> Dict[str, Any]:
    """Run mass investigation on an arbitrary list of stock tickers.

    For each ticker:
      1. Profiles asset dynamically (Group A, B, or C).
      2. Backtests baseline strategy (either user's prompt or adaptive group strategy).
      3. Proposes AI Optimizer improvements.
      4. Backtests improved strategy.
      5. Runs Adversarial Critic strict evaluation.
      6. Records to backtest_history.csv.
      7. Compiles detailed Before vs After metrics.
    """
    agents = DualGeminiAgents(api_key=gemini_api_key, model_name=gemini_model)
    comparison_records: List[Dict[str, Any]] = []
    total_count = len(tickers)

    for idx, raw_ticker in enumerate(tickers, start=1):
        ticker = raw_ticker.upper().strip()
        if progress_callback:
            progress_callback(idx, total_count, ticker, "데이터 수집 및 프로파일링 중...")

        try:
            df = load_stock_data(ticker, period="730d", interval="1h")
        except Exception as e:
            print(f"[{ticker}] 데이터 로드 실패: {e}")
            continue

        if len(df) < 50:
            print(f"[{ticker}] 데이터 부족 ({len(df)}개 봉)")
            continue

        # 1. Dynamic Asset Profiling
        prof = profile_asset(df, ticker)

        # 2. Determine Baseline Strategy
        if user_strategy_prompt:
            base_parse = agents.parse_user_strategy_to_code(user_strategy_prompt)
            base_func = base_parse["func"]
            base_code = base_parse["code"]
            base_desc = user_strategy_prompt
            base_strat_name = "사용자 원본 전략"
        else:
            # Use data-driven adaptive regime strategy as baseline
            strat_fn, s_name, s_desc = STRATEGY_MAP[prof.assigned_group]
            base_func = strat_fn
            base_code = "# Dynamic Adaptive Strategy " + prof.assigned_group
            base_desc = s_desc
            base_strat_name = f"적응형 {s_name}"

        # 3. Run Baseline Backtest
        base_res = run_backtest(df, signal_func=base_func, initial_capital=initial_capital, fee_rate=fee_rate)

        # Log Baseline to History
        save_history_record(
            ticker=ticker,
            iteration_round=0,
            strategy_name=f"{base_strat_name} (수정 전)",
            buy_rules=base_desc,
            sell_rules="기준 청산 규칙",
            position_sizing_rules="기본 산출",
            win_rate_pct=base_res.win_rate_pct,
            total_return_pct=base_res.total_return_pct,
            mdd_pct=base_res.max_drawdown_pct,
            total_trades=base_res.total_trades,
            profit_factor=base_res.profit_factor,
            sharpe_ratio=base_res.sharpe_ratio,
            critic_verdict="BASELINE",
            critic_feedback=f"수정 전 기준 지표 (그룹 {prof.assigned_group}: {prof.group_name})",
            strategy_code=base_code,
        )

        # 4. Optimization & Critic Refinement Loop
        best_res = base_res
        best_code = base_code
        best_rules = base_desc
        best_round = 0
        best_critic_verdict = "BASELINE"
        best_critic_score = 50
        best_critic_feedback = "기준 전략"

        current_metrics = base_res.to_dict()
        last_feedback = None

        for r in range(1, iterations_per_stock + 1):
            if progress_callback:
                progress_callback(idx, total_count, ticker, f"AI 개선 {r}차 최적화 및 비판 심사 중...")

            opt = agents.propose_improved_strategy(
                original_strategy_desc=base_desc,
                current_metrics=current_metrics,
                critic_feedback=last_feedback,
                iteration_round=r,
            )

            try:
                imp_res = run_backtest(
                    df,
                    signal_func=opt["func"],
                    initial_capital=initial_capital,
                    fee_rate=fee_rate,
                    position_sizing_func=opt.get("sizing_func"),  # AI 생성 수식, 로컬 1회 평가
                )
            except Exception as e:
                print(f"[{ticker}] {r}차 개선안 백테스트 에러: {e}")
                continue

            imp_metrics = imp_res.to_dict()

            # Critic Evaluation
            critic = agents.evaluate_with_adversarial_critic(
                strategy_desc=opt["improvements"],
                strategy_code=opt["code"],
                current_metrics=imp_metrics,
                baseline_metrics=base_res.to_dict(),
            )

            save_history_record(
                ticker=ticker,
                iteration_round=r,
                strategy_name=f"AI {r}차 개선 (수정 후)",
                buy_rules=opt["improvements"],
                sell_rules="AI 동적 청산",
                position_sizing_rules="동적 변동성 비중",
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

            if imp_res.total_return_pct > best_res.total_return_pct:
                best_res = imp_res
                best_code = opt["code"]
                best_rules = opt["improvements"]
                best_round = r
                best_critic_verdict = critic["verdict"]
                best_critic_score = critic["score"]
                best_critic_feedback = critic["full_critique"]

            current_metrics = imp_metrics
            last_feedback = critic["full_critique"]

            if critic["verdict"] == "PASS":
                break

        # 5. Compile Before vs After Comparison
        ret_delta = round(best_res.total_return_pct - base_res.total_return_pct, 2)
        win_delta = round(best_res.win_rate_pct - base_res.win_rate_pct, 2)
        mdd_delta = round(best_res.max_drawdown_pct - base_res.max_drawdown_pct, 2)  # negative is good
        pf_delta = round(best_res.profit_factor - base_res.profit_factor, 2)

        comparison_records.append({
            "ticker": ticker,
            "group": prof.assigned_group,
            "group_name": prof.group_name.split(":")[0],
            # Before AI feedback
            "before_return_pct": round(base_res.total_return_pct, 2),
            "before_win_rate_pct": round(base_res.win_rate_pct, 2),
            "before_mdd_pct": round(base_res.max_drawdown_pct, 2),
            "before_trades": base_res.total_trades,
            "before_profit_factor": round(base_res.profit_factor, 2),
            # After AI feedback
            "after_return_pct": round(best_res.total_return_pct, 2),
            "after_win_rate_pct": round(best_res.win_rate_pct, 2),
            "after_mdd_pct": round(best_res.max_drawdown_pct, 2),
            "after_trades": best_res.total_trades,
            "after_profit_factor": round(best_res.profit_factor, 2),
            # Deltas
            "return_improvement_pctp": ret_delta,
            "win_rate_improvement_pctp": win_delta,
            "mdd_change_pctp": mdd_delta,
            "profit_factor_improvement": pf_delta,
            # Critic
            "critic_verdict": best_critic_verdict,
            "critic_score": best_critic_score,
            "best_round": best_round,
            "best_rules_summary": best_rules.replace("\n", " ")[:120],
        })

    comp_df = pd.DataFrame(comparison_records)
    if not comp_df.empty:
        comp_df.to_csv("scanner_results.csv", index=False, encoding="utf-8-sig")

    return {
        "comparison_df": comp_df,
        "total_scanned": len(comp_df),
    }
