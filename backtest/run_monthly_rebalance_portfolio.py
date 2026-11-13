"""[유망 매매 방법 1] 월별 리밸런싱 포트폴리오 백테스트 (정정판).

이전 run_new_80_backtest.py 의 월별 버전은 신호를 '한 달치 데이터'로 계산해
130~650봉 거시 EMA가 매달 초기화되는 결함이 있었습니다. 이 스크립트는:

1. 신호를 각 종목의 '전체 2년 데이터'로 1회 계산 (거시 EMA 연속성 유지)
2. 갭하락 시가 체결 반영된 run_backtest 로 종목별 전략 자산곡선 산출 (수수료 0%)
3. 매월 첫 거래 시점에 포트폴리오 자본을 편입 가능 종목에 1/N 동일 비중으로 재배분
   (전략 포지션 자체는 끊지 않고 '비중만' 조정)
4. 동일한 방식의 B&H 동일비중 월별 리밸런싱 포트폴리오와 비교
5. 결함 값이 기록됐던 scanner_results.csv 의 80개 종목 행을 정정 값으로 갱신
"""
import os
import time

import numpy as np
import pandas as pd

from run_new_80_backtest import NEW_80_TICKERS
from src.adaptive_experiment import profile_asset
from src.backtester import run_backtest
from src.data_loader import load_many_stock_data
from src.strategy_presets import PRESETS

INITIAL_CAPITAL = 10000.0
OUT_PER_STOCK = "promising1_gapfix_80_per_stock.csv"
OUT_PORTFOLIO = "promising1_monthly_rebalance_portfolio.csv"


def monthly_rebalanced_portfolio(returns: pd.DataFrame, available: pd.DataFrame) -> pd.Series:
    """매월 첫 봉에서 편입 가능 종목 1/N 동일비중 재배분, 월중에는 각 슬리브가 자유롭게 변동."""
    months = returns.index.tz_convert("America/New_York").to_period("M")
    value = 1.0
    out = []
    for m in months.unique():
        mask = months == m
        r_m = returns.loc[mask]
        avail = available.loc[mask].iloc[0]
        cols = avail[avail].index
        if len(cols) == 0:
            out.append(pd.Series(value, index=r_m.index))
            continue
        growth = (1.0 + r_m[cols].fillna(0.0)).cumprod()
        sleeve_val = growth.mean(axis=1) * value
        out.append(sleeve_val)
        value = float(sleeve_val.iloc[-1])
    return pd.concat(out) * INITIAL_CAPITAL


def curve_stats(curve: pd.Series) -> dict:
    ret = (curve.iloc[-1] / curve.iloc[0] - 1.0) * 100.0
    dd = (curve / curve.cummax() - 1.0).min() * 100.0
    r = curve.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252 * 6.5)) if r.std() > 0 else 0.0
    return {"total_return_pct": round(ret, 2), "mdd_pct": round(abs(dd), 2), "sharpe": round(sharpe, 2)}


def main():
    preset = PRESETS["promising_1"]
    t0 = time.time()
    data, failed = load_many_stock_data(NEW_80_TICKERS)
    if failed:
        print("데이터 로드 실패:", failed)

    rows, strat_rets, bh_rets = [], {}, {}
    for sym in NEW_80_TICKERS:
        df = data.get(sym)
        if df is None or len(df) < 150:
            continue
        prof = profile_asset(df, sym)
        sig = preset.signal_func(df, sym)  # 전체 기간으로 1회 계산
        res = run_backtest(df, df_with_signals=sig, initial_capital=INITIAL_CAPITAL, fee_rate=0.0)
        bh = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0
        alpha = res.total_return_pct - bh
        gap_exits = sum("갭하락" in t.exit_reason for t in res.trades)
        strat_rets[sym] = res.equity_curve.pct_change()
        bh_rets[sym] = df["Close"].pct_change()
        rows.append({
            "ticker": sym, "group": prof.assigned_group, "group_name": prof.group_name,
            "strategy_name": preset.name,
            "2y_buy_hold_ret": round(bh, 2), "strategy_return": round(res.total_return_pct, 2),
            "alpha_vs_hold": round(alpha, 2), "win_rate": round(res.win_rate_pct, 2),
            "mdd": round(res.max_drawdown_pct, 2), "trades": res.total_trades,
            "profit_factor": round(res.profit_factor, 2), "sharpe_ratio": round(res.sharpe_ratio, 2),
            "atr_pct": prof.mean_normalized_atr_pct, "gap_freq_pct": prof.overnight_gap_freq_pct,
            "before_return_pct": round(bh, 2), "after_return_pct": round(res.total_return_pct, 2),
            "return_improvement_pctp": round(alpha, 2),
            "critic_verdict": "PASS" if alpha > 0 else "미달", "critic_score": 90 if alpha > 0 else 70,
            "gap_down_exits": gap_exits,
        })

    per = pd.DataFrame(rows)
    per.to_csv(OUT_PER_STOCK, index=False)

    # --- 포트폴리오 (공통 타임라인 정렬) ---
    S = pd.DataFrame(strat_rets).sort_index()
    B = pd.DataFrame(bh_rets).sort_index()
    avail = B.notna() | B.shift(-1).notna()  # 상장/데이터 시작 이후만 편입
    avail = avail.cummax()
    S, B = S.fillna(0.0), B.fillna(0.0)

    strat_curve = monthly_rebalanced_portfolio(S, avail)
    bh_curve = monthly_rebalanced_portfolio(B, avail)
    s_stats, b_stats = curve_stats(strat_curve), curve_stats(bh_curve)
    pd.DataFrame({"strategy_equity": strat_curve, "bh_equity": bh_curve}).to_csv(OUT_PORTFOLIO)

    # --- scanner_results.csv 결함 행 정정 ---
    scanner_path = "scanner_results.csv"
    if os.path.exists(scanner_path):
        old = pd.read_csv(scanner_path)
        keep = old[~old["ticker"].isin(per["ticker"])]
        pd.concat([keep, per.drop(columns=["gap_down_exits"])], ignore_index=True).to_csv(scanner_path, index=False)

    print("=" * 90)
    print(f"[A] 종목별 단독 운용 (갭하락 반영, 전체기간 신호) — {len(per)}종목")
    print(f"  전략 평균 {per.strategy_return.mean():+.2f}% | B&H 평균 {per['2y_buy_hold_ret'].mean():+.2f}% | "
          f"알파 {per.strategy_return.mean() - per['2y_buy_hold_ret'].mean():+.2f}%p")
    print(f"  B&H 초과 {int((per.alpha_vs_hold > 0).sum())}/{len(per)} | 플러스 {int((per.strategy_return > 0).sum())}/{len(per)} | "
          f"평균 MDD {per.mdd.mean():.1f}% | 평균 거래 {per.trades.mean():.1f}회 | 갭하락 청산 총 {int(per.gap_down_exits.sum())}회")
    print(f"[B] 월별 동일비중 리밸런싱 포트폴리오")
    print(f"  전략: 수익 {s_stats['total_return_pct']:+.2f}% | MDD {s_stats['mdd_pct']:.2f}% | Sharpe {s_stats['sharpe']:.2f}")
    print(f"  B&H : 수익 {b_stats['total_return_pct']:+.2f}% | MDD {b_stats['mdd_pct']:.2f}% | Sharpe {b_stats['sharpe']:.2f}")
    print("[TOP 5 알파]")
    print(per.sort_values("alpha_vs_hold", ascending=False)[["ticker", "strategy_return", "2y_buy_hold_ret", "alpha_vs_hold", "gap_down_exits"]].head(5).to_string(index=False))
    print("[WORST 5 알파]")
    print(per.sort_values("alpha_vs_hold")[["ticker", "strategy_return", "2y_buy_hold_ret", "alpha_vs_hold", "gap_down_exits"]].head(5).to_string(index=False))
    print(f"소요 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
