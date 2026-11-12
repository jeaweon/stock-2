"""CLI Runner for Alpha Stock Screener & Investment Pipeline.

Scans the local cached stock universe (or custom tickers), identifies stocks
with the 'High-Return Champion DNA', and outputs ranked BUY_READY and WATCHLIST targets.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

from src.data_loader import load_stock_data, CACHE_DIR
from src.screener import screen_candidate_universe


def main():
    parser = argparse.ArgumentParser(description="고수익 주도주 DNA 스크리너 & 매수 후보 발굴 파이프라인")
    parser.add_argument("--min-score", type=float, default=65.0, help="최소 합격 점수 (기본: 65점)")
    parser.add_argument("--top", type=int, default=20, help="출력할 상위 종목 수")
    args = parser.parse_args()

    print("=" * 105)
    print("🔎 [고수익 주도주 챔피언 DNA 스크리너 & 매수 유니버스 발굴 파이프라인]")
    print("   기준: 10/30일선 정배열 + 200시간선 상단 + ATR 변동성 파워(≥1.1%) + 기관 수급 갭(≥35%)")
    print("=" * 105)

    # 로컬 캐시에 저장된 모든 종목 로드
    cache_files = list(CACHE_DIR.glob("*_1h_730d.csv"))
    if not cache_files:
        print("⚠️ 로컬 캐시에 주식 데이터가 없습니다. 먼저 백테스트를 실행해주세요.")
        return

    print(f"📂 로컬 데이터베이스에서 {len(cache_files)}개 종목 로드 및 퀀트 DNA 분석 중...")
    data_dict = {}
    for cf in cache_files:
        sym = cf.name.split("_")[0]
        try:
            df = load_stock_data(sym)
            if len(df) >= 200:
                data_dict[sym] = df
        except Exception:
            continue

    print(f"✅ 유효 분석 대상: {len(data_dict)}개 종목\n")

    ranked = screen_candidate_universe(data_dict, min_score=args.min_score)

    buy_ready = ranked[ranked["status"] == "BUY_READY"]
    watchlist = ranked[ranked["status"] == "WATCHLIST"]
    disqualified = ranked[ranked["status"] == "DISQUALIFIED"]

    print("=" * 105)
    print(f"🎯 [1순위: 실시간 즉시 매수 가능 후보 (BUY_READY)] - 총 {len(buy_ready)}개 종목")
    print("=" * 105)
    if not buy_ready.empty:
        cols = ["ticker", "score", "atr_pct", "gap_freq_pct", "dist_from_high_pct", "rationale"]
        print(buy_ready[cols].head(args.top).to_string(index=False))
    else:
        print("  현재 즉시 매수 타점에 도달한 종목이 없습니다 (신고가 근접 대기 필요).")

    print("\n" + "=" * 105)
    print(f"👀 [2순위: 유망 주도주 관심종목 풀 (WATCHLIST)] - 총 {len(watchlist)}개 종목")
    print("=" * 105)
    if not watchlist.empty:
        cols = ["ticker", "score", "atr_pct", "gap_freq_pct", "dist_from_high_pct", "rationale"]
        print(watchlist[cols].head(args.top).to_string(index=False))

    print("\n" + "=" * 105)
    print(f"📊 [스크리닝 종합 통계 요약]")
    print(f" • 총 분석 종목: {len(ranked)}개")
    print(f" • 🏆 BUY_READY (즉시 매수 진입): {len(buy_ready)}개 ({len(buy_ready)/len(ranked)*100:.1f}%)")
    print(f" • 👀 WATCHLIST (눌림목 대기): {len(watchlist)}개 ({len(watchlist)/len(ranked)*100:.1f}%)")
    print(f" • ❌ DISQUALIFIED (역배열/저변동성 탈락): {len(disqualified)}개 ({len(disqualified)/len(ranked)*100:.1f}%)")
    print("=" * 105)

    # Save to CSV for dashboard integration
    out_file = "screener_qualified_candidates.csv"
    ranked.to_csv(out_file, index=False)
    print(f"💾 스크리닝 결과가 '{out_file}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
