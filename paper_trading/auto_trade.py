"""Headless 1-Hour Automated Paper Trading Runner (GitHub Actions / Cron).

Loads account state from GitHub Gist, runs the 1-hour 157-ticker Champion DNA screener
and Promising Strategy #3 buy/sell/stop-loss cycle, and persists updated state back to Gist.
"""
import argparse
import os
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from engine import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE_KRW,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_SAVE_PATH,
    LIVE_SCAN_UNIVERSE_157,
    SCAN_UNIVERSES,
    SimulatedTrader,
    run_full_hourly_cycle,
)


def _write_step_summary(lines: list[str]) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="1-hour automated paper trading runner")
    parser.add_argument("--dry-run", action="store_true", help="Scan only without executing orders or saving Gist")
    parser.add_argument("--allow-local", action="store_true", help="Allow running without GIST_ID / GIST_TOKEN secrets")
    args = parser.parse_args()

    gist_id = os.getenv("GIST_ID", "").strip()
    gist_token = (os.getenv("GIST_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()

    if not args.dry_run and not args.allow_local and (not gist_id or not gist_token):
        id_state = "정상(설정됨)" if gist_id else "❌ 비어있음"
        tok_state = "정상(설정됨)" if gist_token else "❌ 비어있음"
        msg = (
            f"❌ GitHub Repository Secrets가 전달되지 않았습니다.\n"
            f"- `GIST_ID`: {id_state}\n"
            f"- `GIST_TOKEN`: {tok_state}\n"
            f"👉 Variables 탭이나 Environment secrets가 아닌 **Repository secrets**에 정확한 대문자 이름으로 저장한 뒤, "
            f"이전 실패 실행의 `Re-run` 대신 **Actions → Run workflow**로 새로 실행해 주세요."
        )
        print(msg)
        _write_step_summary(["## ⚠️ 자동매매 실행 중단", msg])
        return 1

    trader = SimulatedTrader(
        initial_balance=DEFAULT_INITIAL_BALANCE_KRW,
        save_file=str(DEFAULT_SAVE_PATH),
        max_positions=DEFAULT_MAX_POSITIONS,
        fee_rate=DEFAULT_FEE_RATE,
        gist_id=gist_id,
        github_token=gist_token,
    )

    if gist_id and gist_token:
        ok_gist, gist_msg = trader.check_gist_connection()
        print(f"[Gist Check] {gist_msg}")
        if not ok_gist and not args.allow_local:
            _write_step_summary(["## ❌ Gist 연결 실패", gist_msg])
            return 1

    universe = SCAN_UNIVERSES.get("🌐 전체 대표 풀 (157종목 · 추천)", LIVE_SCAN_UNIVERSE_157)
    res = run_full_hourly_cycle(
        trader=trader,
        scan_universe=universe,
        auto_trade=not args.dry_run,
        refresh_state_first=False,
    )

    summary_lines = [
        f"## ⏰ 1시간봉 자동 검색 & 매매 결과 ({res['executed_at_utc']})",
        f"- 검사 종목: **{res['scanned_count']}개** (실패 {res['failed_count']}개) | 매수 적격(BUY_READY): **{res['buy_ready_count']}개**",
        f"- 매도 체결: **{len(res['sell_executions'])}건** | 매수 체결: **{len(res['buy_executions'])}건** | 예산초과 스킵: **{len(res['skipped_over_budget'])}건**",
        f"- 예수금 잔고: **{int(round(trader.cash)):,} 원** | 보유 종목 수: **{len(trader.positions)} / {trader.max_positions}**",
    ]
    for m in res["sell_executions"]:
        summary_lines.append(f"  - {m}")
    for m in res["buy_executions"]:
        summary_lines.append(f"  - {m}")

    print("\n".join(summary_lines))
    _write_step_summary(summary_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
