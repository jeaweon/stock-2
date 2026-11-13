"""Paper Trading (모의 투자) & Trading Journal Engine for Champion DNA 1h Strategy.

Supports both:
- KRW Live Paper Trading v2 (default 10,000,000 KRW, live USD/KRW FX, max 20 integer-share slots,
  full entry/exit rationale & metrics capture for the Trading Journal).
- Legacy USD functions (preserved for backward compatibility with existing unit tests).
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.timeframes import ema_days, ema_hours
from src.adaptive_experiment import compute_atr
from src.screener import evaluate_stock_dna

DATA_DIR = Path("data")
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"

DEFAULT_INITIAL_CAPITAL_KRW: float = 10_000_000.0
DEFAULT_FEE_RATE: float = 0.001
DEFAULT_MAX_POSITIONS: int = 20
DEFAULT_STOP_LOSS_PCT: float = 0.12
DEFAULT_FX_RATE: float = 1400.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_hours_diff(start_iso: str, end_iso: str) -> float:
    try:
        t1 = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        return max(0.0, round((t2 - t1).total_seconds() / 3600.0, 1))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# KRW Schema v2 (10,000,000 KRW Live Paper Trading)
# ---------------------------------------------------------------------------

def get_default_krw_portfolio(
    initial_capital_krw: float = DEFAULT_INITIAL_CAPITAL_KRW,
    fee_rate: float = DEFAULT_FEE_RATE,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    fx_rate: float = DEFAULT_FX_RATE,
) -> Dict[str, Any]:
    """Return a fresh KRW-based portfolio starting at 10,000,000 KRW."""
    now_str = _now_iso()
    cap = float(initial_capital_krw)
    return {
        "schema_version": 2,
        "currency": "KRW",
        "initial_capital_krw": cap,
        "cash_krw": cap,
        "fee_rate": float(fee_rate),
        "max_positions": int(max_positions),
        "last_fx_rate": float(fx_rate),
        "positions": {},
        "trade_history": [],
        "scan_logs": [],
        "equity_history": [
            {
                "timestamp": now_str,
                "total_equity_krw": cap,
                "cash_krw": cap,
                "invested_krw": 0.0,
                "fx_rate": float(fx_rate),
            }
        ],
        "created_at": now_str,
        "updated_at": now_str,
    }


def ensure_krw_portfolio_schema(
    port: Dict[str, Any],
    initial_capital_krw: float = DEFAULT_INITIAL_CAPITAL_KRW,
    fee_rate: float = DEFAULT_FEE_RATE,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    fx_rate: float = DEFAULT_FX_RATE,
) -> Dict[str, Any]:
    """Ensure loaded dict conforms to KRW schema v2 (upgrades empty legacy USD schema)."""
    if not isinstance(port, dict):
        return get_default_krw_portfolio(initial_capital_krw, fee_rate, max_positions, fx_rate)

    if port.get("schema_version") != 2 or port.get("currency") != "KRW":
        # Legacy empty USD portfolio -> cleanly initialize as 10,000,000 KRW
        if not port.get("positions") and not port.get("trade_history"):
            return get_default_krw_portfolio(initial_capital_krw, fee_rate, max_positions, fx_rate)

    port.setdefault("schema_version", 2)
    port.setdefault("currency", "KRW")
    port.setdefault("initial_capital_krw", float(initial_capital_krw))
    port.setdefault("cash_krw", float(initial_capital_krw))
    port.setdefault("fee_rate", float(fee_rate))
    port.setdefault("max_positions", int(max_positions))
    port.setdefault("last_fx_rate", float(fx_rate))
    port.setdefault("positions", {})
    port.setdefault("trade_history", [])
    port.setdefault("scan_logs", [])
    port.setdefault("equity_history", [])
    port.setdefault("created_at", _now_iso())
    port.setdefault("updated_at", _now_iso())
    return port


def load_krw_portfolio(
    filepath: Path = PORTFOLIO_FILE,
    initial_capital_krw: float = DEFAULT_INITIAL_CAPITAL_KRW,
    fee_rate: float = DEFAULT_FEE_RATE,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    fx_rate: float = DEFAULT_FX_RATE,
) -> Dict[str, Any]:
    """Load KRW portfolio state from JSON file, initializing or upgrading if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        port = get_default_krw_portfolio(initial_capital_krw, fee_rate, max_positions, fx_rate)
        save_portfolio(port, filepath=filepath)
        return port

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        port = ensure_krw_portfolio_schema(
            raw,
            initial_capital_krw=initial_capital_krw,
            fee_rate=fee_rate,
            max_positions=max_positions,
            fx_rate=fx_rate,
        )
        return port
    except Exception:
        port = get_default_krw_portfolio(initial_capital_krw, fee_rate, max_positions, fx_rate)
        save_portfolio(port, filepath=filepath)
        return port


def reset_krw_portfolio(
    initial_capital_krw: float = DEFAULT_INITIAL_CAPITAL_KRW,
    fee_rate: float = DEFAULT_FEE_RATE,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    fx_rate: float = DEFAULT_FX_RATE,
    filepath: Path = PORTFOLIO_FILE,
) -> Dict[str, Any]:
    """Reset KRW portfolio back to fresh initial capital (10,000,000 KRW)."""
    port = get_default_krw_portfolio(initial_capital_krw, fee_rate, max_positions, fx_rate)
    save_portfolio(port, filepath=filepath)
    return port


def _record_krw_equity_snapshot(portfolio: Dict[str, Any], timestamp: str, fx_rate: float) -> None:
    invested_krw = sum(
        pos["shares"] * pos["current_price_usd"] * fx_rate
        for pos in portfolio.get("positions", {}).values()
    )
    cash_krw = float(portfolio.get("cash_krw", 0.0))
    total_equity_krw = round(cash_krw + invested_krw, 0)
    portfolio.setdefault("equity_history", []).append({
        "timestamp": timestamp,
        "total_equity_krw": total_equity_krw,
        "cash_krw": round(cash_krw, 0),
        "invested_krw": round(invested_krw, 0),
        "fx_rate": round(float(fx_rate), 2),
    })
    if len(portfolio["equity_history"]) > 1000:
        portfolio["equity_history"] = portfolio["equity_history"][-1000:]


def execute_krw_buy(
    portfolio: Dict[str, Any],
    ticker: str,
    price_usd: float,
    fx_rate: float,
    allocation_krw: float,
    reason: str = "🏆 챔피언 DNA 1시간봉 조건 충족 매수",
    indicators: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> Tuple[bool, str]:
    """Execute a KRW-based integer-share simulated buy order on a US stock."""
    sym = ticker.upper().strip()
    if price_usd <= 0.0 or fx_rate <= 0.0:
        return False, f"유효하지 않은 시세/환율입니다: ${price_usd}, ₩{fx_rate}"

    if sym in portfolio.get("positions", {}):
        return False, f"이미 보유 중인 종목입니다: {sym}"

    fee_rate = float(portfolio.get("fee_rate", DEFAULT_FEE_RATE))
    cash_krw = float(portfolio.get("cash_krw", 0.0))
    budget_krw = min(cash_krw, float(allocation_krw))

    unit_total_krw = price_usd * fx_rate * (1.0 + fee_rate)
    shares = int(budget_krw // unit_total_krw)
    if shares <= 0:
        return (
            False,
            f"예산 초과(정수 1주 불가): 1주 필요액 ₩{unit_total_krw:,.0f} > 배분 예산 ₩{budget_krw:,.0f}",
        )

    # Guard against float rounding pushing total deduction slightly past cash_krw
    while shares > 0:
        trade_cost_krw = shares * price_usd * fx_rate
        fee_krw = trade_cost_krw * fee_rate
        total_deduction_krw = trade_cost_krw + fee_krw
        if total_deduction_krw <= cash_krw + 1e-6:
            break
        shares -= 1

    if shares <= 0:
        return False, f"수수료 포함 예수금 부족 (보유: ₩{cash_krw:,.0f})"

    portfolio["cash_krw"] = round(max(0.0, cash_krw - total_deduction_krw), 0)
    portfolio["last_fx_rate"] = round(float(fx_rate), 2)

    now_ts = timestamp or _now_iso()
    stop_loss_usd = round(price_usd * (1.0 - stop_loss_pct), 2)
    portfolio["positions"][sym] = {
        "ticker": sym,
        "shares": shares,
        "entry_price_usd": round(float(price_usd), 4),
        "entry_fx_rate": round(float(fx_rate), 2),
        "entry_cost_krw": round(trade_cost_krw, 0),
        "entry_fee_krw": round(fee_krw, 0),
        "entry_time": now_ts,
        "current_price_usd": round(float(price_usd), 4),
        "current_fx_rate": round(float(fx_rate), 2),
        "stop_loss_price_usd": stop_loss_usd,
        "highest_price_usd": round(float(price_usd), 4),
        "current_value_krw": round(trade_cost_krw, 0),
        "unrealized_pnl_krw": round(-fee_krw * 2.0, 0),
        "unrealized_pnl_pct": round((-fee_krw * 2.0) / (trade_cost_krw + fee_krw) * 100.0, 2),
        "price_return_pct": 0.0,
        "entry_reason": reason,
        "entry_indicators": indicators or {},
    }

    _record_krw_equity_snapshot(portfolio, now_ts, fx_rate)
    return (
        True,
        f"✅ [{sym}] {shares}주 매수 체결 (@${price_usd:,.2f}, 환율 ₩{fx_rate:,.1f}, "
        f"매수금 ₩{trade_cost_krw:,.0f}, 수수료 ₩{fee_krw:,.0f})",
    )


def execute_krw_sell(
    portfolio: Dict[str, Any],
    ticker: str,
    price_usd: float,
    fx_rate: float,
    reason: str = "청산",
    exit_indicators: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Tuple[bool, str]:
    """Execute a KRW-based sell order and append a complete record to the Trading Journal."""
    sym = ticker.upper().strip()
    positions = portfolio.get("positions", {})
    if sym not in positions:
        return False, f"보유하고 있지 않은 종목입니다: {sym}"
    if price_usd <= 0.0 or fx_rate <= 0.0:
        return False, f"유효하지 않은 청산 가격/환율입니다: ${price_usd}, ₩{fx_rate}"

    pos = positions[sym]
    shares = int(pos["shares"])
    entry_price_usd = float(pos["entry_price_usd"])
    entry_fx_rate = float(pos.get("entry_fx_rate", fx_rate))
    entry_cost_krw = float(pos.get("entry_cost_krw", shares * entry_price_usd * entry_fx_rate))
    entry_fee_krw = float(pos.get("entry_fee_krw", 0.0))

    fee_rate = float(portfolio.get("fee_rate", DEFAULT_FEE_RATE))
    gross_proceeds_krw = shares * price_usd * fx_rate
    exit_fee_krw = gross_proceeds_krw * fee_rate
    net_proceeds_krw = gross_proceeds_krw - exit_fee_krw
    total_fees_krw = entry_fee_krw + exit_fee_krw

    invested_total_krw = entry_cost_krw + entry_fee_krw
    net_pnl_krw = round(net_proceeds_krw - invested_total_krw, 0)
    gross_pnl_krw = round(gross_proceeds_krw - entry_cost_krw, 0)
    pnl_pct = round((net_pnl_krw / invested_total_krw) * 100.0, 2) if invested_total_krw > 0 else 0.0
    price_return_pct = round((price_usd - entry_price_usd) / entry_price_usd * 100.0, 2)

    now_ts = timestamp or _now_iso()
    holding_hours = _parse_iso_hours_diff(pos.get("entry_time", now_ts), now_ts)

    portfolio["cash_krw"] = round(float(portfolio.get("cash_krw", 0.0)) + net_proceeds_krw, 0)
    portfolio["last_fx_rate"] = round(float(fx_rate), 2)

    seq = len(portfolio.get("trade_history", [])) + 1
    trade_id = f"{sym}_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{seq}"

    journal_entry = {
        "trade_id": trade_id,
        "ticker": sym,
        "shares": shares,
        "entry_price_usd": round(entry_price_usd, 4),
        "exit_price_usd": round(float(price_usd), 4),
        "entry_fx_rate": round(entry_fx_rate, 2),
        "exit_fx_rate": round(float(fx_rate), 2),
        "entry_cost_krw": round(entry_cost_krw, 0),
        "exit_proceeds_krw": round(gross_proceeds_krw, 0),
        "entry_time": pos.get("entry_time", now_ts),
        "exit_time": now_ts,
        "holding_hours": holding_hours,
        "entry_reason": pos.get("entry_reason", "사유 미기록"),
        "entry_indicators": pos.get("entry_indicators", {}),
        "exit_reason": reason,
        "exit_indicators": exit_indicators or {},
        "gross_pnl_krw": gross_pnl_krw,
        "net_pnl_krw": net_pnl_krw,
        "pnl_pct": pnl_pct,
        "price_return_pct": price_return_pct,
        "total_fees_krw": round(total_fees_krw, 0),
        "memo": "",
    }
    portfolio.setdefault("trade_history", []).append(journal_entry)
    del portfolio["positions"][sym]

    _record_krw_equity_snapshot(portfolio, now_ts, fx_rate)
    return (
        True,
        f"🛑 [{sym}] {shares}주 매도 완료 (@${price_usd:,.2f}, 순손익 ₩{net_pnl_krw:+,.0f} ({pnl_pct:+.2f}%), 사유: {reason})",
    )


def update_krw_market_prices(
    portfolio: Dict[str, Any],
    current_prices_usd: Dict[str, float],
    fx_rate: float,
    record_snapshot: bool = True,
) -> None:
    """Refresh active KRW positions with live USD prices and USD/KRW FX rate."""
    fee_rate = float(portfolio.get("fee_rate", DEFAULT_FEE_RATE))
    portfolio["last_fx_rate"] = round(float(fx_rate), 2)

    for sym, pos in portfolio.get("positions", {}).items():
        px = float(current_prices_usd.get(sym, pos.get("current_price_usd", 0.0)))
        if px <= 0.0:
            continue
        pos["current_price_usd"] = round(px, 4)
        pos["current_fx_rate"] = round(float(fx_rate), 2)
        if px > float(pos.get("highest_price_usd", 0.0)):
            pos["highest_price_usd"] = round(px, 4)

        shares = int(pos["shares"])
        entry_px = float(pos["entry_price_usd"])
        entry_cost_krw = float(pos.get("entry_cost_krw", shares * entry_px * pos.get("entry_fx_rate", fx_rate)))
        entry_fee_krw = float(pos.get("entry_fee_krw", 0.0))

        gross_val_krw = shares * px * fx_rate
        est_exit_fee_krw = gross_val_krw * fee_rate
        invested_total_krw = entry_cost_krw + entry_fee_krw
        unrealized_krw = round((gross_val_krw - est_exit_fee_krw) - invested_total_krw, 0)
        unrealized_pct = round((unrealized_krw / invested_total_krw) * 100.0, 2) if invested_total_krw > 0 else 0.0

        pos["current_value_krw"] = round(gross_val_krw, 0)
        pos["unrealized_pnl_krw"] = unrealized_krw
        pos["unrealized_pnl_pct"] = unrealized_pct
        pos["price_return_pct"] = round((px - entry_px) / entry_px * 100.0, 2)

    if record_snapshot:
        _record_krw_equity_snapshot(portfolio, _now_iso(), fx_rate)


def compute_portfolio_summary_krw(portfolio: Dict[str, Any], fx_rate: Optional[float] = None) -> Dict[str, Any]:
    """Compute top-level KRW account metrics safely."""
    fx = float(fx_rate or portfolio.get("last_fx_rate", DEFAULT_FX_RATE))
    initial_cap = float(portfolio.get("initial_capital_krw", DEFAULT_INITIAL_CAPITAL_KRW))
    cash_krw = float(portfolio.get("cash_krw", initial_cap))
    positions = portfolio.get("positions", {})

    invested_krw = sum(
        int(pos["shares"]) * float(pos["current_price_usd"]) * fx
        for pos in positions.values()
    )
    total_equity_krw = round(cash_krw + invested_krw, 0)
    unrealized_pnl_krw = sum(float(pos.get("unrealized_pnl_krw", 0.0)) for pos in positions.values())
    realized_pnl_krw = sum(float(t.get("net_pnl_krw", 0.0)) for t in portfolio.get("trade_history", []))
    total_return_pct = ((total_equity_krw - initial_cap) / initial_cap * 100.0) if initial_cap > 0 else 0.0

    return {
        "initial_capital_krw": initial_cap,
        "cash_krw": round(cash_krw, 0),
        "invested_krw": round(invested_krw, 0),
        "total_equity_krw": total_equity_krw,
        "total_equity_usd": round(total_equity_krw / fx, 2) if fx > 0 else 0.0,
        "total_return_pct": round(total_return_pct, 2),
        "unrealized_pnl_krw": round(unrealized_pnl_krw, 0),
        "realized_pnl_krw": round(realized_pnl_krw, 0),
        "fx_rate": round(fx, 2),
        "position_count": len(positions),
    }


def compute_journal_metrics(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Trading Journal KPIs (win rate, profit factor, avg return, holding hours)."""
    history = portfolio.get("trade_history", [])
    total = len(history)
    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "total_realized_krw": 0.0,
            "total_fees_krw": 0.0,
            "avg_holding_hours": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
        }

    pnls = [float(t.get("net_pnl_krw", 0.0)) for t in history]
    pcts = [float(t.get("pnl_pct", 0.0)) for t in history]
    fees = [float(t.get("total_fees_krw", 0.0)) for t in history]
    hours = [float(t.get("holding_hours", 0.0)) for t in history]

    win_trades = [t for t in history if float(t.get("net_pnl_krw", 0.0)) > 0]
    loss_trades = [t for t in history if float(t.get("net_pnl_krw", 0.0)) <= 0]

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        "total_trades": total,
        "wins": len(win_trades),
        "losses": len(loss_trades),
        "win_rate_pct": round(len(win_trades) / total * 100.0, 1),
        "avg_return_pct": round(float(np.mean(pcts)), 2),
        "avg_win_pct": round(float(np.mean([float(t.get("pnl_pct", 0.0)) for t in win_trades])), 2) if win_trades else 0.0,
        "avg_loss_pct": round(float(np.mean([float(t.get("pnl_pct", 0.0)) for t in loss_trades])), 2) if loss_trades else 0.0,
        "profit_factor": profit_factor,
        "total_realized_krw": round(sum(pnls), 0),
        "total_fees_krw": round(sum(fees), 0),
        "avg_holding_hours": round(float(np.mean(hours)), 1),
        "best_trade_pct": round(max(pcts), 2),
        "worst_trade_pct": round(min(pcts), 2),
    }


def update_trade_memo(portfolio: Dict[str, Any], trade_id: str, memo_text: str) -> bool:
    """Update reflection memo on a closed trade in the Trading Journal."""
    for rec in portfolio.get("trade_history", []):
        if rec.get("trade_id") == trade_id:
            rec["memo"] = memo_text.strip()
            return True
    return False


# ---------------------------------------------------------------------------
# Signal Evaluation (1-Hour Bar Champion DNA Strategy)
# ---------------------------------------------------------------------------

def evaluate_live_champion_signals(df: pd.DataFrame, ticker: str = "ASSET") -> Dict[str, Any]:
    """Evaluate live Champion DNA 1h rules up to the latest 1-hour bar."""
    sym = ticker.upper().strip()
    if df is None or len(df) < 200:
        last_px = float(df["Close"].iloc[-1]) if df is not None and not df.empty else 0.0
        return {
            "ticker": sym,
            "is_buy_signal": False,
            "is_exit_signal": False,
            "current_price": round(last_px, 4),
            "ema_10d": 0.0,
            "ema_30d": 0.0,
            "ema_200h": 0.0,
            "atr_pct": 0.0,
            "dist_from_high_pct": 999.0,
            "alpha_score": 0.0,
            "latest_bar_time": "",
            "rationale": f"데이터 부족 ({0 if df is None else len(df)}/200봉)",
        }

    close = df["Close"]
    high = df["High"]

    ema_10d = ema_days(close, 10)     # 65h
    ema_30d = ema_days(close, 30)     # 195h
    ema_200h = ema_hours(close, 200)  # 200h

    atr = compute_atr(df, period=14)
    latest_close = float(close.iloc[-1])
    atr_pct = float(atr.iloc[-1] / latest_close * 100.0) if latest_close > 0 else 0.0

    recent_high_60 = float(high.iloc[-60:].max())
    dist_from_high_pct = float((recent_high_60 - latest_close) / recent_high_60 * 100.0) if recent_high_60 > 0 else 999.0

    latest_ema10d = float(ema_10d.iloc[-1])
    latest_ema30d = float(ema_30d.iloc[-1])
    latest_ema200h = float(ema_200h.iloc[-1])

    is_golden = latest_ema10d > latest_ema30d
    is_above_200h = latest_close > latest_ema200h
    is_volatility_fit = atr_pct >= 1.1
    is_momentum_breakout = dist_from_high_pct <= 6.0

    is_buy = is_golden and is_above_200h and is_volatility_fit and is_momentum_breakout
    is_exit = latest_ema10d <= latest_ema30d

    dna_res = evaluate_stock_dna(df, sym)
    alpha_score = float(dna_res.alpha_score)

    if is_buy:
        rationale = (
            f"🏆 1h 챔피언 DNA 충족 (점수 {alpha_score:.0f}, ATR {atr_pct:.2f}%, "
            f"10/30일 정배열, 200h선 상단, 60봉 신고가 -{dist_from_high_pct:.1f}%)"
        )
    else:
        fails = []
        if not is_golden:
            fails.append("10/30일 역배열")
        if not is_above_200h:
            fails.append("200h선 하회")
        if not is_volatility_fit:
            fails.append(f"저변동(ATR {atr_pct:.2f}%)")
        if not is_momentum_breakout:
            fails.append(f"고점대비 -{dist_from_high_pct:.1f}%")
        rationale = "탈락/대기: " + ", ".join(fails)

    bar_ts = ""
    try:
        bar_ts = pd.to_datetime(df.index[-1]).isoformat()
    except Exception:
        pass

    return {
        "ticker": sym,
        "is_buy_signal": is_buy,
        "is_exit_signal": is_exit,
        "current_price": round(latest_close, 4),
        "ema_10d": round(latest_ema10d, 2),
        "ema_30d": round(latest_ema30d, 2),
        "ema_200h": round(latest_ema200h, 2),
        "atr_pct": round(atr_pct, 2),
        "dist_from_high_pct": round(dist_from_high_pct, 1),
        "alpha_score": round(alpha_score, 1),
        "latest_bar_time": bar_ts,
        "rationale": rationale,
    }


def _extract_indicator_snapshot(sig: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "atr_pct": sig.get("atr_pct", 0.0),
        "dist_from_high_pct": sig.get("dist_from_high_pct", 0.0),
        "ema_10d": sig.get("ema_10d", 0.0),
        "ema_30d": sig.get("ema_30d", 0.0),
        "ema_200h": sig.get("ema_200h", 0.0),
        "alpha_score": sig.get("alpha_score", 0.0),
        "bar_time": sig.get("latest_bar_time", ""),
    }


def run_live_scan_and_trade(
    portfolio: Dict[str, Any],
    universe_data: Dict[str, pd.DataFrame],
    fx_rate: float,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    auto_trade: bool = True,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    filepath: Optional[Path] = PORTFOLIO_FILE,
) -> Dict[str, Any]:
    """Run full 1-hour live API scan, manage open positions, and execute integer-share KRW buys."""
    sell_executions: List[str] = []
    buy_executions: List[str] = []
    exit_alerts: List[Dict[str, Any]] = []
    skipped_over_budget: List[Dict[str, Any]] = []

    fee_rate = float(portfolio.get("fee_rate", DEFAULT_FEE_RATE))
    portfolio["max_positions"] = int(max_positions)

    # 1. Inspect existing positions for 12% Hard Stop & 10/30d Macro Dead Cross
    active_syms = list(portfolio.get("positions", {}).keys())
    signals_cache: Dict[str, Dict[str, Any]] = {}

    for sym in active_syms:
        df = universe_data.get(sym)
        if df is None or df.empty:
            continue

        pos = portfolio["positions"][sym]
        latest_close = float(df["Close"].iloc[-1])
        latest_open = float(df["Open"].iloc[-1])
        stop_price = float(pos["stop_loss_price_usd"])

        sig = evaluate_live_champion_signals(df, sym)
        signals_cache[sym] = sig
        exit_ind = _extract_indicator_snapshot(sig)

        # Hard Stop check (gap down opens fill at Open price)
        if latest_close <= stop_price or latest_open <= stop_price:
            exec_px = latest_open if latest_open <= stop_price else latest_close
            exit_reason = f"🚨 12% 하드스탑 손절 도달 (기준가 ${stop_price:,.2f}, 체결 ${exec_px:,.2f})"
            if auto_trade:
                ok, msg = execute_krw_sell(
                    portfolio, sym, price_usd=exec_px, fx_rate=fx_rate,
                    reason=exit_reason, exit_indicators=exit_ind,
                )
                if ok:
                    sell_executions.append(msg)
            else:
                exit_alerts.append({"ticker": sym, "price_usd": exec_px, "reason": exit_reason})
            continue

        # Macro Dead Cross check
        if sig["is_exit_signal"]:
            exit_reason = (
                f"📉 10일 거시선(${sig['ema_10d']:,.2f}) ≤ 30일 거시선(${sig['ema_30d']:,.2f}) 데드크로스 전량 매도"
            )
            if auto_trade:
                ok, msg = execute_krw_sell(
                    portfolio, sym, price_usd=latest_close, fx_rate=fx_rate,
                    reason=exit_reason, exit_indicators=exit_ind,
                )
                if ok:
                    sell_executions.append(msg)
            else:
                exit_alerts.append({"ticker": sym, "price_usd": latest_close, "reason": exit_reason})

    # 2. Refresh prices & FX for remaining positions
    current_prices_usd = {
        s: float(df["Close"].iloc[-1]) for s, df in universe_data.items() if df is not None and not df.empty
    }
    update_krw_market_prices(portfolio, current_prices_usd, fx_rate=fx_rate, record_snapshot=True)

    # 3. Evaluate all scanned stocks in the universe
    buy_candidates: List[Dict[str, Any]] = []
    scan_rows: List[Dict[str, Any]] = []

    for sym, df in universe_data.items():
        sig = signals_cache.get(sym) or evaluate_live_champion_signals(df, sym)
        signals_cache[sym] = sig
        px_usd = float(sig["current_price"])
        unit_krw = round(px_usd * fx_rate * (1.0 + fee_rate), 0)
        is_held = sym in portfolio["positions"]

        if sig["is_buy_signal"] and not is_held:
            buy_candidates.append({
                "ticker": sym,
                "price_usd": px_usd,
                "unit_krw": unit_krw,
                "alpha_score": sig["alpha_score"],
                "atr_pct": sig["atr_pct"],
                "dist_from_high_pct": sig["dist_from_high_pct"],
                "rationale": sig["rationale"],
                "indicators": _extract_indicator_snapshot(sig),
            })

    # Rank candidates by Champion Alpha Score, then Volatility Power (ATR%)
    buy_candidates.sort(key=lambda x: (x["alpha_score"], x["atr_pct"]), reverse=True)

    # 4. Execute Auto-Buys respecting integer-share constraint & up to max_positions
    bought_syms = set()
    over_budget_syms = set()
    available_slots = max(0, int(max_positions) - len(portfolio["positions"]))

    if auto_trade and available_slots > 0 and buy_candidates:
        for cand in buy_candidates:
            if available_slots <= 0:
                break
            cash_krw = float(portfolio.get("cash_krw", 0.0))
            if cash_krw <= 0:
                break
            alloc_krw = cash_krw / available_slots
            sym = cand["ticker"]
            px_usd = cand["price_usd"]
            unit_need_krw = px_usd * fx_rate * (1.0 + fee_rate)

            if int(min(cash_krw, alloc_krw) // unit_need_krw) <= 0:
                over_budget_syms.add(sym)
                skipped_over_budget.append({
                    "ticker": sym,
                    "price_usd": px_usd,
                    "unit_krw": unit_need_krw,
                    "slot_budget_krw": alloc_krw,
                })
                # Do NOT consume slot; proceed to next affordable candidate
                continue

            ok, msg = execute_krw_buy(
                portfolio,
                ticker=sym,
                price_usd=px_usd,
                fx_rate=fx_rate,
                allocation_krw=alloc_krw,
                reason=cand["rationale"],
                indicators=cand["indicators"],
                stop_loss_pct=stop_loss_pct,
            )
            if ok:
                buy_executions.append(msg)
                bought_syms.add(sym)
                available_slots -= 1

    # 5. Build structured scan table rows
    for sym, sig in signals_cache.items():
        px_usd = float(sig["current_price"])
        unit_krw = round(px_usd * fx_rate * (1.0 + fee_rate), 0)
        if sym in bought_syms:
            status = "✅ 매수체결"
        elif sym in portfolio["positions"]:
            status = "📌 보유중"
        elif sym in over_budget_syms:
            status = "💸 예산초과 스킵"
        elif sig["is_buy_signal"]:
            status = "🎯 매수조건 충족"
        elif sig["alpha_score"] >= 65.0:
            status = "👀 관심대기"
        else:
            status = "⚪ 조건미달"

        scan_rows.append({
            "종목": sym,
            "상태": status,
            "현재가($)": px_usd,
            "1주 원화환산(₩)": int(unit_krw),
            "DNA 점수": sig["alpha_score"],
            "시간당 ATR(%)": sig["atr_pct"],
            "60봉 고점대비(%)": -sig["dist_from_high_pct"] if sig["dist_from_high_pct"] < 900 else None,
            "10일선($)": sig["ema_10d"],
            "30일선($)": sig["ema_30d"],
            "200시간선($)": sig["ema_200h"],
            "기준봉 시각": sig["latest_bar_time"][:16].replace("T", " "),
            "진단 사유": sig["rationale"],
        })

    # Sort scan rows meaningfully: Bought -> Buy Signal -> Over Budget -> Held -> Watchlist -> Others
    priority = {
        "✅ 매수체결": 0,
        "🎯 매수조건 충족": 1,
        "💸 예산초과 스킵": 2,
        "📌 보유중": 3,
        "👀 관심대기": 4,
        "⚪ 조건미달": 5,
    }
    scan_rows.sort(key=lambda r: (priority.get(r["상태"], 9), -r["DNA 점수"], -r["시간당 ATR(%)"]))

    portfolio.setdefault("scan_logs", []).append({
        "timestamp": _now_iso(),
        "scanned_count": len(universe_data),
        "qualified_count": len(buy_candidates),
        "bought_count": len(bought_syms),
        "sold_count": len(sell_executions),
        "skipped_over_budget": len(skipped_over_budget),
        "fx_rate": round(float(fx_rate), 2),
    })
    if len(portfolio["scan_logs"]) > 50:
        portfolio["scan_logs"] = portfolio["scan_logs"][-50:]

    if filepath is not None:
        save_portfolio(portfolio, filepath=filepath)

    return {
        "sell_executions": sell_executions,
        "buy_executions": buy_executions,
        "pending_buys": buy_candidates,
        "exit_alerts": exit_alerts,
        "skipped_over_budget": skipped_over_budget,
        "scan_df": pd.DataFrame(scan_rows),
    }


# ---------------------------------------------------------------------------
# Legacy USD Persistence & Functions (Backward Compatibility for Unit Tests)
# ---------------------------------------------------------------------------

def get_default_portfolio(initial_cash: float = 10000.0, fee_rate: float = 0.001) -> Dict[str, Any]:
    """Return a fresh legacy USD default portfolio structure."""
    now_str = _now_iso()
    return {
        "initial_capital": float(initial_cash),
        "cash": float(initial_cash),
        "fee_rate": float(fee_rate),
        "positions": {},
        "trade_history": [],
        "equity_history": [
            {
                "timestamp": now_str,
                "total_equity": float(initial_cash),
                "cash": float(initial_cash),
                "invested": 0.0,
            }
        ],
        "created_at": now_str,
        "updated_at": now_str,
    }


def load_portfolio(filepath: Path = PORTFOLIO_FILE, initial_cash: float = 10000.0, fee_rate: float = 0.001) -> Dict[str, Any]:
    """Load legacy USD portfolio state from JSON file, initializing if not present."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        port = get_default_portfolio(initial_cash=initial_cash, fee_rate=fee_rate)
        save_portfolio(port, filepath=filepath)
        return port

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            port = json.load(f)
            port.setdefault("positions", {})
            port.setdefault("trade_history", [])
            port.setdefault("equity_history", [])
            return port
    except Exception:
        port = get_default_portfolio(initial_cash=initial_cash, fee_rate=fee_rate)
        save_portfolio(port, filepath=filepath)
        return port


def save_portfolio(portfolio: Dict[str, Any], filepath: Path = PORTFOLIO_FILE) -> None:
    """Save portfolio state to JSON file safely."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    portfolio["updated_at"] = _now_iso()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def reset_portfolio(initial_cash: float = 10000.0, fee_rate: float = 0.001, filepath: Path = PORTFOLIO_FILE) -> Dict[str, Any]:
    """Reset legacy USD portfolio state back to initial capital."""
    port = get_default_portfolio(initial_cash=initial_cash, fee_rate=fee_rate)
    save_portfolio(port, filepath=filepath)
    return port


def execute_paper_buy(
    portfolio: Dict[str, Any],
    ticker: str,
    price: float,
    allocation_amount: float,
    reason: str = "🏆 고수익 챔피언 DNA 돌파 매수",
    timestamp: Optional[str] = None,
    stop_loss_pct: float = 0.12,
) -> Tuple[bool, str]:
    """Execute a legacy USD simulated buy order with transaction fees."""
    sym = ticker.upper().strip()
    if price <= 0.0:
        return False, f"유효하지 않은 가격입니다: ${price}"

    fee_rate = portfolio.get("fee_rate", 0.001)
    cash = portfolio.get("cash", 0.0)

    if sym in portfolio["positions"]:
        return False, f"이미 보유 중인 종목입니다: {sym}"

    max_budget = min(cash, allocation_amount)
    shares = int(max_budget // (price * (1.0 + fee_rate)))
    if shares <= 0:
        return False, f"매수 가능 자금이 부족합니다 (현재 현금: ${cash:,.2f}, 필요 예상액: ${price * (1 + fee_rate):,.2f})"

    trade_cost = shares * price
    fee = trade_cost * fee_rate
    total_deduction = trade_cost + fee

    if total_deduction > cash:
        return False, f"수수료 포함 자금이 부족합니다 (보유: ${cash:,.2f}, 필요: ${total_deduction:,.2f})"

    portfolio["cash"] = round(cash - total_deduction, 2)
    now_ts = timestamp or _now_iso()
    stop_loss_price = round(price * (1.0 - stop_loss_pct), 2)
    portfolio["positions"][sym] = {
        "ticker": sym,
        "shares": shares,
        "entry_price": round(price, 2),
        "entry_time": now_ts,
        "current_price": round(price, 2),
        "stop_loss_price": stop_loss_price,
        "highest_price": round(price, 2),
        "cost_basis": round(trade_cost, 2),
        "entry_fee": round(fee, 2),
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "entry_reason": reason,
    }

    _record_equity_snapshot(portfolio, now_ts)
    return True, f"✅ [{sym}] {shares}주 매수 체결완료 (@${price:.2f}, 투자금: ${trade_cost:,.2f}, 수수료: ${fee:.2f})"


def execute_paper_sell(
    portfolio: Dict[str, Any],
    ticker: str,
    price: float,
    reason: str = "청산",
    timestamp: Optional[str] = None,
) -> Tuple[bool, str]:
    """Execute a legacy USD simulated sell order, calculating realized PnL and deducting fees."""
    sym = ticker.upper().strip()
    if sym not in portfolio["positions"]:
        return False, f"보유하고 있지 않은 종목입니다: {sym}"

    pos = portfolio["positions"][sym]
    shares = pos["shares"]
    entry_price = pos["entry_price"]
    entry_fee = pos.get("entry_fee", 0.0)

    fee_rate = portfolio.get("fee_rate", 0.001)
    gross_proceeds = shares * price
    exit_fee = gross_proceeds * fee_rate
    net_proceeds = gross_proceeds - exit_fee
    total_fees = entry_fee + exit_fee

    realized_pnl = round(gross_proceeds - (shares * entry_price) - total_fees, 2)
    pnl_pct = round((price - entry_price) / entry_price * 100.0, 2)
    now_ts = timestamp or _now_iso()

    portfolio["cash"] = round(portfolio["cash"] + net_proceeds, 2)
    trade_record = {
        "trade_id": f"{sym}_{int(datetime.now(timezone.utc).timestamp())}",
        "ticker": sym,
        "shares": shares,
        "entry_price": entry_price,
        "exit_price": round(price, 2),
        "entry_time": pos["entry_time"],
        "exit_time": now_ts,
        "entry_reason": pos.get("entry_reason", ""),
        "gross_pnl": round(gross_proceeds - (shares * entry_price), 2),
        "net_pnl": realized_pnl,
        "pnl_pct": pnl_pct,
        "total_fees": round(total_fees, 2),
        "exit_reason": reason,
    }
    portfolio["trade_history"].append(trade_record)
    del portfolio["positions"][sym]

    _record_equity_snapshot(portfolio, now_ts)
    return True, f"🛑 [{sym}] {shares}주 청산 완료 (@${price:.2f}, 손익: ${realized_pnl:+,.2f} ({pnl_pct:+.2f}%), 사유: {reason})"


def update_market_prices(portfolio: Dict[str, Any], current_prices: Dict[str, float]) -> None:
    """Update legacy USD active positions with current market prices and refresh equity history."""
    for sym, price in current_prices.items():
        if sym in portfolio["positions"] and price > 0.0:
            pos = portfolio["positions"][sym]
            pos["current_price"] = round(price, 2)
            if price > pos.get("highest_price", 0.0):
                pos["highest_price"] = round(price, 2)

            shares = pos["shares"]
            entry_price = pos["entry_price"]
            gross_val = shares * price
            cost = shares * entry_price
            fee_rate = portfolio.get("fee_rate", 0.001)
            est_exit_fee = gross_val * fee_rate

            pos["unrealized_pnl"] = round(gross_val - cost - pos.get("entry_fee", 0.0) - est_exit_fee, 2)
            pos["unrealized_pnl_pct"] = round((price - entry_price) / entry_price * 100.0, 2)

    _record_equity_snapshot(portfolio, _now_iso())


def _record_equity_snapshot(portfolio: Dict[str, Any], timestamp: str) -> None:
    invested = sum(pos["shares"] * pos["current_price"] for pos in portfolio["positions"].values())
    total_equity = round(portfolio["cash"] + invested, 2)
    portfolio["equity_history"].append({
        "timestamp": timestamp,
        "total_equity": total_equity,
        "cash": round(portfolio["cash"], 2),
        "invested": round(invested, 2),
    })
    if len(portfolio["equity_history"]) > 1000:
        portfolio["equity_history"] = portfolio["equity_history"][-1000:]


def scan_and_rebalance_portfolio(
    portfolio: Dict[str, Any],
    universe_data: Dict[str, pd.DataFrame],
    max_positions: int = 5,
    auto_trade: bool = False,
    filepath: Optional[Path] = PORTFOLIO_FILE,
) -> Dict[str, Any]:
    """Legacy USD scanner/rebalancer preserved for existing tests."""
    sell_executions = []
    buy_executions = []
    pending_buys = []
    exit_alerts = []

    active_syms = list(portfolio["positions"].keys())
    for sym in active_syms:
        pos = portfolio["positions"][sym]
        df = universe_data.get(sym)
        if df is None or df.empty:
            continue

        latest_close = float(df["Close"].iloc[-1])
        latest_open = float(df["Open"].iloc[-1])
        pos["current_price"] = round(latest_close, 2)
        stop_price = pos["stop_loss_price"]

        is_hard_stop = latest_close <= stop_price or latest_open <= stop_price
        if is_hard_stop:
            exec_price = latest_open if latest_open <= stop_price else latest_close
            exit_reason = f"🚨 12% 와이드 하드스탑 손절 터치 (손절기준가: ${stop_price:.2f})"
            if auto_trade:
                ok, msg = execute_paper_sell(portfolio, sym, price=exec_price, reason=exit_reason)
                sell_executions.append(msg)
            else:
                exit_alerts.append({"ticker": sym, "price": exec_price, "reason": exit_reason})
            continue

        sig = evaluate_live_champion_signals(df, sym)
        if sig["is_exit_signal"]:
            exit_reason = "📉 10일 거시선 <= 30일선 데드크로스 전량 청산"
            if auto_trade:
                ok, msg = execute_paper_sell(portfolio, sym, price=latest_close, reason=exit_reason)
                sell_executions.append(msg)
            else:
                exit_alerts.append({"ticker": sym, "price": latest_close, "reason": exit_reason})

    current_prices = {s: float(df["Close"].iloc[-1]) for s, df in universe_data.items() if not df.empty}
    update_market_prices(portfolio, current_prices)

    current_positions_count = len(portfolio["positions"])
    available_slots = max(0, max_positions - current_positions_count)

    for sym, df in universe_data.items():
        if sym in portfolio["positions"]:
            continue
        sig = evaluate_live_champion_signals(df, sym)
        if sig["is_buy_signal"]:
            pending_buys.append({
                "ticker": sym,
                "price": sig["current_price"],
                "atr_pct": sig["atr_pct"],
                "dist_from_high_pct": sig["dist_from_high_pct"],
                "rationale": sig["rationale"],
            })

    pending_buys.sort(key=lambda x: x["atr_pct"], reverse=True)

    if auto_trade and available_slots > 0 and pending_buys:
        cash = portfolio["cash"]
        alloc_per_stock = cash / available_slots
        for target in pending_buys[:available_slots]:
            sym = target["ticker"]
            px = target["price"]
            ok, msg = execute_paper_buy(portfolio, sym, price=px, allocation_amount=alloc_per_stock, reason=target["rationale"])
            if ok:
                buy_executions.append(msg)
                available_slots -= 1
            if available_slots <= 0:
                break

    if filepath is not None:
        save_portfolio(portfolio, filepath=filepath)
    return {
        "sell_executions": sell_executions,
        "buy_executions": buy_executions,
        "pending_buys": pending_buys,
        "exit_alerts": exit_alerts,
    }
