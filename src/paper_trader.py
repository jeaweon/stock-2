"""Paper Trading (모의 투자) Engine for Champion DNA Momentum Strategy.

Manages portfolio balance, active positions, trade executions, fee calculations,
stop-loss enforcement, macro exit triggers, and persistence.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.timeframes import ema_days, ema_hours
from src.adaptive_experiment import compute_atr

DATA_DIR = Path("data")
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"


def get_default_portfolio(initial_cash: float = 10000.0, fee_rate: float = 0.001) -> Dict[str, Any]:
    """Return a fresh default portfolio structure."""
    now_str = datetime.now(timezone.utc).isoformat()
    return {
        "initial_capital": float(initial_cash),
        "cash": float(initial_cash),
        "fee_rate": float(fee_rate),
        "positions": {},  # ticker -> position dict
        "trade_history": [],  # list of closed trade dicts
        "equity_history": [  # timestamped equity curve
            {
                "timestamp": now_str,
                "total_equity": float(initial_cash),
                "cash": float(initial_cash),
                "invested": 0.0
            }
        ],
        "created_at": now_str,
        "updated_at": now_str
    }


def load_portfolio(filepath: Path = PORTFOLIO_FILE, initial_cash: float = 10000.0, fee_rate: float = 0.001) -> Dict[str, Any]:
    """Load portfolio state from JSON file, initializing if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        port = get_default_portfolio(initial_cash=initial_cash, fee_rate=fee_rate)
        save_portfolio(port, filepath=filepath)
        return port

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            port = json.load(f)
            # Ensure all required keys exist
            if "positions" not in port:
                port["positions"] = {}
            if "trade_history" not in port:
                port["trade_history"] = []
            if "equity_history" not in port:
                port["equity_history"] = []
            return port
    except Exception:
        port = get_default_portfolio(initial_cash=initial_cash, fee_rate=fee_rate)
        save_portfolio(port, filepath=filepath)
        return port


def save_portfolio(portfolio: Dict[str, Any], filepath: Path = PORTFOLIO_FILE) -> None:
    """Save portfolio state to JSON file safely."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    portfolio["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def reset_portfolio(initial_cash: float = 10000.0, fee_rate: float = 0.001, filepath: Path = PORTFOLIO_FILE) -> Dict[str, Any]:
    """Reset portfolio state back to initial capital."""
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
    stop_loss_pct: float = 0.12
) -> Tuple[bool, str]:
    """Execute a simulated buy order with transaction fees.
    
    Args:
        portfolio: The current portfolio dictionary.
        ticker: Symbol to buy.
        price: Current market price.
        allocation_amount: Target dollar amount to invest.
        reason: Buy rationale.
        timestamp: ISO timestamp string (defaults to now).
        stop_loss_pct: Hard stop loss percentage (default 12%).
        
    Returns:
        (success: bool, message: str)
    """
    sym = ticker.upper().strip()
    if price <= 0.0:
        return False, f"유효하지 않은 가격입니다: ${price}"

    fee_rate = portfolio.get("fee_rate", 0.001)
    cash = portfolio.get("cash", 0.0)

    if sym in portfolio["positions"]:
        return False, f"이미 보유 중인 종목입니다: {sym}"

    # Calculate integer shares or min 1
    max_budget = min(cash, allocation_amount)
    # Estimate cost with fee: shares * price * (1 + fee_rate) <= max_budget
    shares = int(max_budget // (price * (1.0 + fee_rate)))
    if shares <= 0:
        return False, f"매수 가능 자금이 부족합니다 (현재 현금: ${cash:,.2f}, 필요 예상액: ${price * (1 + fee_rate):,.2f})"

    trade_cost = shares * price
    fee = trade_cost * fee_rate
    total_deduction = trade_cost + fee

    if total_deduction > cash:
        return False, f"수수료 포함 자금이 부족합니다 (보유: ${cash:,.2f}, 필요: ${total_deduction:,.2f})"

    # Update Cash
    portfolio["cash"] = round(cash - total_deduction, 2)

    # Record Position
    now_ts = timestamp or datetime.now(timezone.utc).isoformat()
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
        "entry_reason": reason
    }

    # Record equity snapshot
    _record_equity_snapshot(portfolio, now_ts)
    return True, f"✅ [{sym}] {shares}주 매수 체결완료 (@${price:.2f}, 투자금: ${trade_cost:,.2f}, 수수료: ${fee:.2f})"


def execute_paper_sell(
    portfolio: Dict[str, Any],
    ticker: str,
    price: float,
    reason: str = "청산",
    timestamp: Optional[str] = None
) -> Tuple[bool, str]:
    """Execute a simulated sell order, calculating realized PnL and deducting fees.
    
    Returns:
        (success: bool, message: str)
    """
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

    now_ts = timestamp or datetime.now(timezone.utc).isoformat()

    # Update Cash
    portfolio["cash"] = round(portfolio["cash"] + net_proceeds, 2)

    # Record Closed Trade
    trade_record = {
        "trade_id": f"{sym}_{int(datetime.now(timezone.utc).timestamp())}",
        "ticker": sym,
        "shares": shares,
        "entry_price": entry_price,
        "exit_price": round(price, 2),
        "entry_time": pos["entry_time"],
        "exit_time": now_ts,
        "gross_pnl": round(gross_proceeds - (shares * entry_price), 2),
        "net_pnl": realized_pnl,
        "pnl_pct": pnl_pct,
        "total_fees": round(total_fees, 2),
        "exit_reason": reason
    }
    portfolio["trade_history"].append(trade_record)

    # Remove position
    del portfolio["positions"][sym]

    # Record equity snapshot
    _record_equity_snapshot(portfolio, now_ts)
    return True, f"🛑 [{sym}] {shares}주 청산 완료 (@${price:.2f}, 손익: ${realized_pnl:+,.2f} ({pnl_pct:+.2f}%), 사유: {reason})"


def update_market_prices(portfolio: Dict[str, Any], current_prices: Dict[str, float]) -> None:
    """Update active positions with current market prices and refresh equity history."""
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
            net_val = gross_val - est_exit_fee

            pos["unrealized_pnl"] = round(gross_val - cost - pos.get("entry_fee", 0.0) - est_exit_fee, 2)
            pos["unrealized_pnl_pct"] = round((price - entry_price) / entry_price * 100.0, 2)

    now_ts = datetime.now(timezone.utc).isoformat()
    _record_equity_snapshot(portfolio, now_ts)


def _record_equity_snapshot(portfolio: Dict[str, Any], timestamp: str) -> None:
    """Append current portfolio total equity to history."""
    invested = sum(pos["shares"] * pos["current_price"] for pos in portfolio["positions"].values())
    total_equity = round(portfolio["cash"] + invested, 2)
    portfolio["equity_history"].append({
        "timestamp": timestamp,
        "total_equity": total_equity,
        "cash": round(portfolio["cash"], 2),
        "invested": round(invested, 2)
    })
    # Keep up to 1000 snapshots
    if len(portfolio["equity_history"]) > 1000:
        portfolio["equity_history"] = portfolio["equity_history"][-1000:]


def evaluate_live_champion_signals(df: pd.DataFrame, ticker: str = "ASSET") -> Dict[str, Any]:
    """Evaluate live Champion DNA rules on a stock DataFrame up to the latest bar.
    
    Returns:
        dict containing:
            - is_buy_signal: bool
            - is_exit_signal: bool
            - current_price: float
            - ema_10d: float
            - ema_30d: float
            - ema_200h: float
            - atr_pct: float
            - dist_from_high_pct: float
            - rationale: str
    """
    sym = ticker.upper().strip()
    if len(df) < 200:
        return {
            "ticker": sym,
            "is_buy_signal": False,
            "is_exit_signal": False,
            "current_price": float(df["Close"].iloc[-1]) if not df.empty else 0.0,
            "rationale": "데이터 부족 (최소 200시간봉 필요)"
        }

    close = df["Close"]
    high = df["High"]

    ema_10d = ema_days(close, 10)     # 65h
    ema_30d = ema_days(close, 30)     # 195h
    ema_200h = ema_hours(close, 200)  # 200h

    atr = compute_atr(df, period=14)
    atr_pct = float(atr.iloc[-1] / close.iloc[-1] * 100.0)

    recent_high_60 = float(high.iloc[-60:].max())
    latest_close = float(close.iloc[-1])
    dist_from_high_pct = float((recent_high_60 - latest_close) / recent_high_60 * 100.0)

    latest_ema10d = float(ema_10d.iloc[-1])
    latest_ema30d = float(ema_30d.iloc[-1])
    latest_ema200h = float(ema_200h.iloc[-1])

    is_golden = latest_ema10d > latest_ema30d
    is_above_200h = latest_close > latest_ema200h
    is_volatility_fit = atr_pct >= 1.1
    is_momentum_breakout = dist_from_high_pct <= 6.0

    # Buy Signal: all 4 Champion DNA criteria met
    is_buy = is_golden and is_above_200h and is_volatility_fit and is_momentum_breakout

    # Exit Signal: 10d EMA <= 30d EMA (Macro Dead Cross)
    is_exit = latest_ema10d <= latest_ema30d

    rationale_parts = []
    if is_buy:
        rationale_parts.append(f"🏆 고수익 챔피언 DNA 충족! (ATR {atr_pct:.1f}%, 10/30일 정배열, 신고가권 -{dist_from_high_pct:.1f}%)")
    else:
        fails = []
        if not is_golden: fails.append("10/30일 역배열")
        if not is_above_200h: fails.append("200선 하회")
        if not is_volatility_fit: fails.append(f"저변동성(ATR {atr_pct:.1f}%)")
        if not is_momentum_breakout: fails.append(f"신고가 대비 눌림(-{dist_from_high_pct:.1f}%)")
        rationale_parts.append("대기/탈락: " + ", ".join(fails))

    return {
        "ticker": sym,
        "is_buy_signal": is_buy,
        "is_exit_signal": is_exit,
        "current_price": round(latest_close, 2),
        "ema_10d": round(latest_ema10d, 2),
        "ema_30d": round(latest_ema30d, 2),
        "ema_200h": round(latest_ema200h, 2),
        "atr_pct": round(atr_pct, 2),
        "dist_from_high_pct": round(dist_from_high_pct, 1),
        "rationale": " | ".join(rationale_parts)
    }


def scan_and_rebalance_portfolio(
    portfolio: Dict[str, Any],
    universe_data: Dict[str, pd.DataFrame],
    max_positions: int = 5,
    auto_trade: bool = False
) -> Dict[str, Any]:
    """Scan candidate universe, check sell conditions on open positions, and optionally auto-execute.
    
    Returns:
        dict summarizing:
            - sell_executions: list of sell log messages
            - buy_executions: list of buy log messages
            - pending_buys: list of recommended buy targets
            - exit_alerts: list of exit warnings
    """
    sell_executions = []
    buy_executions = []
    pending_buys = []
    exit_alerts = []

    # 1. Update prices & Check Sell Triggers for current positions
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

        # Check 1: 12% Hard Stop Loss
        # Check if overnight gap down open breached stop
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

        # Check 2: Macro Dead Cross (10d <= 30d EMA)
        sig = evaluate_live_champion_signals(df, sym)
        if sig["is_exit_signal"]:
            exit_reason = "📉 10일 거시선 <= 30일선 데드크로스 전량 청산"
            if auto_trade:
                ok, msg = execute_paper_sell(portfolio, sym, price=latest_close, reason=exit_reason)
                sell_executions.append(msg)
            else:
                exit_alerts.append({"ticker": sym, "price": latest_close, "reason": exit_reason})

    # Refresh market prices and equity for remaining positions
    current_prices = {s: float(df["Close"].iloc[-1]) for s, df in universe_data.items() if not df.empty}
    update_market_prices(portfolio, current_prices)

    # 2. Check Buy Signals for unheld tickers
    current_positions_count = len(portfolio["positions"])
    available_slots = max(0, max_positions - current_positions_count)

    for sym, df in universe_data.items():
        if sym in portfolio["positions"]:
            continue  # Already held

        sig = evaluate_live_champion_signals(df, sym)
        if sig["is_buy_signal"]:
            pending_buys.append({
                "ticker": sym,
                "price": sig["current_price"],
                "atr_pct": sig["atr_pct"],
                "dist_from_high_pct": sig["dist_from_high_pct"],
                "rationale": sig["rationale"]
            })

    # Sort pending buys by highest ATR (volatility power)
    pending_buys.sort(key=lambda x: x["atr_pct"], reverse=True)

    # Auto-execute buys if enabled and slots available
    if auto_trade and available_slots > 0 and pending_buys:
        cash = portfolio["cash"]
        # Equal allocation across available slots
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

    # Save updated portfolio
    save_portfolio(portfolio)

    return {
        "sell_executions": sell_executions,
        "buy_executions": buy_executions,
        "pending_buys": pending_buys,
        "exit_alerts": exit_alerts
    }
