"""Persistent storage module for backtest history and dual-AI evaluation logs."""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

DEFAULT_HISTORY_FILE = Path("backtest_history.csv")

HISTORY_COLUMNS = [
    "timestamp",
    "ticker",
    "iteration_round",
    "strategy_name",
    "buy_rules",
    "sell_rules",
    "position_sizing_rules",
    "win_rate_pct",
    "total_return_pct",
    "mdd_pct",
    "total_trades",
    "profit_factor",
    "sharpe_ratio",
    "critic_verdict",
    "critic_feedback",
    "strategy_code",
]


def save_history_record(
    ticker: str,
    iteration_round: int,
    strategy_name: str,
    buy_rules: str,
    sell_rules: str,
    position_sizing_rules: str,
    win_rate_pct: float,
    total_return_pct: float,
    mdd_pct: float,
    total_trades: int,
    profit_factor: float,
    sharpe_ratio: float,
    critic_verdict: str,
    critic_feedback: str,
    strategy_code: str = "",
    file_path: Path = DEFAULT_HISTORY_FILE,
) -> None:
    """Append a new backtest result record into backtest_history.csv using utf-8-sig for Excel compatibility."""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker.upper().strip(),
        "iteration_round": iteration_round,
        "strategy_name": strategy_name,
        "buy_rules": buy_rules.replace("\n", " ").strip(),
        "sell_rules": sell_rules.replace("\n", " ").strip(),
        "position_sizing_rules": position_sizing_rules.replace("\n", " ").strip(),
        "win_rate_pct": round(win_rate_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
        "mdd_pct": round(mdd_pct, 2),
        "total_trades": total_trades,
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "critic_verdict": critic_verdict,
        "critic_feedback": critic_feedback.replace("\n", " ").strip(),
        "strategy_code": strategy_code.strip(),
    }

    df_new = pd.DataFrame([row], columns=HISTORY_COLUMNS)

    if file_path.exists():
        df_existing = pd.read_csv(file_path, encoding="utf-8-sig")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(file_path, index=False, encoding="utf-8-sig")


def load_history_records(file_path: Path = DEFAULT_HISTORY_FILE) -> pd.DataFrame:
    """Load history records DataFrame. Returns empty DataFrame with columns if file doesn't exist."""
    if not file_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def clear_history(file_path: Path = DEFAULT_HISTORY_FILE) -> None:
    """Delete history file."""
    if file_path.exists():
        file_path.unlink()
