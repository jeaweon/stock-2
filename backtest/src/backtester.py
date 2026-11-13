"""Backtesting engine for 1-hour candle stock data with dynamic sizing and realistic fees."""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    shares: float
    cost: float
    proceeds: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    fee_paid: float


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return_pct: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    cagr_pct: float
    trades: List[Trade]
    equity_curve: pd.Series
    buy_markers: List[Dict[str, Any]] = field(default_factory=list)
    sell_markers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "cagr_pct": round(self.cagr_pct, 2),
        }


def _compute_session_open_flags(index: pd.Index) -> np.ndarray:
    """각 봉이 해당 거래일(미 동부시간 기준)의 첫 봉인지 여부.

    tz-aware 인덱스는 America/New_York으로 변환 후 날짜가 바뀌는 첫 봉을 개장 첫 봉으로 간주합니다.
    (yfinance 1h 정규장 첫 봉 = 09:30 ET)
    """
    n = len(index)
    if n == 0:
        return np.zeros(0, dtype=bool)
    try:
        dt_index = pd.DatetimeIndex(index)
    except Exception:
        return np.zeros(n, dtype=bool)
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("America/New_York")
    dates = dt_index.normalize()
    flags = np.zeros(n, dtype=bool)
    flags[1:] = dates[1:] != dates[:-1]
    return flags


def run_backtest(
    df: pd.DataFrame,
    signal_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    df_with_signals: Optional[pd.DataFrame] = None,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,  # 편도 0.1% 실거래 수수료 (실제 손익 평가용)
    default_stop_loss_pct: Optional[float] = None,
    default_take_profit_pct: Optional[float] = None,
    position_sizing_func: Optional[Callable[[pd.DataFrame], Any]] = None,
    slippage_rate: float = 0.0,
) -> BacktestResult:
    """Execute backtest on 1h OHLCV data.

    position_sizing_func:
        Optimizer(Gemini)가 사전에 '코드'로 생성한 포지션 사이징 수식 함수.
        루프 진입 전 1회만 벡터 연산으로 평가되어 'position_size' 열을 덮어씁니다.
        (봉마다 Gemini API를 호출하지 않으므로 API 병목이 없습니다.)
    slippage_rate:
        갭하락 손절(시가 체결) 시 추가로 불리하게 적용할 슬리피지 비율 (기본 0).

    The signal generation should produce at least:
        - 'signal': 1 (Buy/Enter), -1 (Sell/Exit), 0 (Hold)
    Optional columns:
        - 'position_size': fraction of available cash to invest (e.g. 0.5 = 50%, 1.0 = 100%)
        - 'stop_loss_pct': float (e.g. 0.03 for 3%)
        - 'take_profit_pct': float (e.g. 0.06 for 6%)
    """
    if df_with_signals is not None:
        data = df_with_signals.copy()
    elif signal_func is not None:
        data = signal_func(df.copy())
    else:
        raise ValueError("Either signal_func or df_with_signals must be provided.")

    if "signal" not in data.columns:
        raise ValueError("전략 데이터프레임에 'signal' 열이 포함되어 있어야 합니다.")

    # [API 병목 개선] AI 생성 사이징 수식을 로컬 메모리에서 1회 벡터 평가
    if position_sizing_func is not None:
        sizes = position_sizing_func(data)
        sizes = pd.Series(sizes, index=data.index) if not isinstance(sizes, pd.Series) else sizes.reindex(data.index)
        data["position_size"] = pd.to_numeric(sizes, errors="coerce").fillna(1.0).clip(0.05, 1.0)

    # [갭 리스크] 정규장 개장 첫 봉(당일 첫 1시간봉) 여부 사전 계산
    session_open_flags = _compute_session_open_flags(data.index)

    cash = float(initial_capital)
    shares = 0.0
    trades: List[Trade] = []
    equity_list = []
    buy_markers = []
    sell_markers = []

    in_position = False
    entry_time = None
    entry_price = 0.0
    entry_cost = 0.0
    entry_fee = 0.0
    active_stop_loss = default_stop_loss_pct
    active_take_profit = default_take_profit_pct

    for bar_i, (idx, row) in enumerate(data.iterrows()):
        open_p = float(row["Open"]) if "Open" in row and pd.notna(row["Open"]) else float(row["Close"])
        close_p = float(row["Close"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        sig = int(row.get("signal", 0))

        # Check in-position stop-loss / take-profit
        if in_position:
            # Per-row stop-loss / take-profit override if present
            if "stop_loss_pct" in row and pd.notna(row["stop_loss_pct"]) and row["stop_loss_pct"] > 0:
                active_stop_loss = float(row["stop_loss_pct"])
            if "take_profit_pct" in row and pd.notna(row["take_profit_pct"]) and row["take_profit_pct"] > 0:
                active_take_profit = float(row["take_profit_pct"])

            exit_reason = None
            exit_price = close_p

            stop_price = entry_price * (1.0 - active_stop_loss) if active_stop_loss else None

            if stop_price is not None and session_open_flags[bar_i] and open_p < stop_price:
                # 오버나이트 갭하락: 개장 시가가 이미 손절가 아래 -> 손절가가 아닌 시가로 체결
                exit_price = open_p * (1.0 - slippage_rate)
                exit_reason = f"갭하락 손절 (시가 체결, 설정 {active_stop_loss * 100:.1f}%)"
            elif stop_price is not None and low_p <= stop_price:
                exit_price = stop_price
                exit_reason = f"손절 ({active_stop_loss * 100:.1f}%)"
            elif active_take_profit and high_p >= entry_price * (1.0 + active_take_profit):
                exit_price = entry_price * (1.0 + active_take_profit)
                exit_reason = f"익절 ({active_take_profit * 100:.1f}%)"
            elif sig == -1:
                exit_price = close_p
                exit_reason = "매도 시그널"

            if exit_reason:
                gross_proceeds = shares * exit_price
                exit_fee = gross_proceeds * fee_rate
                net_proceeds = gross_proceeds - exit_fee
                cash += net_proceeds
                total_fee = entry_fee + exit_fee
                pnl = net_proceeds - entry_cost
                pnl_pct = (pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0

                trades.append(
                    Trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=idx,
                        exit_price=exit_price,
                        shares=shares,
                        cost=entry_cost,
                        proceeds=net_proceeds,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        fee_paid=total_fee,
                    )
                )
                sell_markers.append(
                    {"time": idx, "price": exit_price, "reason": exit_reason, "pnl_pct": pnl_pct}
                )
                shares = 0.0
                in_position = False
                active_stop_loss = default_stop_loss_pct
                active_take_profit = default_take_profit_pct

        # Check entry signal
        if not in_position and sig == 1:
            pos_size = float(row.get("position_size", 1.0))
            pos_size = max(0.05, min(1.0, pos_size))  # Bound between 5% and 100%
            invest_cash = cash * pos_size

            if invest_cash > 10.0 and close_p > 0:
                entry_fee = invest_cash * fee_rate
                net_invest = invest_cash - entry_fee
                shares = net_invest / close_p
                cash -= invest_cash
                entry_time = idx
                entry_price = close_p
                entry_cost = invest_cash
                in_position = True

                if "stop_loss_pct" in row and pd.notna(row["stop_loss_pct"]):
                    active_stop_loss = float(row["stop_loss_pct"])
                if "take_profit_pct" in row and pd.notna(row["take_profit_pct"]):
                    active_take_profit = float(row["take_profit_pct"])

                buy_markers.append(
                    {"time": idx, "price": entry_price, "sizing_pct": pos_size * 100.0}
                )

        current_equity = cash + (shares * close_p)
        equity_list.append(current_equity)

    # Force close any open position at the last close price for fair evaluation
    if in_position and shares > 0:
        last_idx = data.index[-1]
        last_price = float(data["Close"].iloc[-1])
        gross_proceeds = shares * last_price
        exit_fee = gross_proceeds * fee_rate
        net_proceeds = gross_proceeds - exit_fee
        cash += net_proceeds
        total_fee = entry_fee + exit_fee
        pnl = net_proceeds - entry_cost
        pnl_pct = (pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0

        trades.append(
            Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=last_idx,
                exit_price=last_price,
                shares=shares,
                cost=entry_cost,
                proceeds=net_proceeds,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason="기간 종료 자동 청산",
                fee_paid=total_fee,
            )
        )
        sell_markers.append(
            {"time": last_idx, "price": last_price, "reason": "기간 종료 청산", "pnl_pct": pnl_pct}
        )
        shares = 0.0
        equity_list[-1] = cash

    equity_series = pd.Series(equity_list, index=data.index)
    final_equity = float(equity_series.iloc[-1])
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    # Win rate & profit factor
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl > 0])
    losing_trades = len([t for t in trades if t.pnl <= 0])
    win_rate_pct = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 999.0 if gross_profit > 0 else 0.0

    # Max Drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown_pct = abs(float(drawdown.min())) * 100.0 if not drawdown.empty else 0.0

    # Sharpe ratio (Hourly returns annualized to approx 252 trading days * 6.5 hours/day = 1638 bars)
    hourly_returns = equity_series.pct_change().dropna()
    if len(hourly_returns) > 1 and hourly_returns.std() > 0:
        annual_factor = np.sqrt(252 * 6.5)
        sharpe_ratio = float((hourly_returns.mean() / hourly_returns.std()) * annual_factor)
    else:
        sharpe_ratio = 0.0

    # CAGR calculation
    total_bars = len(data)
    approx_years = total_bars / (252.0 * 6.5)
    if approx_years > 0.05 and final_equity > 0:
        cagr_pct = ((final_equity / initial_capital) ** (1.0 / approx_years) - 1.0) * 100.0
    else:
        cagr_pct = total_return_pct

    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        win_rate_pct=win_rate_pct,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        cagr_pct=cagr_pct,
        trades=trades,
        equity_curve=equity_series,
        buy_markers=buy_markers,
        sell_markers=sell_markers,
    )
