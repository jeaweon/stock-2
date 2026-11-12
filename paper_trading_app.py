"""Streamlit Live Paper Trading (모의 투자) Application.

Runs real-time simulation based on the 4-step High-Return Champion DNA Momentum Strategy.
Supports live market scanning, auto/manual execution, position tracking, stop-loss triggers,
equity curve visualization, and full trade history persistence.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import load_stock_data, CACHE_DIR
from src.paper_trader import (
    load_portfolio,
    save_portfolio,
    reset_portfolio,
    execute_paper_buy,
    execute_paper_sell,
    update_market_prices,
    evaluate_live_champion_signals,
    scan_and_rebalance_portfolio,
    PORTFOLIO_FILE
)

# Page Setup
st.set_page_config(
    page_title="챔피언 DNA 모의 투자 시스템",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #475569;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.badge-buy {
    background-color: #DCFCE7;
    color: #15803D;
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: 700;
}
.badge-sell {
    background-color: #FEE2E2;
    color: #B91C1C;
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# Preset Universes
PRESET_UNIVERSES = {
    "🏆 챔피언 주도주 & 신규 고성장군 (45개)": [
        "RGTI", "RKLB", "MRNA", "WDC", "APP", "BE", "DELL", "QBTS", "OKLO", "MU",
        "HOOD", "SOXL", "ASTS", "LUNR", "VRT", "AMD", "ALAB", "MSTR", "STX", "MRVL",
        "CRWD", "IONQ", "PLTR", "LRCX", "AMAT", "NVDA", "TER", "SNOW", "CAT", "GS",
        "SE", "NEM", "RCL", "SMR", "TEM", "AVGO", "KLAC", "MCK", "CMI", "NET"
    ],
    "🎯 최근 스크리너 적격군 (BUY_READY 22개)": [
        "RBLX", "APP", "ABNB", "LULU", "SQQQ", "MRNA", "UVXY", "SPOT", "BLK", "SWKS",
        "GE", "AMZN", "IBM", "TOST", "TMO", "QRVO", "HON", "PGR", "TGT", "TSCO", "MSFT", "SBUX"
    ],
    "👀 유망 주도주 관심군 (WATCHLIST 15개)": [
        "INTU", "FSLR", "GRAB", "JOBY", "GM", "LLY", "SYK", "SOFI", "NXPI", "LEN",
        "ZTS", "NKE", "REGN", "HII", "ISRG"
    ]
}


def load_universe_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Load latest 1h candle data for a list of tickers."""
    data_dict = {}
    for sym in tickers:
        try:
            df = load_stock_data(sym)
            if len(df) >= 200:
                data_dict[sym] = df
        except Exception:
            continue
    return data_dict


# ---------------- Sidebar Controls ---------------- #
st.sidebar.title("⚙️ 모의 투자 설정")

# Portfolio Initialization
initial_capital = st.sidebar.number_input("초기 자본 ($)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
fee_rate_pct = st.sidebar.number_input("거래 수수료율 (편도 %)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
fee_rate = fee_rate_pct / 100.0

portfolio = load_portfolio(initial_cash=initial_capital, fee_rate=fee_rate)
# Update fee rate if changed in sidebar
portfolio["fee_rate"] = fee_rate

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 포트폴리오 운용 규칙")
max_positions = st.sidebar.slider("최대 동시 보유 종목 수", min_value=1, max_value=20, value=5)
auto_trade_mode = st.sidebar.checkbox("🤖 자동 모의 매매 실행 (Auto-Trade)", value=False,
                                     help="스캔 시 매수 조건 도달 종목을 자동 매수하고, 손절/데드크로스 종목을 자동 청산합니다.")

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 스캔 유니버스 선택")
univ_choice = st.sidebar.selectbox("대상 종목군", list(PRESET_UNIVERSES.keys()) + ["직접 티커 입력"])

if univ_choice == "직접 티커 입력":
    custom_input = st.sidebar.text_area("티커 입력 (쉼표 또는 공백 구분)", "NVDA, AAPL, MSFT, PLTR, TSLA, AMD, AMZN")
    active_universe = [t.strip().upper() for t in custom_input.replace(",", " ").split() if t.strip()]
else:
    active_universe = PRESET_UNIVERSES[univ_choice]

st.sidebar.caption(f"선택된 스캔 대상: 총 **{len(active_universe)}개** 종목")

st.sidebar.markdown("---")
with st.sidebar.expander("🗑️ 포트폴리오 초기화 (Reset)", expanded=False):
    st.warning("모든 보유 포지션과 매매 이력이 삭제되고 초기 자본으로 리셋됩니다.")
    if st.button("🚨 잔고 및 이력 완전 초기화"):
        portfolio = reset_portfolio(initial_cash=initial_capital, fee_rate=fee_rate)
        st.success("포트폴리오가 초기화되었습니다.")
        st.rerun()


# ---------------- Header & KPIs ---------------- #
st.markdown("<div class='main-title'>🚀 미국 주식 챔피언 DNA 실전 모의 투자 시스템</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>298개 전수 백테스트 실증 기반의 <b>고수익 챔피언 DNA(변동성 파워 + 거시 정배열 + 신고가 돌파)</b> "
    "전략으로 실시간 시장을 스캔하고 모의 매매를 수행합니다.</div>",
    unsafe_allow_html=True
)

# Calculate Core Metrics
total_cash = portfolio.get("cash", initial_capital)
positions = portfolio.get("positions", {})
invested_val = sum(pos["shares"] * pos["current_price"] for pos in positions.values())
total_equity = round(total_cash + invested_val, 2)
unrealized_pnl = sum(pos.get("unrealized_pnl", 0.0) for pos in positions.values())

trade_history = portfolio.get("trade_history", [])
realized_pnl = sum(t.get("net_pnl", 0.0) for t in trade_history)
total_return_pct = ((total_equity - portfolio.get("initial_capital", initial_capital)) / portfolio.get("initial_capital", initial_capital)) * 100.0

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)
col_kpi1.metric("💰 총 평가 자산", f"${total_equity:,.2f}", f"{total_return_pct:+.2f}%")
col_kpi2.metric("💵 보유 현금", f"${total_cash:,.2f}")
col_kpi3.metric("📈 주식 평가액", f"${invested_val:,.2f}")
col_kpi4.metric("📊 미실현 손익", f"${unrealized_pnl:+,.2f}")
col_kpi5.metric("🎯 실현 손익", f"${realized_pnl:+,.2f}")
col_kpi6.metric("🏆 보유 포지션", f"{len(positions)} / {max_positions}개")

st.markdown("---")

# ---------------- Action Bar ---------------- #
act_col1, act_col2, act_col3 = st.columns([1.5, 1.5, 3])

if act_col1.button("🔄 실시간 시장 스캔 & 모의 매매", type="primary", use_container_width=True):
    with st.spinner(f"선택된 {len(active_universe)}개 종목 실시간 데이터 수집 및 퀀트 DNA 분석 중..."):
        # Load Universe Data
        universe_data = load_universe_data(active_universe)
        # Also ensure held positions are loaded for price update
        for sym in positions.keys():
            if sym not in universe_data:
                try:
                    df_held = load_stock_data(sym)
                    if len(df_held) >= 200:
                        universe_data[sym] = df_held
                except Exception:
                    pass

        # Scan & Rebalance
        scan_results = scan_and_rebalance_portfolio(
            portfolio,
            universe_data=universe_data,
            max_positions=max_positions,
            auto_trade=auto_trade_mode
        )

        # Notify Results
        if scan_results["sell_executions"]:
            for s_msg in scan_results["sell_executions"]:
                st.warning(s_msg)
        if scan_results["buy_executions"]:
            for b_msg in scan_results["buy_executions"]:
                st.success(b_msg)
        if not scan_results["sell_executions"] and not scan_results["buy_executions"]:
            st.info("시세가 최신화되었습니다. (자동 매매 실행 내역 없음)")
        st.rerun()

if act_col2.button("⚡ 보유 포지션 시세만 즉시 업데이트", use_container_width=True):
    with st.spinner("보유 종목 시세 최신화 중..."):
        held_syms = list(positions.keys())
        if held_syms:
            curr_px = {}
            for s in held_syms:
                try:
                    df = load_stock_data(s)
                    if not df.empty:
                        curr_px[s] = float(df["Close"].iloc[-1])
                except Exception:
                    pass
            update_market_prices(portfolio, curr_px)
            save_portfolio(portfolio)
            st.success("보유 종목 시세가 업데이트되었습니다.")
            st.rerun()
        else:
            st.info("현재 보유 중인 포지션이 없습니다.")

act_col3.caption(f"🕒 마지막 업데이트: {portfolio.get('updated_at', 'N/A')[:19].replace('T', ' ')} (UTC)")

st.markdown("---")

# ---------------- Main Tabs ---------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 보유 포지션 현황",
    "🎯 실시간 매수 신호 레이더",
    "📜 체결 및 거래 이력",
    "📈 자산 변화 곡선 (Equity Curve)",
    "📖 챔피언 DNA 원칙 가이드"
])

# ---------------- Tab 1: Active Positions ---------------- #
with tab1:
    st.subheader("📌 현재 보유 포지션 (Active Holdings)")
    if positions:
        pos_rows = []
        for sym, pos in positions.items():
            cost_basis = pos["shares"] * pos["entry_price"]
            curr_val = pos["shares"] * pos["current_price"]
            pnl = pos.get("unrealized_pnl", curr_val - cost_basis)
            pnl_pct = pos.get("unrealized_pnl_pct", (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] * 100.0)

            pos_rows.append({
                "종목": sym,
                "보유 수량": f"{pos['shares']:,}주",
                "매수가": f"${pos['entry_price']:.2f}",
                "현재가": f"${pos['current_price']:.2f}",
                "평가액": f"${curr_val:,.2f}",
                "미실현 손익": f"${pnl:+,.2f}",
                "손익률": f"{pnl_pct:+.2f}%",
                "12% 손절가": f"${pos['stop_loss_price']:.2f}",
                "진입 일시": pos.get("entry_time", "")[:16].replace("T", " "),
                "매수 사유": pos.get("entry_reason", "")
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

        st.markdown("#### 🚨 보유 종목 수동 청산 (Manual Sell)")
        c_sel, c_btn = st.columns([2, 1])
        sell_target = c_sel.selectbox("수동 청산할 종목 선택", list(positions.keys()))
        if c_btn.button(f"🛑 [{sell_target}] 전량 즉시 매도", type="secondary"):
            px = positions[sell_target]["current_price"]
            ok, msg = execute_paper_sell(portfolio, sell_target, price=px, reason="사용자 수동 청산")
            if ok:
                save_portfolio(portfolio)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        st.info("현재 보유 중인 포지션이 없습니다. 상단의 '실시간 시장 스캔' 버튼을 눌러 유망 종목을 매수해보세요.")


# ---------------- Tab 2: Signal Radar ---------------- #
with tab2:
    st.subheader("🎯 실시간 챔피언 DNA 매수 신호 레이더")
    st.caption("10일/30일 거시 정배열 + 200시간선 상단 + ATR 변동성(≥1.1%) + 신고가 돌파권(-6% 이내)을 모두 만족하는 종목을 실시간 탐지합니다.")

    with st.spinner("레이더 신호 분석 중..."):
        radar_data = load_universe_data(active_universe[:30])  # Scan top 30 for speed
        signals_list = []
        for sym, df in radar_data.items():
            sig = evaluate_live_champion_signals(df, sym)
            is_held = sym in positions
            signals_list.append({
                "종목": sym,
                "상태": "보유중" if is_held else ("매수추천" if sig["is_buy_signal"] else "관찰대기"),
                "현재가": f"${sig['current_price']:.2f}",
                "시간당 ATR": f"{sig['atr_pct']:.2f}%",
                "신고가 대비": f"-{sig['dist_from_high_pct']:.1f}%",
                "10일선 / 30일선": f"${sig['ema_10d']:.1f} / ${sig['ema_30d']:.1f}",
                "200시간선": f"${sig['ema_200h']:.1f}",
                "진단 상세": sig["rationale"],
                "_buy_ok": sig["is_buy_signal"] and not is_held,
                "_px": sig["current_price"]
            })

    radar_df = pd.DataFrame(signals_list)
    if not radar_df.empty:
        # Sort BUY recommended first
        radar_df = radar_df.sort_values(by=["_buy_ok", "시간당 ATR"], ascending=[False, False])
        display_df = radar_df.drop(columns=["_buy_ok", "_px"])
        st.dataframe(display_df, use_container_width=True, height=350)

        # 1-Click Buy Execution
        buyable = radar_df[radar_df["_buy_ok"]]
        if not buyable.empty:
            st.markdown("#### 🛒 1-Click 수동 매수 실행")
            col_b1, col_b2, col_b3 = st.columns([2, 1.5, 1])
            chosen_sym = col_b1.selectbox("매수할 종목 선택", buyable["종목"].tolist())
            chosen_px = float(buyable[buyable["종목"] == chosen_sym]["_px"].iloc[0])
            def_alloc = min(total_cash, round(total_cash / max(1, (max_positions - len(positions))), 2))
            alloc_amt = col_b2.number_input("투자 금액 ($)", min_value=100.0, max_value=float(max(100.0, total_cash)), value=float(max(100.0, def_alloc)), step=100.0)

            if col_b3.button(f"🚀 {chosen_sym} 즉시 매수", type="primary"):
                ok, msg = execute_paper_buy(portfolio, chosen_sym, price=chosen_px, allocation_amount=alloc_amt)
                if ok:
                    save_portfolio(portfolio)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.caption("현재 즉시 매수 기준을 충족하는 미보유 종목이 없습니다.")


# ---------------- Tab 3: Trade History ---------------- #
with tab3:
    st.subheader("📜 모의 투자 완료 체결 이력 (Closed Trades)")
    if trade_history:
        th_df = pd.DataFrame(trade_history)

        # Summary KPIs
        total_closed = len(th_df)
        wins = (th_df["net_pnl"] > 0).sum()
        win_rate = (wins / total_closed) * 100.0
        total_fees = th_df["total_fees"].sum()
        gross_profits = th_df[th_df["net_pnl"] > 0]["net_pnl"].sum()
        gross_losses = abs(th_df[th_df["net_pnl"] < 0]["net_pnl"].sum())
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 99.0

        c_th1, c_th2, c_th3, c_th4 = st.columns(4)
        c_th1.metric("총 매매 완료 건수", f"{total_closed}건")
        c_th2.metric("실전 승률", f"{win_rate:.1f}% ({wins}/{total_closed})")
        c_th3.metric("손익비 (Profit Factor)", f"{profit_factor:.2f}")
        c_th4.metric("누적 지출 수수료", f"${total_fees:,.2f}")

        # Formatted Display Table
        disp_th = th_df[[
            "trade_id", "ticker", "shares", "entry_price", "exit_price",
            "net_pnl", "pnl_pct", "total_fees", "exit_reason", "entry_time", "exit_time"
        ]].copy()
        disp_th.columns = [
            "거래 ID", "종목", "수량", "매수가", "매도가",
            "실현 손익", "손익률(%)", "수수료", "청산 사유", "진입 일시", "청산 일시"
        ]
        st.dataframe(disp_th.iloc[::-1], use_container_width=True)

        csv_bytes = disp_th.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 거래 이력 CSV 다운로드",
            data=csv_bytes,
            file_name="paper_trade_history.csv",
            mime="text/csv"
        )
    else:
        st.info("아직 완료된 거래 이력이 없습니다.")


# ---------------- Tab 4: Equity Curve ---------------- #
with tab4:
    st.subheader("📈 자산 변화 곡선 (Equity Curve)")
    equity_hist = portfolio.get("equity_history", [])
    if len(equity_hist) >= 2:
        eq_df = pd.DataFrame(equity_hist)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"],
            y=eq_df["total_equity"],
            mode="lines+markers",
            name="총 평가 자산 ($)",
            line=dict(color="#2563EB", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"],
            y=[portfolio["initial_capital"]] * len(eq_df),
            mode="lines",
            name="원금 기준선 ($)",
            line=dict(color="#94A3B8", dash="dash")
        ))
        fig.update_layout(
            title="포트폴리오 자산 성장 추이",
            xaxis_title="일시",
            yaxis_title="평가액 ($)",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("시뮬레이션 스캔 또는 매매가 진행되면 자산 변화 추이 그래프가 생성됩니다.")


# ---------------- Tab 5: Champion DNA Rules ---------------- #
with tab5:
    st.subheader("📖 고수익 챔피언 DNA 모멘텀 매매 규칙")
    st.markdown(r"""
    #### 1. 매수 진입 원칙 (4단계 퀀트 필터)
    - **거시 골든크로스**: 10일 거시선(65시간봉) > 30일 거시선(195시간봉)
    - **장기 추세 지지**: 현재 주가 > 200시간선(200시간봉)
    - **변동성 파워**: 시간당 변동성(ATR 14) $\ge 1.1\%$ (저변동 소외주 진입 방지)
    - **신고가 돌파**: 최근 60봉(약 10거래일) 최고가 대비 **-6% 이내** 안착

    #### 2. 청산 및 손절 원칙
    - **거시 추세 청산**: 10일선 $\le$ 30일선 데드크로스 발생 시 전량 청산 (하락장 100% 현금 방어)
    - **와이드 하드스탑**: **-12%** 터치 시 손절 (개장 첫 봉 오버나이트 갭하락 시 개장 시가로 체결 반영)
    - **실거래 수수료**: 편도 0.1% (매수/매도 각각 차감)
    """)
