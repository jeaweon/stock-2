"""Streamlit Results Viewer & Analysis Dashboard for Multi-Stock Investigations.

Dynamically displays results from backtest_history.csv and scanner_results.csv
with side-by-side Before vs After AI feedback comparisons, interactive candlestick charts,
equity curve overlays, trade logs, and universe rankings.
"""
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_stock_data
from src.backtester import run_backtest, BacktestResult
from src.gemini_agents import compile_strategy_code
from src.storage import load_history_records, clear_history, DEFAULT_HISTORY_FILE

# Page Configuration
st.set_page_config(
    page_title="Stock AI Backtest Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .pass-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .fail-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .delta-positive {
        color: #16A34A;
        font-weight: 700;
    }
    .delta-negative {
        color: #DC2626;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def plot_backtest_chart(df: pd.DataFrame, result: BacktestResult, title: str = "1시간 봉 차트 및 매매 타점"):
    """Create interactive Plotly candlestick chart with buy/sell markers."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{title} (가격 및 매매 타점)", "거래량 (Volume)"),
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444",
        ),
        row=1,
        col=1,
    )

    # 2. Buy markers
    if result.buy_markers:
        buy_times = [m["time"] for m in result.buy_markers]
        buy_prices = [m["price"] for m in result.buy_markers]
        buy_hover = [f"매수 진입<br>가격: ${m['price']:.2f}<br>투자비중: {m.get('sizing_pct', 100):.1f}%" for m in result.buy_markers]
        fig.add_trace(
            go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode="markers",
                marker=dict(symbol="triangle-up", size=11, color="#10B981", line=dict(width=1, color="black")),
                name="매수 진입",
                hovertext=buy_hover,
                hoverinfo="text+x",
            ),
            row=1,
            col=1,
        )

    # 3. Sell markers
    if result.sell_markers:
        sell_times = [m["time"] for m in result.sell_markers]
        sell_prices = [m["price"] for m in result.sell_markers]
        sell_hover = [
            f"매도 청산 ({m.get('reason', '')})<br>가격: ${m['price']:.2f}<br>손익률: {m.get('pnl_pct', 0.0):+.2f}%"
            for m in result.sell_markers
        ]
        fig.add_trace(
            go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode="markers",
                marker=dict(symbol="triangle-down", size=11, color="#EF4444", line=dict(width=1, color="black")),
                name="매도 청산",
                hovertext=sell_hover,
                hoverinfo="text+x",
            ),
            row=1,
            col=1,
        )

    # 4. Volume Bar
    colors = ["#10B981" if c >= o else "#EF4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=540,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def plot_equity_overlay(baseline_series: pd.Series, improved_series: Optional[pd.Series] = None):
    """Plot overlay of portfolio equity before vs after feedback."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baseline_series.index,
            y=baseline_series.values,
            mode="lines",
            name="수정 전 (Before Feedback)",
            line=dict(color="#64748B", width=2, dash="dot"),
        )
    )

    if improved_series is not None:
        fig.add_trace(
            go.Scatter(
                x=improved_series.index,
                y=improved_series.values,
                mode="lines",
                name="수정 후 (After AI Feedback)",
                line=dict(color="#10B981", width=2.5),
            )
        )

    fig.update_layout(
        title="포트폴리오 자산 가치 변화 ($) - 피드백 전후 비교",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        yaxis_title="자산 총액 ($)",
        xaxis_title="일시",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# Title
st.markdown("<div class='main-header'>📊 주식 백테스트 & AI 피드백 성과 분석 대시보드</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Antigravity 에이전트가 실행한 수많은 종목의 백테스트 결과를 조회하고, "
    "<b>AI 피드백 전후의 승률, 수익률, MDD, 타점 차트</b>를 상세하게 비교 분석합니다.</div>",
    unsafe_allow_html=True,
)

# Load History Data Dynamically
history_df = load_history_records()
scanner_file = Path("scanner_results.csv")
scanner_df = pd.read_csv(scanner_file, encoding="utf-8-sig") if scanner_file.exists() else pd.DataFrame()

# Extract Available Tickers Dynamically (NO HARDCODING)
available_tickers = []
if not history_df.empty:
    available_tickers = sorted(history_df["ticker"].dropna().unique().tolist())
if not scanner_df.empty:
    available_tickers = sorted(list(set(available_tickers + scanner_df["ticker"].dropna().unique().tolist())))

# Sidebar Controls
st.sidebar.title("🔍 종목 검색 및 필터")
st.sidebar.caption(f"현재 데이터베이스에 총 **{len(available_tickers)}개**의 종목이 누적 저장되어 있습니다.")

st.sidebar.markdown("---")
with st.sidebar.expander("🌟 [공식 승인 3호] 고수익 챔피언 DNA 모멘텀 전략", expanded=True):
    st.markdown("**초고수익 주도주 DNA 4단계 필터링형**")
    st.caption("• 45개 주도주 검증: 평균 **+289.43%** (플러스 수익률 **95.6%**)")
    st.caption("• 10/30일 거시 정배열 + 200시간선 상단 + ATR 변동성(≥1.1%) + 신고가권(-6%) 돌파")
    st.caption("• 10/30일 거시 데드크로스 전량 청산 + 12% 와이드 하드스탑 (갭하락 반영)")
    st.caption("• RGTI(+1950.4%), RKLB(+1533.0%), MRNA(+791.1%), WDC(+761.9%), APP(+603.7%)")

with st.sidebar.expander("🌟 [공식 승인 2호] 추세 가속 & 잔파동 휩쏘 차단형", expanded=False):
    st.markdown("**Fast 10일 / Slow 30일 EMA & 200시간선 필터**")
    st.caption("• 80개 종목 검증: 평균 **+20.59%** (+7.48%p 개선)")
    st.caption("• 10일 EMA > 30일 EMA 골든크로스 + 200 EMA 위 100% 진입")
    st.caption("• MSTR(+243.5%), HII(+71.5%), MRVL(+186.2%), LULU(+54.4%p 방어)")

with st.sidebar.expander("🌟 [공식 승인 1호] 거시 추세 라이딩 & 폭락 차단형", expanded=False):
    st.markdown("**20일/50일/100일 EMA & 200 EMA 위 돌파**")
    st.caption("• 16개 주식 평균: **+89.98%** (B&H -5.55% 대비 **+95.53%p 압승**)")
    st.caption("• 16개 중 14개 종목(87.5%) B&H 초과")
    st.caption("• 무제한 익절 라이딩 + 거시 데드크로스 전량 청산")
st.sidebar.markdown("---")

if available_tickers:
    selected_ticker = st.sidebar.selectbox("조회할 종목 선택", options=available_tickers, index=0)
else:
    selected_ticker = st.sidebar.text_input("조회할 종목 티커 입력 (예: AAPL, NVDA)", value="AAPL").upper().strip()

# Custom input option in sidebar
custom_search = st.sidebar.text_input("직접 티커 검색/입력 (예: TSLA, AMD, SPY)").upper().strip()
if custom_search:
    selected_ticker = custom_search

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 차트 및 데이터 설정")
initial_capital = st.sidebar.number_input("초기 자본 ($)", value=10000.0, step=1000.0)
fee_rate = st.sidebar.number_input("거래 수수료율 (%)", value=0.1, step=0.01) / 100.0

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 종목별 피드백 전후 상세 비교",
    "🏆 전체 종목 조사 리더보드 (Rankings)",
    "🎯 고수익 챔피언 DNA 스크리너",
    "📁 원본 데이터베이스 (backtest_history.csv)",
])


# ---------------- Tab 1: Detailed Before vs After Comparison ---------------- #
with tab1:
    st.subheader(f"🔍 [{selected_ticker}] AI 피드백 전후 성과 심층 비교")

    # Filter history for selected ticker
    ticker_history = history_df[history_df["ticker"] == selected_ticker].sort_values("iteration_round") if not history_df.empty else pd.DataFrame()

    if ticker_history.empty:
        st.warning(f"'{selected_ticker}'에 대한 기존 백테스트 기록이 아직 없습니다. Antigravity에서 먼저 실행하거나 아래 버튼으로 바로 백테스트할 수 있습니다.")
        if st.button(f"🚀 {selected_ticker} 즉시 백테스트 실행"):
            with st.spinner(f"{selected_ticker} 데이터 수집 및 백테스트 실행 중..."):
                from src.adaptive_experiment import run_single_asset_experiment
                run_single_asset_experiment(selected_ticker, initial_capital=initial_capital, fee_rate=fee_rate)
                st.rerun()
    else:
        # Get baseline row (round 0 or minimum iteration)
        baseline_rows = ticker_history[ticker_history["iteration_round"] == 0]
        if baseline_rows.empty:
            baseline_row = ticker_history.iloc[0]
        else:
            baseline_row = baseline_rows.iloc[-1]

        # Get best improved row
        improved_rows = ticker_history[ticker_history["iteration_round"] > 0]
        if not improved_rows.empty:
            # Pick highest return or latest round
            best_improved_row = improved_rows.sort_values("total_return_pct", ascending=False).iloc[0]
        else:
            best_improved_row = baseline_row

        # Calculate Deltas
        b_ret = float(baseline_row["total_return_pct"])
        a_ret = float(best_improved_row["total_return_pct"])
        ret_diff = a_ret - b_ret

        b_win = float(baseline_row["win_rate_pct"])
        a_win = float(best_improved_row["win_rate_pct"])
        win_diff = a_win - b_win

        b_mdd = float(baseline_row["mdd_pct"])
        a_mdd = float(best_improved_row["mdd_pct"])
        mdd_diff = a_mdd - b_mdd  # negative is good

        b_pf = float(baseline_row["profit_factor"])
        a_pf = float(best_improved_row["profit_factor"])
        pf_diff = a_pf - b_pf

        b_trades = int(baseline_row["total_trades"])
        a_trades = int(best_improved_row["total_trades"])

        # Display Top Metrics Grid
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric(
            "총 수익률 (Total Return)",
            f"{a_ret:+.2f}%",
            delta=f"{ret_diff:+.2f}%p (수정 전: {b_ret:+.2f}%)",
        )
        col_m2.metric(
            "승률 (Win Rate)",
            f"{a_win:.1f}%",
            delta=f"{win_diff:+.1f}%p (수정 전: {b_win:.1f}%)",
        )
        col_m3.metric(
            "최대 낙폭 (MDD)",
            f"{a_mdd:.2f}%",
            delta=f"{mdd_diff:+.2f}%p (수정 전: {b_mdd:.2f}%)",
            delta_color="inverse",
        )
        col_m4.metric(
            "손익 팩터 (Profit Factor)",
            f"{a_pf:.2f}",
            delta=f"{pf_diff:+.2f} (수정 전: {b_pf:.2f})",
        )
        col_m5.metric(
            "총 거래 횟수",
            f"{a_trades}회",
            delta=f"{a_trades - b_trades:+d}회 (수정 전: {b_trades}회)",
        )

        st.markdown("---")

        # Two Column Comparison Box
        col_box1, col_box2 = st.columns(2)
        with col_box1:
            st.markdown("#### 🔴 수정 전 (Baseline Strategy)")
            st.markdown(f"**전략명:** `{baseline_row['strategy_name']}`")
            st.info(f"**매수/매도 규칙:**\n\n{baseline_row['buy_rules']}")
            st.caption(f"초기 평가: {baseline_row['critic_feedback']}")

        with col_box2:
            st.markdown("#### 🟢 수정 후 (AI Optimized Strategy)")
            st.markdown(f"**전략명:** `{best_improved_row['strategy_name']}`")
            st.success(f"**개선된 매수/매도 규칙:**\n\n{best_improved_row['buy_rules']}")
            verdict = best_improved_row["critic_verdict"]
            badge_class = "pass-badge" if verdict == "PASS" else "fail-badge"
            st.markdown(
                f"**심사역(Critic) 판정:** <span class='{badge_class}'>{verdict}</span>",
                unsafe_allow_html=True,
            )
            with st.expander("📝 Critic 상세 평가 의견 보기", expanded=False):
                st.write(best_improved_row["critic_feedback"])

        # Re-run for interactive chart visualization
        st.markdown("---")
        st.markdown("### 📊 차트 분석 및 매매 타점 시각화")

        with st.spinner("1시간 봉 데이터 로드 및 시각화 준비 중..."):
            try:
                df_stock = load_stock_data(selected_ticker, period="730d", interval="1h")
                
                # Run baseline backtest
                base_fn = compile_strategy_code(baseline_row["strategy_code"]) if pd.notna(baseline_row["strategy_code"]) and "def generate_signals" in str(baseline_row["strategy_code"]) else None
                imp_fn = compile_strategy_code(best_improved_row["strategy_code"]) if pd.notna(best_improved_row["strategy_code"]) and "def generate_signals" in str(best_improved_row["strategy_code"]) else None

                res_base = run_backtest(df_stock, signal_func=base_fn, initial_capital=initial_capital, fee_rate=fee_rate) if base_fn else None
                res_imp = run_backtest(df_stock, signal_func=imp_fn, initial_capital=initial_capital, fee_rate=fee_rate) if imp_fn else None

                # Equity Curve Comparison
                if res_base and res_imp:
                    st.plotly_chart(
                        plot_equity_overlay(res_base.equity_curve, res_imp.equity_curve),
                        width="stretch",
                    )
                
                # Candlestick with Improved Markers
                active_res = res_imp if res_imp else res_base
                if active_res:
                    st.plotly_chart(
                        plot_backtest_chart(df_stock, active_res, title=f"{selected_ticker} 1시간 봉 매매 시그널 분석"),
                        width="stretch",
                    )

                    # Trade Log Table
                    if active_res.trades:
                        st.markdown("#### 📜 최근 20건 매매 체결 내역")
                        t_rows = [
                            {
                                "진입 일시": str(t.entry_time),
                                "진입 가격": f"${t.entry_price:.2f}",
                                "청산 일시": str(t.exit_time),
                                "청산 가격": f"${t.exit_price:.2f}",
                                "손익 ($)": f"${t.pnl:+,.2f}",
                                "손익률 (%)": f"{t.pnl_pct:+.2f}%",
                                "청산 사유": t.exit_reason,
                            }
                            for t in active_res.trades[-20:]
                        ]
                        st.dataframe(pd.DataFrame(t_rows), width="stretch")
            except Exception as e:
                st.error(f"차트 렌더링 중 오류 발생: {e}")


# ---------------- Tab 2: Rankings & Leaderboard ---------------- #
with tab2:
    st.subheader("🏆 전체 조사 종목 리더보드 (Rankings & Comparison Table)")
    st.caption("지금까지 Antigravity로 조사한 모든 종목의 피드백 전후 성과를 한눈에 비교하고 랭킹을 확인합니다.")

    if not scanner_df.empty:
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("총 조사 종목 수", f"{len(scanner_df)}개")
        col_kpi2.metric("평균 수익률 개선폭", f"{scanner_df['return_improvement_pctp'].mean():+.2f}%p")
        col_kpi3.metric("최고 수익률 종목", f"{scanner_df.loc[scanner_df['after_return_pct'].idxmax()]['ticker']} ({scanner_df['after_return_pct'].max():+.1f}%)")
        col_kpi4.metric("최대 수익률 개선", f"{scanner_df.loc[scanner_df['return_improvement_pctp'].idxmax()]['ticker']} ({scanner_df['return_improvement_pctp'].max():+.1f}%p)")

        # Sort options
        col_s1, col_s2 = st.columns([1, 1])
        sort_by = col_s1.selectbox(
            "정렬 기준",
            options=["after_return_pct", "return_improvement_pctp", "after_win_rate_pct", "after_mdd_pct"],
            format_func=lambda x: {
                "after_return_pct": "수정 후 최종 수익률 높은 순",
                "return_improvement_pctp": "수익률 개선폭(%p) 높은 순",
                "after_win_rate_pct": "승률 높은 순",
                "after_mdd_pct": "MDD(최대낙폭) 낮은 순",
            }[x],
        )
        ascending = True if sort_by == "after_mdd_pct" else False
        display_scanner_df = scanner_df.sort_values(sort_by, ascending=ascending)

        # Styled Table
        st.dataframe(
            display_scanner_df[[
                "ticker", "group", "group_name",
                "before_return_pct", "after_return_pct", "return_improvement_pctp",
                "before_win_rate_pct", "after_win_rate_pct", "win_rate_improvement_pctp",
                "before_mdd_pct", "after_mdd_pct", "after_profit_factor", "critic_verdict", "critic_score"
            ]],
            width="stretch",
            height=450,
        )

        csv_bytes = scanner_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 전체 조사 결과 CSV 다운로드 (scanner_results.csv)",
            data=csv_bytes,
            file_name="scanner_results.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("아직 대규모 조사 결과 파일(scanner_results.csv)이 생성되지 않았습니다. Antigravity에서 `python run_batch.py`를 실행해보세요.")


# ---------------- Tab 3: High-Return Champion DNA Screener ---------------- #
with tab3:
    st.subheader("🎯 고수익 챔피언 DNA 스크리너 & 매수 후보 발굴 파이프라인")
    st.caption("298개 종목 백테스트 실증 분석을 통해 발견된 4대 핵심 승리 공식(변동성 에너지 + 기관 수급 갭 + 거시 정배열 + 추세 탄력성)을 실시간 검증하여 유망 매수 후보를 자동 발굴합니다.")

    screener_file = Path("screener_qualified_candidates.csv")
    if screener_file.exists():
        sc_df = pd.read_csv(screener_file)

        # KPI Cards
        sc_ready = sc_df[sc_df["status"] == "BUY_READY"]
        sc_watch = sc_df[sc_df["status"] == "WATCHLIST"]
        sc_dis = sc_df[sc_df["status"] == "DISQUALIFIED"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 분석 종목", f"{len(sc_df)}개")
        k2.metric("🏆 즉시 매수 가능 (BUY_READY)", f"{len(sc_ready)}개")
        k3.metric("👀 눌림목 관심종목 (WATCHLIST)", f"{len(sc_watch)}개")
        k4.metric("❌ 역배열/저변동성 탈락", f"{len(sc_dis)}개")

        st.markdown("---")

        # DNA Feature Comparison Expander
        with st.expander("🧬 298개 종목 전수조사: 초고수익주(≥100%) vs 손실군(<0%) 4대 특성 통계", expanded=True):
            dna_comp_data = {
                "분석 지표": [
                    "평균 2년 전략 수익률", "평균 2년 B&H 수익률", "평균 승률", 
                    "평균 거래 횟수", "평균 변동성 (ATR %)", "기관 수급 갭 빈도 (Overnight Gap)",
                    "평균 손익비 (Profit Factor)", "평균 최대낙폭 (MDD)"
                ],
                "초고수익군 (≥100%, 37개)": [
                    "+345.68%", "+341.03%", "44.39%", 
                    "11.68회 (최소 매매)", "1.69% (강력)", "48.26% (기관 주도)",
                    "7.98", "42.84%"
                ],
                "우수군 (30~100%, 69개)": [
                    "+54.50%", "+87.88%", "43.21%", 
                    "12.38회", "0.99%", "27.75%",
                    "19.17", "28.79%"
                ],
                "중위군 (0~30%, 81개)": [
                    "+13.00%", "+42.07%", "35.14%", 
                    "13.75회", "0.85% (저변동)", "21.17%",
                    "1.41", "30.23%"
                ],
                "손실군 (<0%, 111개)": [
                    "-24.38%", "-3.01% (침체/하락)", "23.74%", 
                    "16.00회 (잦은 휩쏘)", "1.09%", "28.40%",
                    "0.55", "43.43%"
                ]
            }
            st.dataframe(pd.DataFrame(dna_comp_data), width="stretch")

        st.markdown("### 📋 실시간 퀀트 스크리닝 결과")
        status_filter = st.radio("상태별 필터링", ["전체", "BUY_READY (즉시 매수)", "WATCHLIST (눌림목 대기)", "DISQUALIFIED (탈락)"], horizontal=True)

        filter_map = {
            "전체": None,
            "BUY_READY (즉시 매수)": "BUY_READY",
            "WATCHLIST (눌림목 대기)": "WATCHLIST",
            "DISQUALIFIED (탈락)": "DISQUALIFIED",
        }

        view_df = sc_df.copy()
        target_status = filter_map[status_filter]
        if target_status:
            view_df = view_df[view_df["status"] == target_status]

        st.dataframe(
            view_df[["ticker", "status", "score", "atr_pct", "gap_freq_pct", "dist_from_high_pct", "above_200h", "golden_aligned", "rationale"]],
            width="stretch",
            height=400,
        )

        sc_csv = sc_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 스크리너 전체 결과 CSV 다운로드 (screener_qualified_candidates.csv)",
            data=sc_csv,
            file_name="screener_qualified_candidates.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("스크리너 결과 파일(screener_qualified_candidates.csv)이 없습니다. 터미널에서 `python run_screener.py`를 실행하세요.")


# ---------------- Tab 4: Raw Historical Database ---------------- #
with tab4:
    st.subheader("📁 영구 누적 데이터베이스 (backtest_history.csv)")
    st.caption("모든 전략 반복 회차, 생성된 파이썬 코드, 심사위원 상세 의견 전문이 기록되어 있습니다.")

    if not history_df.empty:
        col_f1, col_f2 = st.columns(2)
        all_syms = ["전체"] + sorted(history_df["ticker"].dropna().unique().tolist())
        filter_sym = col_f1.selectbox("종목 필터링", all_syms, index=0)
        filter_ver = col_f2.selectbox("심사 판정 필터", ["전체", "PASS", "FAIL", "BASELINE"], index=0)

        f_df = history_df.copy()
        if filter_sym != "전체":
            f_df = f_df[f_df["ticker"] == filter_sym]
        if filter_ver != "전체":
            f_df = f_df[f_df["critic_verdict"] == filter_ver]

        st.dataframe(
            f_df[[
                "timestamp", "ticker", "iteration_round", "strategy_name",
                "total_return_pct", "win_rate_pct", "mdd_pct", "total_trades",
                "profit_factor", "critic_verdict", "buy_rules", "critic_feedback"
            ]],
            width="stretch",
            height=420,
        )

        col_d1, col_d2 = st.columns(2)
        raw_csv = history_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        col_d1.download_button(
            label="📥 누적 이력 전체 다운로드 (backtest_history.csv)",
            data=raw_csv,
            file_name="backtest_history.csv",
            mime="text/csv",
            width="stretch",
        )
        if col_d2.button("🗑️ 이력 데이터 초기화"):
            clear_history()
            st.success("이력이 초기화되었습니다.")
            st.rerun()
    else:
        st.info("저장된 이력이 없습니다.")
