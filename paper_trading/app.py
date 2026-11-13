"""Streamlit 1-Hour Live Paper Trading Application (KRW 10,000,000 Base).

Standalone paper trading dashboard modeled after https://github.com/jeaweon/stock-
with 1-hour candle API screening (Champion DNA 4-step screener),
Promising Strategy #3 execution, KRW FX conversion, extended-hours 1m overlay,
20-slot integer-share sizing, 0.1% one-way fees, and GitHub Gist cloud persistence.
"""
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Ensure local imports work regardless of whether run from repo root or paper_trading/
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from engine import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE_KRW,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_SAVE_PATH,
    DnaEvaluation,
    SCAN_UNIVERSES,
    SimulatedTrader,
    apply_extended_hours_price,
    convert_usd_to_krw_1h,
    evaluate_champion_dna_1h,
    fetch_live_1h_batch,
    fetch_market_indices,
    fetch_usd_krw_hourly,
    normalize_tickers,
    standardize_ohlcv_df,
)

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="1시간봉 챔피언 DNA 실전 모의투자", layout="wide")
st.title("⚡ 1시간봉 챔피언 DNA 실전 모의투자 시스템")
st.caption(
    "🔄 데이터 기준: 1시간 봉(1h) + 장외 실시간 체결가 반영 · 1,000만원 원화 환산 · "
    "최대 20종목 정수 분할 · 편도 수수료 0.1% · 매매일지 & GitHub Gist 영구 보존"
)


def _get_secret(key: str, default: str = "") -> str:
    try:
        val = st.secrets.get(key, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()


# ---------------------------------------------------------
# Cached Live Data Helpers (Cleared on Manual Refresh / Scan)
# ---------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def cached_fx_hourly() -> pd.Series:
    return fetch_usd_krw_hourly(period="100d")


@st.cache_data(ttl=300, show_spinner=False)
def cached_market_indices() -> Dict[str, Dict[str, float]]:
    return fetch_market_indices()


@st.cache_data(ttl=600, show_spinner=False)
def cached_single_stock_1h(ticker: str, apply_ext: bool = True) -> Optional[pd.DataFrame]:
    sym = ticker.upper().strip()
    try:
        raw = yf.Ticker(sym).history(period="90d", interval="1h", auto_adjust=True)
        if raw is None or raw.empty:
            return None
        df_usd = standardize_ohlcv_df(raw)
        if apply_ext:
            df_usd = apply_extended_hours_price(sym, df_usd)
        return df_usd
    except Exception:
        return None


# ---------------------------------------------------------
# Session Initialization (SimulatedTrader + Gist Sync)
# ---------------------------------------------------------
gist_id = _get_secret("GIST_ID")
github_token = _get_secret("GITHUB_TOKEN")

if "trader" not in st.session_state:
    st.session_state.trader = SimulatedTrader(
        initial_balance=DEFAULT_INITIAL_BALANCE_KRW,
        save_file=str(DEFAULT_SAVE_PATH),
        max_positions=DEFAULT_MAX_POSITIONS,
        fee_rate=DEFAULT_FEE_RATE,
        gist_id=gist_id,
        github_token=github_token,
    )

trader: SimulatedTrader = st.session_state.trader

if "scan_usd_cache" not in st.session_state:
    st.session_state.scan_usd_cache = {}
if "scan_krw_cache" not in st.session_state:
    st.session_state.scan_krw_cache = {}
if "scan_evals" not in st.session_state:
    st.session_state.scan_evals = {}
if "last_scan_summary" not in st.session_state:
    st.session_state.last_scan_summary = None
if "action_notices" not in st.session_state:
    st.session_state.action_notices = []


# ---------------------------------------------------------
# Sidebar Controls (Matching stock- Backup/Restore/Reset/Gist Flow)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 시스템 관리")

    if st.button("📈 보유 종목 · 지수 최신화", width="stretch"):
        cached_fx_hourly.clear()
        cached_market_indices.clear()
        cached_single_stock_1h.clear()
        st.session_state.action_notices.append(("success", "최신 환율, 지수 및 보유 종목 실시간(장외 포함) 시세를 불러왔습니다!"))
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 API 종목 검색(스크리너) 설정")
    univ_label = st.selectbox("검색 대상 종목 풀", list(SCAN_UNIVERSES.keys()))
    extra_tickers_raw = st.text_area(
        "추가 검색 티커 (쉼표/공백 구분)",
        "",
        placeholder="예: TSLA, ARM, OKLO",
    )
    extra_list = normalize_tickers(extra_tickers_raw.replace(",", " ").split()) if extra_tickers_raw.strip() else []
    scan_universe = normalize_tickers(SCAN_UNIVERSES[univ_label] + extra_list)
    st.caption(f"🔎 총 검색 대상: **{len(scan_universe)}개 종목**")

    max_pos_input = st.slider("최대 동시 보유 종목 (균등 분할)", min_value=1, max_value=30, value=int(trader.max_positions))
    fee_pct_input = st.number_input("편도 거래 수수료 (%)", min_value=0.0, max_value=1.0, value=round(trader.fee_rate * 100.0, 2), step=0.01)
    if max_pos_input != trader.max_positions or abs((fee_pct_input / 100.0) - trader.fee_rate) > 1e-6:
        trader.max_positions = int(max_pos_input)
        trader.fee_rate = float(fee_pct_input / 100.0)
        trader._save_to_file()

    st.caption(
        f"💡 기준 ₩{int(trader.initial_balance):,} ÷ {trader.max_positions}종목 = "
        f"슬롯당 약 **₩{int(trader.initial_balance / trader.max_positions):,}** (소수점 불가 · 1주 이상 정수)"
    )

    st.markdown("---")
    st.markdown("### 💾 데이터 백업 및 복구")
    st.caption("수동 백업용: 내 컴퓨터에 저장하고 다시 불러오기")

    state_json_str = json.dumps(trader._build_state_dict(), ensure_ascii=False, indent=4)
    st.download_button(
        label="📥 현재 계좌 상태 다운로드 (수동 백업)",
        data=state_json_str.encode("utf-8"),
        file_name="trader_state.json",
        mime="application/json",
        help="클라우드 동기화와 별개로 내 컴퓨터에 수동으로 백업할 때 사용하세요.",
        width="stretch",
    )

    uploaded_file = st.file_uploader("📤 수동 백업 파일 복구", type=["json"])
    if uploaded_file is not None:
        if st.button("수동 데이터 복구 실행", width="stretch"):
            ok_restore, r_msg = trader.restore_from_json_bytes(uploaded_file.getvalue())
            if ok_restore:
                st.session_state.action_notices.append(("success", f"백업 데이터 복구 완료: {r_msg}"))
                st.rerun()
            else:
                st.error(r_msg)

    st.markdown("---")
    if st.button("🔄 모의투자 계좌 초기화 (완전 리셋)", width="stretch"):
        err_rst = trader.reset_account(new_initial_balance=DEFAULT_INITIAL_BALANCE_KRW)
        st.session_state.scan_evals = {}
        st.session_state.last_scan_summary = None
        msg_rst = f"모의투자 계좌가 ₩{DEFAULT_INITIAL_BALANCE_KRW:,}원으로 완전 초기화되었습니다."
        if err_rst:
            msg_rst += f" (Gist 참고: {err_rst})"
        st.session_state.action_notices.append(("info", msg_rst))
        st.rerun()

    if st.button("🔍 클라우드 연동 상태 체크", width="stretch"):
        ok_gist, gist_msg = trader.check_gist_connection()
        if ok_gist:
            st.sidebar.success(gist_msg)
        else:
            st.sidebar.error(gist_msg)


# ---------------------------------------------------------
# Display Action Notices & Primary API Scan Button
# ---------------------------------------------------------
if st.session_state.action_notices:
    for kind, msg in st.session_state.action_notices:
        if kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        elif kind == "error":
            st.error(msg)
        else:
            st.info(msg)
    st.session_state.action_notices = []

act_col1, act_col2 = st.columns([2.3, 2.7])

if act_col1.button("🚀 주식 API 1시간봉 종목 검색 & 자동 매수/매도 실행", type="primary", width="stretch"):
    prog = st.progress(0.0, text="실시간 환율(KRW=X) 조회 중...")
    cached_fx_hourly.clear()
    fx_series = cached_fx_hourly()

    targets = normalize_tickers(list(trader.positions.keys()) + scan_universe)

    def _on_prog(done: int, total: int, label: str) -> None:
        prog.progress(min(1.0, done / max(1, total)), text=f"1시간봉 API 데이터 조회 중 ({done}/{total}) - {label}")

    raw_batch, failed_map = fetch_live_1h_batch(targets, period="90d", chunk_size=25, progress_callback=_on_prog)
    prog.progress(0.88, text="챔피언 DNA 4단계 스크리너 분석 및 장외 실시간 가격(prepost) 반영 중...")

    # First-pass evaluation to find BUY_READY & held tickers so extended-hours 1m overlay runs fast on key targets
    usd_map: Dict[str, pd.DataFrame] = {}
    krw_map: Dict[str, pd.DataFrame] = {}
    eval_map: Dict[str, DnaEvaluation] = {}

    for sym, df_u in raw_batch.items():
        # Apply live extended-hours 1m execution price to held positions immediately
        if sym in trader.positions:
            df_u = apply_extended_hours_price(sym, df_u)
        df_k = convert_usd_to_krw_1h(df_u, fx_series)
        ev = evaluate_champion_dna_1h(df_u, df_k, sym)
        usd_map[sym] = df_u
        krw_map[sym] = df_k
        eval_map[sym] = ev

    # Also apply extended-hours 1m price overlay on BUY_READY candidates before final order execution
    for sym, ev in list(eval_map.items()):
        if ev.is_buy_signal and sym not in trader.positions:
            df_u_ext = apply_extended_hours_price(sym, usd_map[sym])
            df_k_ext = convert_usd_to_krw_1h(df_u_ext, fx_series)
            usd_map[sym] = df_u_ext
            krw_map[sym] = df_k_ext
            eval_map[sym] = evaluate_champion_dna_1h(df_u_ext, df_k_ext, sym)

    prog.progress(0.96, text="유망 매매 방법 3호 매수/매도 체결 엔진 실행 중...")
    scan_res = trader.run_api_scan_and_trade(eval_map, auto_trade=True)
    scan_res["scanned_count"] = len(eval_map)
    scan_res["failed_count"] = len(failed_map)

    st.session_state.scan_usd_cache = usd_map
    st.session_state.scan_krw_cache = krw_map
    st.session_state.scan_evals = eval_map
    st.session_state.last_scan_summary = scan_res

    prog.progress(1.0, text="검색 및 자동 매매 완료!")

    notices = []
    for s_msg in scan_res["sell_executions"]:
        notices.append(("warning", s_msg))
    for b_msg in scan_res["buy_executions"]:
        notices.append(("success", b_msg))
    if not scan_res["sell_executions"] and not scan_res["buy_executions"]:
        notices.append((
            "info",
            f"🔍 {len(eval_map)}개 종목 API 검색 완료 (매수 적격 BUY_READY: {scan_res['buy_ready_count']}종목, "
            f"예산초과 스킵: {len(scan_res['skipped_over_budget'])}종목). 신규 체결은 없습니다.",
        ))
    st.session_state.action_notices = notices
    st.rerun()

last_sum = st.session_state.get("last_scan_summary")
if last_sum:
    act_col2.info(
        f"🔎 최근 API 검색: **{last_sum['scanned_count']}종목** 검사 | "
        f"매수체결 **{len(last_sum['buy_executions'])}건** | "
        f"매도체결 **{len(last_sum['sell_executions'])}건** | "
        f"예산초과 스킵 **{len(last_sum['skipped_over_budget'])}건**"
    )
else:
    act_col2.info("👆 왼쪽 버튼을 누르면 150+ 종목의 1시간봉 API를 검사해 조건 충족 시 자동 매수·기록합니다.")


# ---------------------------------------------------------
# Ensure Live Data for Active Positions (Lightweight Refresh)
# ---------------------------------------------------------
fx_series = cached_fx_hourly()
latest_fx = float(fx_series.iloc[-1]) if not fx_series.empty else 1400.0

held_krw_frames: Dict[str, pd.DataFrame] = {}
total_eval_krw = 0.0

for sym, pos in trader.positions.items():
    df_k = st.session_state.scan_krw_cache.get(sym)
    if df_k is None or df_k.empty:
        df_u = cached_single_stock_1h(sym, apply_ext=True)
        if df_u is not None and not df_u.empty:
            df_k = convert_usd_to_krw_1h(df_u, fx_series)
            st.session_state.scan_usd_cache[sym] = df_u
            st.session_state.scan_krw_cache[sym] = df_k
            st.session_state.scan_evals[sym] = evaluate_champion_dna_1h(df_u, df_k, sym)
    if df_k is not None and not df_k.empty:
        held_krw_frames[sym] = df_k
        curr_px_krw = float(df_k["Close"].iloc[-1])
        total_eval_krw += int(pos["qty"]) * curr_px_krw
    else:
        total_eval_krw += int(pos["qty"]) * float(pos.get("avg_price", 0.0))


# ---------------------------------------------------------
# 1. Account Overview UI (Matching stock-)
# ---------------------------------------------------------
total_assets = trader.cash + total_eval_krw
total_profit = total_assets - trader.initial_balance
total_roi = (total_profit / trader.initial_balance) * 100.0 if trader.initial_balance > 0 else 0.0

st.subheader("💼 계좌 현황")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("계좌 잔고 (예수금)", f"{int(round(trader.cash)):,} 원")
c2.metric("주식 평가 금액", f"{int(round(total_eval_krw)):,} 원", f"{len(trader.positions)} / {trader.max_positions} 종목")
c3.metric("총 평가 자산", f"{int(round(total_assets)):,} 원")
c4.metric("총 수익금", f"{int(round(total_profit)):+,} 원", delta=f"{int(round(total_profit)):+,} 원")
c5.metric("총 수익률", f"{total_roi:+.2f} %", delta=f"환율 ₩{latest_fx:,.1f}/$")

st.markdown("---")


# ---------------------------------------------------------
# 2. Representative Market Indices (Matching stock-)
# ---------------------------------------------------------
st.subheader("🌐 대표 시장 지수")
indices = cached_market_indices()
idx_cols = st.columns(len(indices))
for idx_i, (name, val) in enumerate(indices.items()):
    idx_cols[idx_i].metric(label=name, value=f"{val['price']:,.2f}", delta=f"{val['change']:+.2f}%")

st.markdown("---")


# ---------------------------------------------------------
# 3. 1-Hour KRW Chart Analysis & Manual Trade Controls
# ---------------------------------------------------------
st.subheader("📈 1시간 봉 차트 분석 (원화 환산 · 최근 90일 데이터)")

chart_pool = normalize_tickers(
    list(trader.positions.keys())
    + [s for s, e in st.session_state.scan_evals.items() if e.is_buy_signal]
    + [s for s, e in st.session_state.scan_evals.items() if e.status == "WATCHLIST"]
    + list(st.session_state.scan_krw_cache.keys())
)
if not chart_pool:
    chart_pool = ["NVDA", "PLTR", "AMD", "AAPL", "RKLB", "OKLO"]

sel_c1, sel_c2, sel_c3 = st.columns([2.0, 1.5, 1.2])
selected_stock = sel_c1.selectbox("차트 종목 선택 (보유주 · DNA 검색 후보)", chart_pool)
custom_sym_input = sel_c2.text_input("또는 직접 티커 1시간봉 API 조회", "", placeholder="예: TSLA").upper().strip()
if sel_c3.button("🔎 선택한 티커 실시간 조회", width="stretch"):
    target_sym = custom_sym_input or selected_stock
    df_u_one = cached_single_stock_1h(target_sym, apply_ext=True)
    if df_u_one is not None and not df_u_one.empty:
        df_k_one = convert_usd_to_krw_1h(df_u_one, fx_series)
        st.session_state.scan_usd_cache[target_sym] = df_u_one
        st.session_state.scan_krw_cache[target_sym] = df_k_one
        st.session_state.scan_evals[target_sym] = evaluate_champion_dna_1h(df_u_one, df_k_one, target_sym)
        st.session_state["chart_override_sym"] = target_sym
        st.rerun()
    else:
        st.error(f"'{target_sym}'의 1시간봉 데이터를 조회하지 못했습니다.")

active_sym = st.session_state.pop("chart_override_sym", None) or selected_stock
df_chart_k = st.session_state.scan_krw_cache.get(active_sym)
df_chart_u = st.session_state.scan_usd_cache.get(active_sym)

if df_chart_k is None or df_chart_k.empty:
    with st.spinner(f"{active_sym} 실시간 1시간봉 불러오는 중..."):
        df_chart_u = cached_single_stock_1h(active_sym, apply_ext=True)
        if df_chart_u is not None and not df_chart_u.empty:
            df_chart_k = convert_usd_to_krw_1h(df_chart_u, fx_series)
            st.session_state.scan_usd_cache[active_sym] = df_chart_u
            st.session_state.scan_krw_cache[active_sym] = df_chart_k
            st.session_state.scan_evals[active_sym] = evaluate_champion_dna_1h(df_chart_u, df_chart_k, active_sym)

if df_chart_k is not None and len(df_chart_k) >= 2:
    ev_active: DnaEvaluation = st.session_state.scan_evals.get(
        active_sym,
        evaluate_champion_dna_1h(df_chart_u, df_chart_k, active_sym),
    )
    curr_krw = float(df_chart_k["Close"].iloc[-1])
    prev_krw = float(df_chart_k["Close"].iloc[-2])
    pct_chg = ((curr_krw - prev_krw) / prev_krw) * 100.0 if prev_krw > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{active_sym} 현재가 (원화)", f"{int(round(curr_krw)):,} 원", delta=f"{pct_chg:+.2f}% (${ev_active.price_usd:,.2f})")
    m2.metric("DNA 종합 점수", f"{ev_active.alpha_score:.0f} / 100", ev_active.status)
    m3.metric("10일/30일 거시선", "✅ 정배열" if ev_active.golden_aligned else "❌ 역배열", f"{int(ev_active.ema_10d_krw):,} / {int(ev_active.ema_30d_krw):,}")
    m4.metric("시간당 ATR (≥1.1%)", f"{ev_active.atr_pct:.2f}%", "변동성 통과" if ev_active.atr_pct >= 1.1 else "변동성 부족")
    m5.metric("60봉 고점 대비", f"-{ev_active.dist_from_high_pct:.1f}%", "돌파권(-6%)" if ev_active.dist_from_high_pct <= 6.0 else "이격 대기")

    plot_df = df_chart_k.tail(220).copy()
    max_idx = plot_df["High"].idxmax()
    max_val = float(plot_df["High"].max())
    min_idx = plot_df["Low"].idxmin()
    min_val = float(plot_df["Low"].min())
    init_start = plot_df.index[max(0, len(plot_df) - 90)]
    init_end = plot_df.index[-1]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.82, 0.18],
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="1h 캔들(원)",
            increasing_line_color="#e15241",
            increasing_fillcolor="#e15241",
            decreasing_line_color="#267af3",
            decreasing_fillcolor="#267af3",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA_10D"], line=dict(color="#34c759", width=1.6), name="10일 거시선(65h)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA_30D"], line=dict(color="#ff9500", width=1.6), name="30일 거시선(195h)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA_200H"], line=dict(color="#af52de", width=1.5, dash="dot"), name="200시간선(200h)"), row=1, col=1)

    pos_active = trader.positions.get(active_sym)
    if pos_active:
        fig.add_hline(y=float(pos_active["avg_price"]), line_color="#0F172A", line_dash="dash", annotation_text=f"매수평단 {int(pos_active['avg_price']):,}원", row=1, col=1)
        fig.add_hline(y=float(pos_active["stop_loss_price"]), line_color="#DC2626", line_dash="dash", annotation_text=f"12% 손절선 {int(pos_active['stop_loss_price']):,}원", row=1, col=1)

    fig.add_annotation(x=max_idx, y=max_val, text=f"최고 {int(max_val):,}원", showarrow=True, arrowhead=2, arrowcolor="#e15241", ax=0, ay=-22, row=1, col=1)
    fig.add_annotation(x=min_idx, y=min_val, text=f"최저 {int(min_val):,}원", showarrow=True, arrowhead=2, arrowcolor="#267af3", ax=0, ay=22, row=1, col=1)

    vol_colors = ["#e15241" if c >= o else "#267af3" for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="거래량", marker_color=vol_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Vol_Avg_90"], line=dict(color="#ff3b30", width=1), name="90봉 평균 거래량"), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=680,
        dragmode="pan",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000"),
        legend=dict(font=dict(color="#000000"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    if active_sym.endswith("-USD"):
        fig.update_xaxes(range=[init_start, init_end], showgrid=True, gridcolor="#f2f2f7", tickfont=dict(color="#000000"))
    else:
        fig.update_xaxes(
            range=[init_start, init_end],
            showgrid=True,
            gridcolor="#f2f2f7",
            tickfont=dict(color="#000000"),
            rangebreaks=[dict(bounds=["sat", "mon"])],
        )
    fig.update_yaxes(showgrid=True, gridcolor="#f2f2f7", fixedrange=False, tickfont=dict(color="#000000"))

    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    st.caption(f"🔎 진단 상세: **{ev_active.rationale}** | 기준 1시간봉: `{ev_active.bar_ts} UTC`")

    # Quick Manual Buy / Sell Controls for Selected Chart Stock
    man_c1, man_c2, man_c3 = st.columns([1.8, 2.2, 1.2])
    if pos_active:
        sell_memo = man_c2.text_input("수동 매도 사유", "차트 확인 후 수동 익절/청산", key=f"sell_reason_{active_sym}")
        man_c1.metric("현재 보유 중", f"{int(pos_active['qty'])}주", f"평단 {int(pos_active['avg_price']):,}원")
        if man_c3.button(f"🛑 {active_sym} 전량 수동 매도", width="stretch"):
            ok_s, msg_s = trader.sell_all_shares(active_sym, curr_krw, sell_memo.strip() or "수동 매도", ev_active.bar_ts)
            if ok_s:
                st.session_state.action_notices.append(("success", msg_s))
                st.rerun()
            else:
                st.error(msg_s)
    else:
        slots_left = max(1, trader.max_positions - len(trader.positions))
        def_alloc = int(min(trader.cash, max(100_000, trader.cash // slots_left))) if trader.cash >= 10_000 else 0
        alloc_input = man_c1.number_input(
            "수동 매수 배분금 (원)",
            min_value=0,
            max_value=max(0, int(trader.cash)),
            value=min(max(0, int(trader.cash)), def_alloc),
            step=50_000,
            format="%d",
        )
        est_qty = int(alloc_input // (curr_krw * (1.0 + trader.fee_rate))) if curr_krw > 0 else 0
        man_c2.text_input("예상 체결 수량 (소수점 불가)", value=f"{est_qty}주 (1주 약 {int(curr_krw):,}원)", disabled=True)
        if man_c3.button(f"🚀 {active_sym} 정수 매수", width="stretch"):
            strat_txt = f"수동매수 (DNA {ev_active.alpha_score:.0f} / ATR {ev_active.atr_pct:.1f}%)"
            ok_b, msg_b = trader.buy_integer_shares(
                symbol=active_sym,
                price_krw=curr_krw,
                allocation_krw=float(alloc_input),
                strategy_name=strat_txt,
                reason=f"[차트 수동 매수] {ev_active.rationale}",
                bar_ts=ev_active.bar_ts,
                indicators={"alpha_score": ev_active.alpha_score, "atr_pct": ev_active.atr_pct},
            )
            if ok_b:
                st.session_state.action_notices.append(("success", msg_b))
                st.rerun()
            else:
                st.error(msg_b)
else:
    st.warning("선택한 종목의 1시간봉 데이터를 불러올 수 없습니다.")

st.markdown("---")


# ---------------------------------------------------------
# 4. Trading Journal, Screener Results & Performance Analytics
# ---------------------------------------------------------
st.subheader("📑 주식 매매일지 및 전략/검색 성과 분석")

tab1, tab2, tab3, tab4 = st.tabs([
    "💼 현재 보유 종목 (실시간 수익률)",
    "🔍 API 종목 검색 결과 (DNA 스크리너)",
    "📜 매매 기록 전체보기 (매매일지)",
    "📊 승률 · 수익률 · 전략 통계",
])

with tab1:
    if trader.positions:
        pos_rows = []
        for sym, pos_info in trader.positions.items():
            qty = int(pos_info.get("qty", 0))
            if qty <= 0:
                continue
            avg_px = float(pos_info["avg_price"])
            df_k = held_krw_frames.get(sym)
            curr_px = float(df_k["Close"].iloc[-1]) if df_k is not None and not df_k.empty else avg_px

            invested_total = float(pos_info.get("cost_basis_krw", avg_px * qty)) + float(pos_info.get("entry_fee", 0.0))
            est_exit_net = (curr_px * qty) * (1.0 - trader.fee_rate)
            profit_krw = est_exit_net - invested_total
            roi = (profit_krw / invested_total) * 100.0 if invested_total > 0 else 0.0

            pos_rows.append({
                "종목": sym,
                "보유 수량": f"{qty:,} 주",
                "매수 평단가": f"{int(round(avg_px)):,} 원",
                "현재가": f"{int(round(curr_px)):,} 원",
                "평가금액": f"{int(round(curr_px * qty)):,} 원",
                "예상 손익(수수료반영)": f"{int(round(profit_krw)):+,} 원",
                "현재 수익률(%)": round(roi, 2),
                "12% 손절가": f"{int(round(pos_info['stop_loss_price'])):,} 원",
                "진입 1h봉 시각": pos_info.get("entry_bar_ts", "-"),
                "진입 전략 / 매수 이유": pos_info.get("entry_reason", pos_info.get("entry_strategy", "-")),
            })
        st.dataframe(pd.DataFrame(pos_rows), width="stretch")
    else:
        st.info("현재 보유 중인 종목이 없습니다. 상단의 **'주식 API 1시간봉 종목 검색 & 자동 매수/매도 실행'** 버튼을 눌러보세요.")

with tab2:
    if last_sum and not last_sum["scan_df"].empty:
        scan_df = last_sum["scan_df"]
        if last_sum.get("skipped_over_budget"):
            skipped_desc = ", ".join(f"{x['ticker']}(1주 {int(x['unit_krw']):,}원)" for x in last_sum["skipped_over_budget"][:8])
            st.warning(f"💸 슬롯 배분 예산 초과로 자동 건너뛴 고가주 ({len(last_sum['skipped_over_budget'])}건): {skipped_desc}")

        status_filter = st.selectbox("상태별 필터", ["전체"] + scan_df["상태"].unique().tolist())
        shown_df = scan_df if status_filter == "전체" else scan_df[scan_df["상태"] == status_filter]
        st.dataframe(shown_df, width="stretch", height=400)
    else:
        st.info("아직 API 종목 검색을 실행하지 않았습니다. 상단 실행 버튼을 누르면 150+ 종목 DNA 스크리너 결과가 여기에 표시됩니다.")

with tab3:
    if trader.trade_logs:
        logs_df = pd.DataFrame(trader.trade_logs).iloc[::-1]
        st.dataframe(logs_df, width="stretch")

        csv_bytes = logs_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 매매일지 전체 CSV 다운로드",
            data=csv_bytes,
            file_name="paper_trading_journal.csv",
            mime="text/csv",
        )
    else:
        st.info("현재까지 실행된 매매 내역이 없습니다.")

with tab4:
    perf = trader.get_performance_summary()
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("청산 완료 매매", f"{perf['total_closed']} 건")
    p2.metric("실전 승률", f"{perf['win_rate']:.1f} %", f"{perf['wins']}승 {perf['losses']}패")
    p3.metric("평균 수익률", f"{perf['avg_roi']:+.2f} %")
    p4.metric("손익비 (PF)", f"{perf['profit_factor']:.2f}")
    p5.metric("누적 거래 수수료", f"{perf['total_fees_krw']:,} 원", f"순손익 {perf['realized_profit_krw']:+,}원")

    if trader.trade_logs:
        logs_df = pd.DataFrame(trader.trade_logs)
        sell_logs = logs_df[logs_df["타입"] == "매도"]
        if not sell_logs.empty:
            st.markdown("#### 📊 진입 전략별 승률 & 수익률 통계")
            grp = (
                sell_logs.groupby("진입 전략")
                .agg(
                    총매매횟수=("수익률(%)", "count"),
                    익절횟수=("실현손익(원)", lambda x: int((x > 0).sum())),
                    손절횟수=("실현손익(원)", lambda x: int((x <= 0).sum())),
                    평균수익률=("수익률(%)", "mean"),
                    누적실현손익원=("실현손익(원)", "sum"),
                )
                .reset_index()
            )
            grp["승률(%)"] = ((grp["익절횟수"] / grp["총매매횟수"]) * 100.0).round(2)
            grp["평균수익률"] = grp["평균수익률"].round(2)
            grp = grp[["진입 전략", "총매매횟수", "승률(%)", "평균수익률", "익절횟수", "손절횟수", "누적실현손익원"]]
            st.dataframe(grp, width="stretch")
        else:
            st.info("아직 매도(청산) 완료 거래가 없어 전략별 승률 통계를 산출할 수 없습니다.")
    else:
        st.info("거래 내역이 없습니다.")
