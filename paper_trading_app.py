"""Streamlit Live Paper Trading & Trading Journal Application (1-Hour Bar Base).

Default 10,000,000 KRW account, live USD/KRW FX rate integration, 100+ stock live
yfinance 1-hour candle API batch scanning, max 20 equal-weight integer-share slots,
1-hour Candlestick & EMA chart viewer, and full Trading Journal (entry/exit reasons,
win rate, return %, profit factor, reflection memos, and optional GitHub sync).
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import (
    fetch_live_hourly_batch,
    fetch_live_usd_krw_rate,
    _normalize_tickers,
)
from src.timeframes import ema_days, ema_hours
from src.adaptive_experiment import compute_atr
from src.universes import SCAN_UNIVERSES
from src.github_sync import (
    DEFAULT_GITHUB_REPO,
    DEFAULT_PORTFOLIO_REPO_PATH,
    load_portfolio_from_github,
    push_portfolio_to_github,
)
from src.paper_trader import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CAPITAL_KRW,
    DEFAULT_MAX_POSITIONS,
    PORTFOLIO_FILE,
    compute_journal_metrics,
    compute_portfolio_summary_krw,
    ensure_krw_portfolio_schema,
    evaluate_live_champion_signals,
    execute_krw_buy,
    execute_krw_sell,
    load_krw_portfolio,
    reset_krw_portfolio,
    run_live_scan_and_trade,
    save_portfolio,
    update_krw_market_prices,
    update_trade_memo,
)

# Page Setup
st.set_page_config(
    page_title="실시간 1시간봉 모의투자 & 매매일지",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.main-title {
    font-size: 2.05rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 0.98rem;
    color: #475569;
    margin-bottom: 1.0rem;
}
.fx-pill {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Helper Functions ---------------- #

def _get_github_config() -> Dict[str, str]:
    token = ""
    repo = DEFAULT_GITHUB_REPO
    try:
        token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
        repo = str(st.secrets.get("GITHUB_REPO", DEFAULT_GITHUB_REPO)).strip() or DEFAULT_GITHUB_REPO
    except Exception:
        pass
    if not token:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not repo:
        repo = os.environ.get("GITHUB_REPO", DEFAULT_GITHUB_REPO).strip() or DEFAULT_GITHUB_REPO
    return {"token": token, "repo": repo}


def _persist_all(portfolio: Dict[str, Any], gh_cfg: Dict[str, str], commit_msg: str) -> Optional[str]:
    save_portfolio(portfolio, filepath=PORTFOLIO_FILE)
    if gh_cfg["token"]:
        ok, msg = push_portfolio_to_github(
            portfolio,
            token=gh_cfg["token"],
            repo=gh_cfg["repo"],
            file_path=DEFAULT_PORTFOLIO_REPO_PATH,
            commit_message=commit_msg,
        )
        return msg if ok else f"⚠️ {msg}"
    return None


def _build_hourly_chart(df: pd.DataFrame, ticker: str, position: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Render 2-row Plotly 1-hour candlestick + EMA + ATR chart."""
    plot_df = df.tail(220).copy()
    close = df["Close"]
    ema10d = ema_days(close, 10).loc[plot_df.index]
    ema30d = ema_days(close, 30).loc[plot_df.index]
    ema200h = ema_hours(close, 200).loc[plot_df.index]
    atr_pct = (compute_atr(df, period=14) / close * 100.0).loc[plot_df.index]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.74, 0.26],
        subplot_titles=(f"{ticker} 1시간 봉 차트 & 거시 이평선", "시간당 변동성 (ATR 14 %)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="1h 캔들",
            increasing_line_color="#EF4444",
            decreasing_line_color="#3B82F6",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=ema10d, name="10일 거시선 (65h)", line=dict(color="#F59E0B", width=1.8)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=ema30d, name="30일 거시선 (195h)", line=dict(color="#10B981", width=1.8)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=ema200h, name="200시간선 (200h)", line=dict(color="#6366F1", width=1.6, dash="dot")),
        row=1,
        col=1,
    )

    if position:
        entry_px = float(position["entry_price_usd"])
        stop_px = float(position["stop_loss_price_usd"])
        fig.add_hline(
            y=entry_px,
            line_dash="dash",
            line_color="#0F172A",
            annotation_text=f"매수가 ${entry_px:,.2f}",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=stop_px,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"12% 손절가 ${stop_px:,.2f}",
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=atr_pct, name="ATR (%)", line=dict(color="#8B5CF6", width=1.6)),
        row=2,
        col=1,
    )
    fig.add_hline(y=1.1, line_dash="dot", line_color="#EF4444", annotation_text="최소기준 1.1%", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=560,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_signal_checklist(sig: Dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    golden_ok = sig["ema_10d"] > sig["ema_30d"]
    above200_ok = sig["current_price"] > sig["ema_200h"]
    atr_ok = sig["atr_pct"] >= 1.1
    high_ok = sig["dist_from_high_pct"] <= 6.0

    c1.metric("10일/30일 거시 정배열", "✅ 통과" if golden_ok else "❌ 역배열", f"${sig['ema_10d']:.1f} / ${sig['ema_30d']:.1f}")
    c2.metric("200시간선 상단 지지", "✅ 통과" if above200_ok else "❌ 하회", f"현재 ${sig['current_price']:.2f} vs ${sig['ema_200h']:.1f}")
    c3.metric("시간당 ATR 변동성 (≥1.1%)", "✅ 통과" if atr_ok else "❌ 부족", f"{sig['atr_pct']:.2f}%")
    c4.metric("60봉 신고가권 (-6% 이내)", "✅ 통과" if high_ok else "❌ 이격", f"-{sig['dist_from_high_pct']:.1f}%")


# ---------------- Session State & Initialization ---------------- #

gh_cfg = _get_github_config()

if "live_fx_rate" not in st.session_state:
    rate, fx_ts = fetch_live_usd_krw_rate(fallback=1400.0)
    st.session_state["live_fx_rate"] = rate
    st.session_state["live_fx_ts"] = fx_ts

if "github_bootstrapped" not in st.session_state:
    st.session_state["github_bootstrapped"] = True
    if gh_cfg["token"]:
        gh_port, _ = load_portfolio_from_github(
            token=gh_cfg["token"],
            repo=gh_cfg["repo"],
            file_path=DEFAULT_PORTFOLIO_REPO_PATH,
        )
        local_port = load_krw_portfolio(PORTFOLIO_FILE, fx_rate=st.session_state["live_fx_rate"])
        if gh_port and gh_port.get("updated_at", "") > local_port.get("updated_at", ""):
            gh_port = ensure_krw_portfolio_schema(gh_port, fx_rate=st.session_state["live_fx_rate"])
            save_portfolio(gh_port, filepath=PORTFOLIO_FILE)

if "live_candles" not in st.session_state:
    st.session_state["live_candles"] = {}
if "last_scan_result" not in st.session_state:
    st.session_state["last_scan_result"] = None
if "last_action_messages" not in st.session_state:
    st.session_state["last_action_messages"] = []
if "last_restored_upload_sig" not in st.session_state:
    st.session_state["last_restored_upload_sig"] = None


# ---------------- Sidebar Controls ---------------- #

st.sidebar.title("⚙️ 실시간 모의투자 설정")

initial_cap_krw = st.sidebar.number_input(
    "기본 자본금 (원)",
    min_value=1_000_000,
    max_value=1_000_000_000,
    value=int(DEFAULT_INITIAL_CAPITAL_KRW),
    step=1_000_000,
    format="%d",
)
fee_rate_pct = st.sidebar.number_input(
    "거래 수수료율 (편도 %)",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
)
fee_rate = fee_rate_pct / 100.0

portfolio = load_krw_portfolio(
    filepath=PORTFOLIO_FILE,
    initial_capital_krw=float(initial_cap_krw),
    fee_rate=fee_rate,
    max_positions=DEFAULT_MAX_POSITIONS,
    fx_rate=st.session_state["live_fx_rate"],
)
portfolio["fee_rate"] = fee_rate

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 포트폴리오 운용 규칙")
max_positions = st.sidebar.slider(
    "최대 동시 보유 종목 수 (균등 분할)",
    min_value=1,
    max_value=30,
    value=int(portfolio.get("max_positions", DEFAULT_MAX_POSITIONS)),
)
st.sidebar.caption(
    f"💡 1,000만원 ÷ {max_positions}종목 = 종목당 약 **₩{initial_cap_krw / max_positions:,.0f}** 배분 (소수점 불가 · 1주 이상 정수 매수)"
)
auto_trade_mode = st.sidebar.checkbox(
    "🤖 버튼 클릭 시 자동 매수·매도 실행",
    value=True,
    help="API 스캔 버튼 클릭 시 1시간봉 조건 부합 종목을 즉시 자동 매수하고, 손절(-12%)/데드크로스 보유 종목을 자동 청산합니다.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 API 스캔 종목 풀 (1시간봉)")
univ_choice = st.sidebar.selectbox("기본 종목군 선택", list(SCAN_UNIVERSES.keys()))
extra_input = st.sidebar.text_area(
    "추가 검사 티커 (선택, 쉼표·공백 구분)",
    "",
    placeholder="예: TSLA, IONQ, PLTR",
)
extra_tickers = _normalize_tickers(extra_input.replace(",", " ").split()) if extra_input.strip() else []
active_universe = _normalize_tickers(SCAN_UNIVERSES[univ_choice] + extra_tickers)
st.sidebar.caption(f"총 API 검사 대상: **{len(active_universe)}개 종목**")

st.sidebar.markdown("---")
with st.sidebar.expander("💾 매매일지 백업 · 복원 & GitHub 동기화", expanded=False):
    if gh_cfg["token"]:
        st.success(f"GitHub 자동 동기화 연결됨 ({gh_cfg['repo']})")
        if st.button("☁️ 지금 GitHub에 수동 동기화"):
            ok, g_msg = push_portfolio_to_github(portfolio, token=gh_cfg["token"], repo=gh_cfg["repo"])
            st.info(g_msg)
    else:
        st.info("현재 로컬 JSON 저장 모드입니다. Streamlit Cloud 배포 시 `GITHUB_TOKEN` Secret을 넣으면 매매 시마다 GitHub에 자동 저장됩니다.")

    backup_bytes = json.dumps(portfolio, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        label="📥 계좌 · 매매일지 백업 (JSON)",
        data=backup_bytes,
        file_name="paper_portfolio_backup.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("📤 백업 JSON 업로드하여 복원", type=["json"])
    if uploaded is not None:
        upload_sig = f"{uploaded.name}_{uploaded.size}"
        if upload_sig != st.session_state["last_restored_upload_sig"]:
            try:
                restored_raw = json.loads(uploaded.read().decode("utf-8"))
                restored = ensure_krw_portfolio_schema(
                    restored_raw,
                    initial_capital_krw=float(initial_cap_krw),
                    fee_rate=fee_rate,
                    max_positions=max_positions,
                    fx_rate=st.session_state["live_fx_rate"],
                )
                save_portfolio(restored, filepath=PORTFOLIO_FILE)
                st.session_state["last_restored_upload_sig"] = upload_sig
                st.success("백업 파일에서 계좌와 매매일지를 복원했습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"복원 실패: {e}")

with st.sidebar.expander("🗑️ 계좌 초기화 (Reset)", expanded=False):
    st.warning("모든 보유 종목과 매매일지가 삭제되고 기본 자본으로 초기화됩니다.")
    if st.button("🚨 ₩10,000,000 초기 상태로 리셋"):
        portfolio = reset_krw_portfolio(
            initial_capital_krw=float(initial_cap_krw),
            fee_rate=fee_rate,
            max_positions=max_positions,
            fx_rate=st.session_state["live_fx_rate"],
            filepath=PORTFOLIO_FILE,
        )
        if gh_cfg["token"]:
            push_portfolio_to_github(portfolio, token=gh_cfg["token"], repo=gh_cfg["repo"], commit_message="chore: reset paper portfolio")
        st.session_state["last_scan_result"] = None
        st.session_state["last_action_messages"] = [("info", "계좌가 ₩10,000,000 초기 상태로 리셋되었습니다.")]
        st.rerun()


# ---------------- Header & Top KPIs ---------------- #

fx_rate = float(st.session_state["live_fx_rate"])
summary = compute_portfolio_summary_krw(portfolio, fx_rate=fx_rate)

st.markdown("<div class='main-title'>📈 실시간 미국 주식 1시간봉 모의투자 & 매매일지</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>버튼 클릭 시 <b>Yahoo Finance 실시간 1시간봉 API</b>로 100+ 종목을 검사하고, "
    "챔피언 DNA 조건 충족 종목을 <b>1,000만원 기준 최대 20종목 균등 분할(정수 1주 이상)</b>로 자동 매수·기록합니다.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<span class='fx-pill'>💱 실시간 환율: 1 USD = ₩{fx_rate:,.2f}</span>"
    f"<span class='fx-pill'>🕒 마지막 계좌 갱신: {portfolio.get('updated_at', '')[:19].replace('T', ' ')} UTC</span>",
    unsafe_allow_html=True,
)
st.write("")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("💰 총 평가자산", f"₩{summary['total_equity_krw']:,.0f}", f"{summary['total_return_pct']:+.2f}% (${summary['total_equity_usd']:,.0f})")
k2.metric("💵 보유 예수금", f"₩{summary['cash_krw']:,.0f}")
k3.metric("📈 주식 평가액", f"₩{summary['invested_krw']:,.0f}")
k4.metric("📊 미실현 손익", f"₩{summary['unrealized_pnl_krw']:+,.0f}")
k5.metric("🎯 누적 실현 손익", f"₩{summary['realized_pnl_krw']:+,.0f}")
k6.metric("🏆 보유 포지션", f"{summary['position_count']} / {max_positions}종목")

# Display persisted action messages after rerun
if st.session_state["last_action_messages"]:
    for kind, txt in st.session_state["last_action_messages"]:
        if kind == "success":
            st.success(txt)
        elif kind == "warning":
            st.warning(txt)
        elif kind == "error":
            st.error(txt)
        else:
            st.info(txt)
    st.session_state["last_action_messages"] = []

st.markdown("---")


# ---------------- Primary Action Bar ---------------- #

act1, act2, act3 = st.columns([2.2, 1.5, 2.3])

if act1.button("🚀 실시간 1시간봉 API 검사 & 자동 매수/매도 실행", type="primary", use_container_width=True):
    prog_bar = st.progress(0.0, text="실시간 원/달러 환율 조회 중...")
    new_fx, new_fx_ts = fetch_live_usd_krw_rate(fallback=fx_rate)
    st.session_state["live_fx_rate"] = new_fx
    st.session_state["live_fx_ts"] = new_fx_ts

    # Ensure held positions are also included in the API scan
    scan_targets = _normalize_tickers(list(portfolio.get("positions", {}).keys()) + active_universe)

    def _on_progress(done: int, total: int, label: str) -> None:
        ratio = min(1.0, done / max(1, total))
        prog_bar.progress(ratio, text=f"실시간 1시간봉 API 수신 중 ({done}/{total}) - {label}")

    live_data, failed_map = fetch_live_hourly_batch(
        scan_targets,
        period="90d",
        interval="1h",
        chunk_size=25,
        progress_callback=_on_progress,
    )
    prog_bar.progress(1.0, text=f"API 데이터 수집 완료 ({len(live_data)}개 성공 / {len(failed_map)}개 실패). 매매 조건 검사 중...")

    st.session_state["live_candles"] = live_data
    scan_res = run_live_scan_and_trade(
        portfolio,
        universe_data=live_data,
        fx_rate=new_fx,
        max_positions=max_positions,
        auto_trade=auto_trade_mode,
    )
    scan_res["failed_map"] = failed_map
    scan_res["scanned_total"] = len(live_data)
    st.session_state["last_scan_result"] = scan_res

    msgs = []
    for s_msg in scan_res["sell_executions"]:
        msgs.append(("warning", s_msg))
    for b_msg in scan_res["buy_executions"]:
        msgs.append(("success", b_msg))
    for alert in scan_res["exit_alerts"]:
        msgs.append(("warning", f"⚠️ 청산 신호 감지 [{alert['ticker']}]: {alert['reason']}"))
    if not scan_res["sell_executions"] and not scan_res["buy_executions"]:
        buy_sig_cnt = len(scan_res["pending_buys"])
        skip_cnt = len(scan_res["skipped_over_budget"])
        msgs.append((
            "info",
            f"🔍 {len(live_data)}개 종목 1시간봉 API 검사 완료 (신규 조건 충족 {buy_sig_cnt}건, "
            f"슬롯 예산초과 스킵 {skip_cnt}건). 체결 내역은 없습니다.",
        ))

    if gh_cfg["token"] and (scan_res["sell_executions"] or scan_res["buy_executions"]):
        ok_gh, gh_msg = push_portfolio_to_github(
            portfolio,
            token=gh_cfg["token"],
            repo=gh_cfg["repo"],
            commit_message=f"trade: live scan buy={len(scan_res['buy_executions'])} sell={len(scan_res['sell_executions'])}",
        )
        msgs.append(("info" if ok_gh else "warning", f"☁️ {gh_msg}"))

    st.session_state["last_action_messages"] = msgs
    st.rerun()

if act2.button("⚡ 보유 종목 시세 · 환율만 최신화", use_container_width=True):
    new_fx, new_fx_ts = fetch_live_usd_krw_rate(fallback=fx_rate)
    st.session_state["live_fx_rate"] = new_fx
    st.session_state["live_fx_ts"] = new_fx_ts

    held_syms = list(portfolio.get("positions", {}).keys())
    if held_syms:
        with st.spinner(f"보유 {len(held_syms)}개 종목 최신 1시간봉 API 조회 중..."):
            held_data, _ = fetch_live_hourly_batch(held_syms, period="90d", interval="1h", chunk_size=25)
            st.session_state["live_candles"].update(held_data)
            curr_px = {s: float(df["Close"].iloc[-1]) for s, df in held_data.items() if not df.empty}
            update_krw_market_prices(portfolio, curr_px, fx_rate=new_fx, record_snapshot=True)
            gh_status = _persist_all(portfolio, gh_cfg, "chore: refresh live prices & FX")
            msgs = [("success", f"보유 {len(curr_px)}개 종목 시세 및 환율(₩{new_fx:,.2f})을 업데이트했습니다.")]
            if gh_status:
                msgs.append(("info", gh_status))
            st.session_state["last_action_messages"] = msgs
            st.rerun()
    else:
        st.info(f"보유 종목이 없어 환율(₩{new_fx:,.2f})만 최신화했습니다.")

last_scan = st.session_state.get("last_scan_result")
if last_scan:
    act3.caption(
        f"✅ 최근 API 스캔: **{last_scan.get('scanned_total', 0)}종목** 검사 | "
        f"매수 체결 **{len(last_scan.get('buy_executions', []))}건** | "
        f"예산초과 스킵 **{len(last_scan.get('skipped_over_budget', []))}건**"
    )
else:
    act3.caption("👆 왼쪽 **'실시간 1시간봉 API 검사 & 자동 매수/매도 실행'** 버튼을 누르면 실제 시세를 받아 매매를 시작합니다.")

st.markdown("---")


# ---------------- Main 5 Tabs ---------------- #

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 보유 현황 & 1시간 봉 차트",
    "🔍 API 스캔 결과 & 수동 매수",
    "📓 매매일지 (이유 · 승률 · 수익률)",
    "📈 계좌 자산 변화 곡선",
    "📖 전략 규칙 & 클라우드 저장 가이드",
])


# ---------------- Tab 1: Active Positions & 1h Candlestick Chart ---------------- #
with tab1:
    positions = portfolio.get("positions", {})
    st.subheader(f"📌 현재 보유 포지션 ({len(positions)} / {max_positions}종목)")

    if positions:
        pos_rows = []
        for sym, pos in positions.items():
            pos_rows.append({
                "종목": sym,
                "보유 수량": f"{int(pos['shares']):,}주",
                "매수가($)": round(float(pos["entry_price_usd"]), 2),
                "현재가($)": round(float(pos["current_price_usd"]), 2),
                "원화 평가액(₩)": int(round(pos.get("current_value_krw", 0))),
                "미실현 손익(₩)": int(round(pos.get("unrealized_pnl_krw", 0))),
                "원화 수익률(%)": round(float(pos.get("unrealized_pnl_pct", 0.0)), 2),
                "주가 변동(%)": round(float(pos.get("price_return_pct", 0.0)), 2),
                "12% 손절가($)": round(float(pos["stop_loss_price_usd"]), 2),
                "진입환율(₩)": round(float(pos.get("entry_fx_rate", fx_rate)), 1),
                "매수 이유": pos.get("entry_reason", ""),
                "진입 일시": pos.get("entry_time", "")[:16].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

        st.markdown("#### 🚨 보유 종목 수동 전량 매도")
        m_col1, m_col2, m_col3 = st.columns([1.5, 2.5, 1.2])
        sell_target = m_col1.selectbox("매도할 종목", list(positions.keys()), key="manual_sell_ticker")
        sell_reason = m_col2.text_input("매도 이유 (매매일지 기록)", "사용자 판단 수동 익절/청산", key="manual_sell_reason")
        if m_col3.button(f"🛑 {sell_target} 즉시 매도", type="secondary", use_container_width=True):
            live_candles = st.session_state.get("live_candles", {})
            px_usd = float(positions[sell_target]["current_price_usd"])
            exit_ind = {}
            if sell_target in live_candles:
                sig = evaluate_live_champion_signals(live_candles[sell_target], sell_target)
                px_usd = float(sig["current_price"])
                exit_ind = {
                    "atr_pct": sig["atr_pct"],
                    "ema_10d": sig["ema_10d"],
                    "ema_30d": sig["ema_30d"],
                    "ema_200h": sig["ema_200h"],
                }
            ok, msg = execute_krw_sell(
                portfolio,
                sell_target,
                price_usd=px_usd,
                fx_rate=fx_rate,
                reason=sell_reason.strip() or "사용자 수동 청산",
                exit_indicators=exit_ind,
            )
            if ok:
                gh_status = _persist_all(portfolio, gh_cfg, f"trade: manual sell {sell_target}")
                msgs = [("success", msg)]
                if gh_status:
                    msgs.append(("info", gh_status))
                st.session_state["last_action_messages"] = msgs
                st.rerun()
            else:
                st.error(msg)
    else:
        st.info("현재 보유 중인 종목이 없습니다. 상단의 **'실시간 1시간봉 API 검사 & 자동 매수/매도 실행'** 버튼을 눌러 투자를 시작해 보세요.")

    st.markdown("---")
    st.subheader("📊 실시간 1시간 봉 차트 & 지표 확인 (보고 투자하기)")

    live_candles: Dict[str, pd.DataFrame] = st.session_state.get("live_candles", {})
    chart_candidates = _normalize_tickers(list(positions.keys()) + list(live_candles.keys()))
    ch_col1, ch_col2, ch_col3 = st.columns([1.8, 1.5, 1.2])
    selected_chart_sym = ch_col1.selectbox(
        "차트 조회 종목 선택 (스캔된 종목 목록)",
        chart_candidates if chart_candidates else ["NVDA", "PLTR", "AMD", "AAPL"],
        key="chart_sym_select",
    )
    custom_chart_sym = ch_col2.text_input("또는 직접 티커 1시간봉 즉시 조회", "", placeholder="예: TSLA").upper().strip()
    if ch_col3.button("🔎 1시간봉 차트 API 불러오기", use_container_width=True):
        target_sym = custom_chart_sym or selected_chart_sym
        with st.spinner(f"{target_sym} 최신 1시간봉 API 수신 중..."):
            fetched, err = fetch_live_hourly_batch([target_sym], period="90d", interval="1h", chunk_size=1)
            if target_sym in fetched:
                st.session_state["live_candles"][target_sym] = fetched[target_sym]
                st.session_state["chart_active_override"] = target_sym
                st.rerun()
            else:
                st.error(err.get(target_sym, f"{target_sym} 1시간봉 조회 실패"))

    active_sym = st.session_state.pop("chart_active_override", None) or selected_chart_sym
    if active_sym in st.session_state.get("live_candles", {}):
        df_chart = st.session_state["live_candles"][active_sym]
        sig_chart = evaluate_live_champion_signals(df_chart, active_sym)
        _render_signal_checklist(sig_chart)
        fig_hourly = _build_hourly_chart(df_chart, active_sym, position=positions.get(active_sym))
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.caption(f"진단 요약: **{sig_chart['rationale']}** | 최신 1시간봉: `{sig_chart['latest_bar_time']}`")
    else:
        st.info("상단 스캔 버튼을 누르거나 오른쪽 **'1시간봉 차트 API 불러오기'** 버튼을 누르면 실시간 1시간 봉 캔들과 이동평균선 차트가 표시됩니다.")


# ---------------- Tab 2: Live Scan Results & Manual Buy ---------------- #
with tab2:
    st.subheader("🔍 실시간 API 스캔 결과 & 수동 정수 매수")
    if last_scan and not last_scan["scan_df"].empty:
        scan_df: pd.DataFrame = last_scan["scan_df"]

        if last_scan.get("skipped_over_budget"):
            skipped_names = ", ".join(
                f"{x['ticker']}(1주 ≈ ₩{x['unit_krw']:,.0f})" for x in last_scan["skipped_over_budget"][:8]
            )
            st.warning(
                f"💸 **20분할 슬롯 예산 초과로 자동 건너뛴 고가주 ({len(last_scan['skipped_over_budget'])}건)**: "
                f"{skipped_names} → 소수점 거래 불허용 원칙에 따라 다음 적격 종목으로 슬롯을 배분했습니다."
            )

        status_opts = ["전체"] + scan_df["상태"].unique().tolist()
        sel_status = st.selectbox("상태 필터", status_opts, index=0)
        filtered_df = scan_df if sel_status == "전체" else scan_df[scan_df["상태"] == sel_status]
        st.dataframe(filtered_df, use_container_width=True, height=400)

        st.markdown("#### 🛒 스캔 종목 수동 정수 매수")
        unheld = scan_df[~scan_df["종목"].isin(positions.keys())]
        if not unheld.empty:
            b1, b2, b3, b4 = st.columns([1.8, 1.4, 2.0, 1.2])
            buy_sym = b1.selectbox("매수할 종목", unheld["종목"].tolist(), key="manual_buy_sym")
            row_sel = unheld[unheld["종목"] == buy_sym].iloc[0]
            px_usd = float(row_sel["현재가($)"])
            slots_left = max(1, max_positions - len(positions))
            default_krw = min(int(summary["cash_krw"]), max(100_000, int(summary["cash_krw"] // slots_left)))
            alloc_krw = b2.number_input(
                "배분 예산 (원)",
                min_value=10_000,
                max_value=max(10_000, int(summary["cash_krw"])),
                value=min(max(10_000, int(summary["cash_krw"])), default_krw),
                step=50_000,
                format="%d",
            )
            est_shares = int(alloc_krw // (px_usd * fx_rate * (1.0 + fee_rate)))
            b3.text_input(
                "예상 매수 수량 / 매수 사유",
                value=f"약 {est_shares}주 (@${px_usd:,.2f}) | {row_sel['진단 사유']}",
                disabled=True,
            )
            if b4.button(f"🚀 {buy_sym} 수동 매수", type="primary", use_container_width=True):
                live_df = st.session_state["live_candles"].get(buy_sym)
                sig_b = evaluate_live_champion_signals(live_df, buy_sym) if live_df is not None else {}
                ind_snap = {
                    "atr_pct": sig_b.get("atr_pct", 0.0),
                    "dist_from_high_pct": sig_b.get("dist_from_high_pct", 0.0),
                    "ema_10d": sig_b.get("ema_10d", 0.0),
                    "ema_30d": sig_b.get("ema_30d", 0.0),
                    "ema_200h": sig_b.get("ema_200h", 0.0),
                    "alpha_score": sig_b.get("alpha_score", 0.0),
                }
                ok, msg = execute_krw_buy(
                    portfolio,
                    ticker=buy_sym,
                    price_usd=px_usd,
                    fx_rate=fx_rate,
                    allocation_krw=float(alloc_krw),
                    reason=f"[수동 매수] {row_sel['진단 사유']}",
                    indicators=ind_snap,
                )
                if ok:
                    gh_status = _persist_all(portfolio, gh_cfg, f"trade: manual buy {buy_sym}")
                    msgs = [("success", msg)]
                    if gh_status:
                        msgs.append(("info", gh_status))
                    st.session_state["last_action_messages"] = msgs
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.info("상단의 **'실시간 1시간봉 API 검사 & 자동 매수/매도 실행'** 버튼을 누르면 100개 이상 종목의 스캔 결과가 여기에 표시됩니다.")


# ---------------- Tab 3: Trading Journal ---------------- #
with tab3:
    st.subheader("📓 실전 모의투자 매매일지 (매매 이유 · 승률 · 수익률 · 복기 메모)")
    jm = compute_journal_metrics(portfolio)

    j1, j2, j3, j4, j5, j6 = st.columns(6)
    j1.metric("총 완료 매매", f"{jm['total_trades']}건")
    j2.metric("실전 승률", f"{jm['win_rate_pct']:.1f}%", f"{jm['wins']}승 {jm['losses']}패")
    j3.metric("평균 수익률", f"{jm['avg_return_pct']:+.2f}%", f"승 {jm['avg_win_pct']:+.1f}% / 패 {jm['avg_loss_pct']:+.1f}%")
    j4.metric("손익비 (PF)", f"{jm['profit_factor']:.2f}")
    j5.metric("평균 보유시간", f"{jm['avg_holding_hours']:.1f}시간")
    j6.metric("누적 지출 수수료", f"₩{jm['total_fees_krw']:,.0f}")

    trade_history: List[Dict[str, Any]] = portfolio.get("trade_history", [])
    if trade_history:
        rows = []
        for t in reversed(trade_history):
            rows.append({
                "거래 ID": t["trade_id"],
                "종목": t["ticker"],
                "수량": f"{int(t['shares'])}주",
                "진입가($)": round(float(t["entry_price_usd"]), 2),
                "청산가($)": round(float(t["exit_price_usd"]), 2),
                "실현 손익(₩)": int(round(float(t["net_pnl_krw"]))),
                "수익률(%)": round(float(t["pnl_pct"]), 2),
                "주가변동(%)": round(float(t.get("price_return_pct", 0.0)), 2),
                "보유(시간)": float(t.get("holding_hours", 0.0)),
                "수수료(₩)": int(round(float(t["total_fees_krw"]))),
                "매수 이유": t.get("entry_reason", ""),
                "매도 이유": t.get("exit_reason", ""),
                "복기 메모": t.get("memo", ""),
                "진입 일시": t.get("entry_time", "")[:16].replace("T", " "),
                "청산 일시": t.get("exit_time", "")[:16].replace("T", " "),
            })
        journal_df = pd.DataFrame(rows)
        st.dataframe(journal_df.drop(columns=["거래 ID"]), use_container_width=True)

        csv_bytes = journal_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 매매일지 전체 CSV 다운로드 (엑셀 호환)",
            data=csv_bytes,
            file_name="trading_journal.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### ✍️ 거래별 진입/청산 지표 상세 비교 & 복기 메모 작성")
        trade_labels = {
            f"[{t['ticker']}] {t['exit_time'][:16].replace('T', ' ')} | ₩{t['net_pnl_krw']:+,.0f} ({t['pnl_pct']:+.2f}%)": t
            for t in reversed(trade_history)
        }
        chosen_label = st.selectbox("상세 보기 및 메모 남길 거래 선택", list(trade_labels.keys()))
        chosen_t = trade_labels[chosen_label]

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**🟢 매수 진입 스냅샷**")
            st.write(f"- **매수 이유**: {chosen_t.get('entry_reason', '-')}")
            st.write(f"- **단가 / 환율**: `${chosen_t['entry_price_usd']:.2f}` / `₩{chosen_t.get('entry_fx_rate', 0):,.1f}`")
            e_ind = chosen_t.get("entry_indicators", {})
            if e_ind:
                st.json(e_ind)
        with d2:
            st.markdown("**🔴 매도 청산 스냅샷**")
            st.write(f"- **매도 이유**: {chosen_t.get('exit_reason', '-')}")
            st.write(f"- **단가 / 환율**: `${chosen_t['exit_price_usd']:.2f}` / `₩{chosen_t.get('exit_fx_rate', 0):,.1f}`")
            x_ind = chosen_t.get("exit_indicators", {})
            if x_ind:
                st.json(x_ind)

        memo_val = st.text_area("📝 매매 복기 메모", value=chosen_t.get("memo", ""), height=90)
        if st.button("💾 복기 메모 저장"):
            if update_trade_memo(portfolio, chosen_t["trade_id"], memo_val):
                gh_status = _persist_all(portfolio, gh_cfg, f"journal: update memo for {chosen_t['ticker']}")
                st.session_state["last_action_messages"] = [
                    ("success", f"{chosen_t['ticker']} 매매일지 메모가 저장되었습니다." + (f" ({gh_status})" if gh_status else ""))
                ]
                st.rerun()
    else:
        st.info("아직 청산이 완료된 거래가 없습니다. 매수 후 청산(자동/수동)이 발생하면 매수·매도 이유와 승률·수익률이 매매일지에 자동 기록됩니다.")


# ---------------- Tab 4: Equity Curve ---------------- #
with tab4:
    st.subheader("📈 계좌 자산 변화 곡선 (원화 1,000만원 기준)")
    eq_hist = portfolio.get("equity_history", [])
    if len(eq_hist) >= 2:
        eq_df = pd.DataFrame(eq_hist)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])

        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df["timestamp"],
                y=eq_df["total_equity_krw"],
                mode="lines+markers",
                name="총 평가자산 (₩)",
                line=dict(color="#2563EB", width=2.5),
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df["timestamp"],
                y=[portfolio["initial_capital_krw"]] * len(eq_df),
                mode="lines",
                name="원금 기준선 (₩10,000,000)",
                line=dict(color="#94A3B8", dash="dash"),
            )
        )
        fig_eq.update_layout(
            title="원화 기준 포트폴리오 자산 성장 추이",
            xaxis_title="일시 (UTC)",
            yaxis_title="평가금액 (원)",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("API 스캔 또는 매매가 진행되면 원화 자산 추이 그래프가 그려집니다.")

    if trade_history:
        st.markdown("#### 📊 거래별 실현 손익 (₩)")
        th_bar = pd.DataFrame(trade_history)
        colors = ["#16A34A" if v > 0 else "#DC2626" for v in th_bar["net_pnl_krw"]]
        fig_bar = go.Figure(
            go.Bar(
                x=[f"{r['ticker']} ({r['exit_time'][5:10]})" for _, r in th_bar.iterrows()],
                y=th_bar["net_pnl_krw"],
                marker_color=colors,
            )
        )
        fig_bar.update_layout(template="plotly_white", height=320, xaxis_title="거래", yaxis_title="순손익 (₩)")
        st.plotly_chart(fig_bar, use_container_width=True)


# ---------------- Tab 5: Strategy & Cloud Persistence Guide ---------------- #
with tab5:
    st.subheader("📖 1시간 봉 챔피언 DNA 매매 규칙 & 클라우드 영구 저장 안내")
    st.markdown(
        """
    #### 1. 실시간 1시간 봉 매수 기준 (4단계 필터 모두 충족 시)
    - **거시 골든크로스**: 10일 거시선(65개 1시간봉 EMA) > 30일 거시선(195개 1시간봉 EMA)
    - **장기 추세 지지**: 현재 주가 > 200시간선(200개 1시간봉 EMA)
    - **변동성 파워**: 1시간봉 ATR(14) 비율 ≥ **1.1%** (저변동 횡보주 배제)
    - **신고가 돌파권**: 최근 60개 1시간봉(약 10거래일) 최고가 대비 **-6% 이내**

    #### 2. 자금 관리 & 20종목 정수 분할 원칙
    - **기본 원화 자본**: `₩10,000,000` (실시간 `USD/KRW` 환율 자동 반영)
    - **20종목 균등 분할 & 소수점 거래 불허용**: 가용 현금을 남은 슬롯 수로 균등 배분해 **정수 1주 이상**만 매수합니다. 1주 가격이 슬롯 배분금보다 비싼 고가주는 자동 건너뛰고 다음 적격 종목을 매수합니다.
    - **수수료**: 편도 **0.1%** 매수/매도 각각 차감.

    #### 3. 청산 및 손절 원칙
    - **-12% 와이드 하드스탑**: 손절가 이하로 하락하거나 개장 갭하락 시 즉시 청산 (갭하락 시 개장 시가 체결 반영)
    - **거시 데드크로스**: 10일 거시선 ≤ 30일 거시선 교차 시 전량 매도

    #### 4. Streamlit Cloud에서 매매일지를 영구 보존하는 방법 (선택 사항)
    Streamlit 무료 서버는 일정 시간 미사용 시 재부팅되며 서버 내부 파일이 초기화될 수 있습니다. 아래 중 편한 방법을 쓰세요:
    1. **간편 백업/복원**: 왼쪽 사이드바의 **'계좌·매매일지 백업 (JSON)'** 버튼으로 저장하고 필요 시 업로드하여 1초 만에 복원.
    2. **GitHub 완전 자동 동기화**: Streamlit Cloud 앱 설정 → **Settings → Secrets**에 아래 2줄을 넣으면 매매 및 메모 저장 시 자동으로 `jeaweon/stock-2` 저장소에 커밋됩니다.
       ```toml
       GITHUB_TOKEN = "ghp_여기에_GitHub_Personal_Access_Token_입력"
       GITHUB_REPO = "jeaweon/stock-2"
       ```
    """
    )
