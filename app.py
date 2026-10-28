import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import datetime
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

def flatten_yf_columns(df, ticker):
    """야후 파이낸스 다중 컬럼 인덱스 평탄화"""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(1):
            return df.xs(ticker, level=1, axis=1)
        else:
            return df.droplevel(1, axis=1)
    return df

# 고정 종목 리스트
TICKERS = {
    "Apple (AAPL)": "AAPL",
    "SOXL (반도체 3배 레버리지)": "SOXL",
    "ETHU (이더리움 2배 레버리지)": "ETHU",
    "TQQQ (나스닥 3배 레버리지)": "TQQQ",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT",
    "TSLL (테슬라 1.5배 레버리지)": "TSLL",
    "Nvidia (NVDA)": "NVDA",
    "KORU (한국 MSCI 3배 레버리지)": "KORU"
}

st.set_page_config(page_title="주식 모의투자 백테스트", layout="wide")
st.title("📉 단순 볼린저 밴드 하단 매수 (+1% 익절 / -1% 손절)")

with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    
    selected_ticker_name = st.selectbox("관심 종목 선택", list(TICKERS.keys()))
    ticker = TICKERS[selected_ticker_name]
    
    timeframe_option = st.selectbox("시간 봉 선택", ["90분 봉 (90m)", "1시간 봉 (1h)", "2시간 봉 (2h)", "4시간 봉 (4h)"], index=0)
    tf_code_map = {"90분 봉 (90m)": "90m", "1시간 봉 (1h)": "1h", "2시간 봉 (2h)": "2h", "4시간 봉 (4h)": "4h"}
    selected_tf = tf_code_map[timeframe_option]
    
    max_days_allowed = 59 if selected_tf == "90m" else 720
    if selected_tf == "90m":
        st.warning("⚠️ '90분 봉'은 최근 60일 이내로만 조회가 가능합니다.")
        
    today = datetime.date.today()
    limit_ago = today - datetime.timedelta(days=max_days_allowed)
    
    date_range = st.date_input(
        "조회 기간 선택", 
        value=(limit_ago, today), 
        max_value=today, 
        min_value=today - datetime.timedelta(days=730)
    )
    
    st.markdown("---")
    
    st.markdown("""
    **초기 자본금: 10,000,000원**
    * **매수 규칙:** 주가 최저가(Low)가 볼린저 밴드 하단 터치 시 **전량 매수 (100%)**
    * **익절 규칙:** 매수가 대비 **+1% 이상 상승 시 전량 매도**
    * **손절 규칙:** 매수가 대비 **-1% 이하 하락 시 전량 매도**
    """)
    
    run_btn = st.button("🚀 데이터 갱신 및 시뮬레이션", use_container_width=True)

if run_btn:
    if len(date_range) != 2:
        st.warning("시작일과 종료일을 모두 선택해 주세요.")
    else:
        start_date, end_date = date_range
        end_date_yf = end_date + datetime.timedelta(days=1)
        
        if selected_tf == "90m" and (end_date - start_date).days > 60:
            st.error("90분 봉은 최대 60일까지만 조회 가능합니다. 달력에서 기간을 줄여주세요.")
        else:
            with st.spinner(f"'{ticker}' 데이터를 가져오는 중..."):
                df_intraday = pd.DataFrame()
                
                if selected_tf == "90m":
                    df_in = yf.download(ticker, start=start_date, end=end_date_yf, interval='90m', progress=False)
                    if not df_in.empty:
                        df_intraday = flatten_yf_columns(df_in, ticker)
                    else:
                        df_in = yf.download(ticker, start=start_date, end=end_date_yf, interval='30m', progress=False)
                        if not df_in.empty:
                            df_temp = flatten_yf_columns(df_in, ticker)
                            df_temp.dropna(inplace=True)
                            df_intraday = df_temp.resample('90min').agg({
                                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                            }).dropna()
                else:
                    df_in = yf.download(ticker, start=start_date, end=end_date_yf, interval='1h', progress=False)
                    if not df_in.empty:
                        df_temp = flatten_yf_columns(df_in, ticker)
                        df_temp.dropna(inplace=True)
                        if selected_tf == '1h':
                            df_intraday = df_temp
                        else:
                            df_intraday = df_temp.resample(selected_tf).agg({
                                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                            }).dropna()

                if df_intraday.empty:
                    st.error("해당 기간의 시간 봉 데이터를 가져올 수 없습니다. 장이 열린 평일이 포함되어 있는지 확인해 주세요.")
                else:
                    df_intraday.dropna(inplace=True)
                    
                    bb = ta.volatility.BollingerBands(close=df_intraday['Close'], window=20, window_dev=2)
                    df_intraday['BB_Lower'] = bb.bollinger_lband()
                    df_intraday['MA20'] = df_intraday['Close'].rolling(window=20).mean()
                    df_intraday.dropna(inplace=True)
                    
                    if len(df_intraday) < 25:
                        st.error("데이터 개수가 너무 적습니다. 조회 기간을 넓혀주세요.")
                    else:
                        # 모의투자 시뮬레이션
                        INITIAL_CAPITAL = 10000000
                        cash = INITIAL_CAPITAL
                        shares = 0.0
                        avg_price = 0.0
                        first_buy_date = None
                        
                        trade_log = []
                        buy_markers = []
                        sell_markers = []
                        
                        for i in range(len(df_intraday)):
                            date = df_intraday.index[i].strftime('%Y-%m-%d %H:%M')
                            close = float(df_intraday['Close'].iloc[i])
                            low = float(df_intraday['Low'].iloc[i])
                            bb_lower = float(df_intraday['BB_Lower'].iloc[i])
                            
                            is_bb_touch = low <= bb_lower
                            
                            # 포지션 없을 때 -> 매수 (100% 매수)
                            if shares == 0:
                                if is_bb_touch:
                                    bought_shares = cash / close
                                    shares = bought_shares
                                    cash = 0.0
                                    avg_price = close
                                    first_buy_date = date
                                    buy_markers.append({'date': date, 'price': low, 'reason': "BB 하단 매수 (100%)"})
                            
                            # 포지션 있을 때 -> 청산 조건 검사
                            elif shares > 0:
                                target_price = avg_price * 1.01  # +1% 익절
                                stop_price = avg_price * 0.99    # -1% 손절
                                
                                is_sold = False
                                sell_reason = ""
                                
                                if close >= target_price:
                                    is_sold = True
                                    sell_reason = "목표 익절 (+1% 달성)"
                                elif close <= stop_price:
                                    is_sold = True
                                    sell_reason = "손절매 (-1% 이탈)"
                                
                                if is_sold:
                                    sell_amount = shares * close
                                    cash += sell_amount
                                    trade_return = (close - avg_price) / avg_price * 100
                                    
                                    trade_log.append({
                                        '매수일': first_buy_date,
                                        '매도일': date,
                                        '매수가': round(avg_price, 2),
                                        '매도가': round(close, 2),
                                        '수익률(%)': round(trade_return, 2),
                                        '매도사유': sell_reason
                                    })
                                    sell_markers.append({'date': date, 'price': close, 'reason': sell_reason})
                                    
                                    shares = 0.0
                                    avg_price = 0.0

                        final_equity = cash + (shares * float(df_intraday['Close'].iloc[-1]))
                        total_cumulative_return = ((final_equity / INITIAL_CAPITAL) - 1) * 100

                        # 결과 UI 대시보드
                        log_df = pd.DataFrame(trade_log)
                        
                        st.subheader(f"🎯 [{ticker}] 모의투자 결과 요약 (+1% 익절 / -1% 손절)")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        if log_df.empty:
                            col1.metric("총 매매 완료", "0회")
                            col2.metric("승률", "0%")
                        else:
                            win_trades = log_df[log_df['수익률(%)'] > 0]
                            win_rate = (len(win_trades) / len(log_df)) * 100
                            col1.metric("총 매매 완료", f"{len(log_df)}회")
                            col2.metric("승률", f"{win_rate:.2f}%")
                            
                        col3.metric("최종 자산 평가액", f"{int(final_equity):,}원")
                        col4.metric("총 누적 수익률", f"{total_cumulative_return:.2f}%")
                        
                        st.divider()
                        
                        if not log_df.empty:
                            col_tbl1, col_tbl2 = st.columns([1, 2])
                            with col_tbl1:
                                st.subheader("📝 매도 사유 분포")
                                sell_stats = log_df['매도사유'].value_counts().reset_index()
                                sell_stats.columns = ['매도 사유', '발생 횟수']
                                st.dataframe(sell_stats, use_container_width=True)
                            
                            with col_tbl2:
                                st.subheader("📝 상세 매매 일지")
                                st.dataframe(log_df, use_container_width=True)
                            
                        st.subheader(f"📊 매수/매도 타점 차트 ({selected_tf} 주가 + BB 하단)")
                        
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, subplot_titles=(f'{ticker} 주가 흐름 및 지표', '거래량'),
                                            row_width=[0.2, 0.7])
                        
                        fig.add_trace(go.Candlestick(x=df_intraday.index, open=df_intraday['Open'], high=df_intraday['High'], low=df_intraday['Low'], close=df_intraday['Close'], name='주가'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['BB_Lower'], line=dict(color='purple', width=1, dash='dot'), name='BB 하단'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['MA20'], line=dict(color='gray', width=1), name='시간봉 20선'), row=1, col=1)
                        
                        if buy_markers:
                            fig.add_trace(go.Scatter(x=[m['date'] for m in buy_markers], y=[m['price'] for m in buy_markers], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')), name='Buy', text=[m['reason'] for m in buy_markers], hoverinfo='text'), row=1, col=1)
                        if sell_markers:
                            fig.add_trace(go.Scatter(x=[m['date'] for m in sell_markers], y=[m['price'] for m in sell_markers], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')), name='Sell', text=[m['reason'] for m in sell_markers], hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=df_intraday.index, y=df_intraday['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
