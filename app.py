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

def calculate_volume_profile(df, window=120, bins=50):
    """지정된 기간 동안의 매물대(Point of Control) 계산"""
    poc_series = pd.Series(index=df.index, dtype=float)
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        hist, bin_edges = np.histogram(window_df['Close'], bins=bins, weights=window_df['Volume'])
        max_vol_idx = np.argmax(hist)
        poc_series.iloc[i] = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
    return poc_series

st.set_page_config(page_title="주식 백테스트 대시보드", layout="wide")
st.title("📈 실전 튜닝: 주식 돌파/수축 백테스트")

with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    ticker = st.text_input("종목 코드 입력 (예: 005930.KS, AAPL)", "005930.KS")
    
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=365 * 5)
    date_range = st.date_input("조회 기간 선택", value=(five_years_ago, today), max_value=today)
    
    st.markdown("---")
    st.markdown("**매도 기준 설정**")
    stop_loss_pct = st.number_input("손절매 기준 (%)", value=5.0, step=1.0)
    take_profit_pct = st.number_input("목표 익절 기준 (%)", value=15.0, step=1.0)
    
    run_btn = st.button("🚀 튜닝된 백테스트 실행", use_container_width=True)

if run_btn:
    if len(date_range) != 2:
        st.warning("시작일과 종료일을 모두 선택해 주세요.")
    else:
        start_date, end_date = date_range
        
        with st.spinner(f"'{ticker}' 데이터를 가져와 분석 중입니다..."):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.error("데이터를 불러오지 못했습니다.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    if ticker in df.columns.get_level_values(1):
                        df = df.xs(ticker, level=1, axis=1)
                    else:
                        df = df.droplevel(1, axis=1)
                        
                df.dropna(inplace=True)
                
                if len(df) < 120:
                    st.error("데이터가 부족합니다. 최소 120일 이상의 기간이 필요합니다.")
                else:
                    df['MA5'] = df['Close'].rolling(window=5).mean()
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    # 90일 평균 거래량을 20일(한 달)로 단축하여 최근 모멘텀에 민감하게 반응하도록 수정
                    df['Vol20_MA'] = df['Volume'].rolling(window=20).mean()
                    
                    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
                    df['BB_Upper'] = bb.bollinger_hband()
                    df['BB_Lower'] = bb.bollinger_lband()
                    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] 
                    
                    df['POC'] = calculate_volume_profile(df)
                    
                    position = 0
                    buy_date, buy_price = None, 0
                    current_buy_reason = ""
                    trade_log = []
                    
                    for i in range(120, len(df)):
                        date = df.index[i].strftime('%Y-%m-%d')
                        price = float(df['Close'].iloc[i])
                        open_price = float(df['Open'].iloc[i])
                        vol = float(df['Volume'].iloc[i])
                        ma5 = float(df['MA5'].iloc[i])
                        ma20 = float(df['MA20'].iloc[i])
                        vol20 = float(df['Vol20_MA'].iloc[i])
                        poc = float(df['POC'].iloc[i])
                        bb_width = float(df['BB_Width'].iloc[i])
                        prev_price = float(df['Close'].iloc[i-1])
                        
                        # [조건 튜닝]
                        cond1 = (prev_price < poc) and (price > poc) and (ma5 > ma20)
                        cond2 = (vol > vol20 * 1.5) and (ma5 > ma20)
                        # 조건 3 완화: 밴드폭 10% 미만 수축, 거래량 2.5배 폭발
                        cond3 = (bb_width < 0.10) and (vol > vol20 * 2.5)
                        # 조건 4 완화: 매물대 지지 오차 범위를 4% 이내로 확대
                        cond4 = (abs(price - poc) / poc < 0.04) and (price > open_price)
                        
                        is_buy_signal = cond1 or cond2 or cond3 or cond4
                        
                        # 포지션이 없을 때 진입
                        if position == 0 and is_buy_signal:
                            position = 1
                            buy_date = date
                            buy_price = price
                            
                            if cond3: current_buy_reason = "조건3: VCP 수축 폭발"
                            elif cond4: current_buy_reason = "조건4: 매물대 지지 반등"
                            elif cond1: current_buy_reason = "조건1: 매물대 돌파"
                            elif cond2: current_buy_reason = "조건2: 거래량 1.5배 돌파"
                            
                        # 포지션이 있을 때 청산 로직 (손절매, 익절, 데드크로스)
                        elif position == 1:
                            is_stop_loss = price <= buy_price * (1 - stop_loss_pct / 100)
                            is_take_profit = price >= buy_price * (1 + take_profit_pct / 100)
                            is_dead_cross = ma20 > ma5
                            
                            sell_reason = ""
                            if is_stop_loss: sell_reason = f"손절매 (-{stop_loss_pct}%)"
                            elif is_take_profit: sell_reason = f"목표 달성 익절 (+{take_profit_pct}%)"
                            elif is_dead_cross: sell_reason = "추세 이탈 (데드크로스)"
                            
                            if is_stop_loss or is_take_profit or is_dead_cross:
                                sell_price = price
                                trade_return = (sell_price - buy_price) / buy_price * 100
                                
                                trade_log.append({
                                    '매수일': buy_date,
                                    '매수단가': round(buy_price, 2),
                                    '매도일': date,
                                    '매도단가': round(sell_price, 2),
                                    '수익률(%)': round(trade_return, 2),
                                    '매수조건': current_buy_reason,
                                    '매도사유': sell_reason
                                })
                                
                                position = 0
                                buy_date, buy_price = None, 0
                                current_buy_reason = ""

                    # --- 결과 출력 ---
                    log_df = pd.DataFrame(trade_log)
                    
                    if log_df.empty:
                        st.warning("조건을 완화했음에도 해당 기간 내 매매 조건이 발생하지 않았습니다.")
                    else:
                        st.subheader(f"🎯 [{ticker}] 튜닝된 백테스트 요약")
                        
                        win_trades = log_df[log_df['수익률(%)'] > 0]
                        win_rate = (len(win_trades) / len(log_df)) * 100
                        
                        cumulative_return = 1.0
                        for ret in log_df['수익률(%)']:
                            cumulative_return *= (1 + ret / 100)
                        cumulative_return = (cumulative_return - 1) * 100

                        col1, col2, col3 = st.columns(3)
                        col1.metric("총 매매 횟수", f"{len(log_df)}회")
                        col2.metric("전체 승률 (Win Rate)", f"{win_rate:.2f}%")
                        col3.metric("전체 누적 수익률", f"{cumulative_return:.2f}%")
                        
                        st.divider()
                        
                        # 종목 분석 리포트 요약
                        st.subheader("📝 매도 사유별 성과")
                        sell_stats = log_df['매도사유'].value_counts().reset_index()
                        sell_stats.columns = ['매도 사유', '발생 횟수']
                        st.dataframe(sell_stats, use_container_width=True)
                        
                        st.divider()
                        
                        st.subheader("📝 상세 매매 일지")
                        st.dataframe(log_df, use_container_width=True)
                        
                        # --- 차트 시각화 ---
                        st.subheader("📊 매수/매도 타점 차트 시각화")
                        
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, subplot_titles=(f'{ticker} 주가 흐름', '거래량'),
                                            row_width=[0.2, 0.7])
                        
                        fig.add_trace(go.Candlestick(x=df.index,
                                        open=df['Open'], high=df['High'],
                                        low=df['Low'], close=df['Close'],
                                        name='주가'), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='blue', width=1), name='5일선'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='red', width=1), name='20일선'), row=1, col=1)
                        
                        buy_dates = [pd.to_datetime(log['매수일']) for log in trade_log]
                        sell_dates = [pd.to_datetime(log['매도일']) for log in trade_log]
                        
                        buy_prices = df.loc[df.index.isin(buy_dates), 'Close']
                        sell_prices = df.loc[df.index.isin(sell_dates), 'Close']
                        
                        buy_hover_texts = [f"Buy: {log['매수조건']}<br>Price: {log['매수단가']}" for log in trade_log]
                        sell_hover_texts = [f"Sell: {log['매도사유']}<br>Return: {log['수익률(%)']}%" for log in trade_log]

                        fig.add_trace(go.Scatter(x=buy_prices.index, y=buy_prices, 
                                                 mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')),
                                                 name='Buy Signal', text=buy_hover_texts, hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(x=sell_prices.index, y=sell_prices, 
                                                 mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')),
                                                 name='Sell Signal', text=sell_hover_texts, hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                        
                        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                        
                        st.plotly_chart(fig, use_container_width=True)
