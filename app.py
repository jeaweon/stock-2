import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import datetime
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# yfinance 경고문구 숨김
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

# 웹 페이지 기본 설정
st.set_page_config(page_title="주식 백테스트 대시보드", layout="wide")

st.title("📈 주식 맞춤형 돌파/수축 백테스트")

# 왼쪽 사이드바 UI 구성
with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    
    ticker = st.text_input("종목 코드 입력 (예: 005930.KS, AAPL)", "005930.KS")
    
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=365 * 5)
    
    date_range = st.date_input(
        "조회 기간 선택",
        value=(five_years_ago, today),
        max_value=today
    )
    
    run_btn = st.button("🚀 백테스트 실행하기", use_container_width=True)

if run_btn:
    if len(date_range) != 2:
        st.warning("시작일과 종료일을 모두 선택해 주세요.")
    else:
        start_date, end_date = date_range
        
        with st.spinner(f"'{ticker}' 데이터를 가져와 분석 중입니다..."):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.error("데이터를 불러오지 못했습니다. 종목 코드와 날짜를 다시 확인해 주세요.")
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
                    df['Vol90_MA'] = df['Volume'].rolling(window=90).mean()
                    
                    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
                    df['BB_Upper'] = bb.bollinger_hband()
                    df['BB_Lower'] = bb.bollinger_lband()
                    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] 
                    
                    df['POC'] = calculate_volume_profile(df)
                    
                    position = 0
                    buy_date, buy_price = None, 0
                    current_buy_reason = "" # 현재 진입한 매수 조건 기록
                    trade_log = []
                    
                    for i in range(120, len(df)):
                        date = df.index[i].strftime('%Y-%m-%d')
                        price = float(df['Close'].iloc[i])
                        open_price = float(df['Open'].iloc[i])
                        vol = float(df['Volume'].iloc[i])
                        ma5 = float(df['MA5'].iloc[i])
                        ma20 = float(df['MA20'].iloc[i])
                        vol90 = float(df['Vol90_MA'].iloc[i])
                        poc = float(df['POC'].iloc[i])
                        bb_width = float(df['BB_Width'].iloc[i])
                        prev_price = float(df['Close'].iloc[i-1])
                        
                        cond1 = (prev_price < poc) and (price > poc) and (ma5 > ma20)
                        cond2 = (vol > vol90 * 1.5) and (ma5 > ma20)
                        cond3 = (bb_width < 0.05) and (vol > vol90 * 3.0)
                        cond4 = (abs(price - poc) / poc < 0.02) and (price > open_price)
                        
                        is_buy_signal = cond1 or cond2 or cond3 or cond4
                        is_sell_signal = ma20 > ma5
                        
                        if is_buy_signal and position == 0:
                            position = 1
                            buy_date = date
                            buy_price = price
                            
                            # 어떤 조건이 먼저 만족되었는지 텍스트로 기록
                            if cond1: current_buy_reason = "조건1: 매물대 돌파"
                            elif cond2: current_buy_reason = "조건2: 거래량 1.5배 돌파"
                            elif cond3: current_buy_reason = "조건3: VCP 수축 폭발"
                            elif cond4: current_buy_reason = "조건4: 매물대 지지 반등"
                            
                        elif is_sell_signal and position == 1:
                            sell_price = price
                            trade_return = (sell_price - buy_price) / buy_price * 100
                            
                            trade_log.append({
                                '매수일': buy_date,
                                '매수단가': round(buy_price, 2),
                                '매도일': date,
                                '매도단가': round(sell_price, 2),
                                '수익률(%)': round(trade_return, 2),
                                '매수조건': current_buy_reason # 매매 일지에 매수 조건 추가
                            })
                            
                            position = 0
                            buy_date, buy_price = None, 0
                            current_buy_reason = ""

                    log_df = pd.DataFrame(trade_log)
                    
                    if log_df.empty:
                        st.warning("선택한 기간 동안 설정한 매수 조건에 맞는 구간이 없었습니다.")
                    else:
                        st.subheader(f"🎯 [{ticker}] 전체 백테스트 요약")
                        
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
                        
                        # --- 2. 매수 조건별 성과 분석 ---
                        st.subheader("🛠️ 매수 조건별 성과 분석")
                        
                        # 매수조건별로 그룹화하여 통계 산출
                        condition_stats = []
                        for condition, group in log_df.groupby('매수조건'):
                            trades = len(group)
                            wins = len(group[group['수익률(%)'] > 0])
                            c_win_rate = (wins / trades) * 100
                            
                            # 해당 조건으로 거래했을 때의 누적 수익률 계산
                            c_cum_ret = 1.0
                            for r in group['수익률(%)']:
                                c_cum_ret *= (1 + r / 100)
                            c_cum_ret = (c_cum_ret - 1) * 100
                            
                            avg_ret = group['수익률(%)'].mean()
                            
                            condition_stats.append({
                                '매수 조건': condition,
                                '매매 횟수': trades,
                                '승률(%)': round(c_win_rate, 2),
                                '누적 수익률(%)': round(c_cum_ret, 2),
                                '평균 수익률(%)': round(avg_ret, 2)
                            })
                            
                        cond_df = pd.DataFrame(condition_stats).sort_values(by='누적 수익률(%)', ascending=False)
                        st.dataframe(cond_df, use_container_width=True)
                        
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
                        
                        # 매수 마커에 마우스를 올리면 어떤 조건으로 샀는지 텍스트 표시
                        buy_hover_texts = [f"Buy: {log['매수조건']}<br>Price: {log['매수단가']}" for log in trade_log]

                        fig.add_trace(go.Scatter(x=buy_prices.index, y=buy_prices, 
                                                 mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')),
                                                 name='Buy Signal', text=buy_hover_texts, hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(x=sell_prices.index, y=sell_prices, 
                                                 mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')),
                                                 name='Sell Signal'), row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                        
                        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                        
                        st.plotly_chart(fig, use_container_width=True)
