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

def calculate_volume_profile(df, window=120, bins=50):
    """지정된 기간 동안의 매물대(POC) 계산"""
    poc_series = pd.Series(index=df.index, dtype=float)
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        if window_df.empty: continue
        hist, bin_edges = np.histogram(window_df['Close'], bins=bins, weights=window_df['Volume'])
        max_vol_idx = np.argmax(hist)
        poc_series.iloc[i] = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
    return poc_series

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
st.title("📉 BB 하단 분할 매수 + 다중 진입/청산 조건")

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
    * **매수:** 아래 4가지 조건 중 하나를 만족하면서 BB 하단 터치 시
        1. 1차 터치: 잔고 30% 매수 ➔ **+3% 익절**
        2. 2차 터치: 잔고 50% 추가 매수 ➔ **+1% 익절**
        3. 3차 터치: 잔고 50% 추가 매수 ➔ **0% 본전 탈출**
    * **매수 추가 조건:**
        1. 매물대 돌파 + (5일선 > 20일선)
        2. 거래량 1.5배 돌파 + (5일선 > 20일선)
        3. VCP (BB수축 + 거래량 3배 돌파)
        4. 매물대 지지 반등
    * **매도 추가 조건:** 20일선 > 5일선 역전 시 (단, 수익 중일 때만 매도)
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
                # ==========================================
                # 1. 일봉 데이터 다운로드 및 일봉 지표 계산
                # ==========================================
                df_daily = yf.download(ticker, start=start_date - datetime.timedelta(days=200), end=end_date_yf, interval='1d', progress=False)
                
                if df_daily.empty:
                    st.error("일봉 데이터를 불러오지 못했습니다.")
                else:
                    df_daily = flatten_yf_columns(df_daily, ticker)
                    df_daily.dropna(inplace=True)
                    
                    df_daily['MA5_Daily'] = df_daily['Close'].rolling(window=5).mean()
                    df_daily['MA20_Daily'] = df_daily['Close'].rolling(window=20).mean()
                    df_daily['Vol90_MA_Daily'] = df_daily['Volume'].rolling(window=90).mean()
                    df_daily['POC_Daily'] = calculate_volume_profile(df_daily)
                    
                    df_daily['Date_Str'] = df_daily.index.strftime('%Y-%m-%d')
                    daily_indicators = df_daily[['Date_Str', 'MA5_Daily', 'MA20_Daily', 'Vol90_MA_Daily', 'POC_Daily']].copy()

                # ==========================================
                # 2. 시간 봉 데이터 수집 및 병합
                # ==========================================
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
                    df_intraday['BB_Upper'] = bb.bollinger_hband()
                    df_intraday['BB_Width'] = (df_intraday['BB_Upper'] - df_intraday['BB_Lower']) / df_intraday['Close'].rolling(window=20).mean()
                    
                    df_intraday['Date_Str'] = df_intraday.index.strftime('%Y-%m-%d')
                    df = pd.merge(df_intraday.reset_index(), daily_indicators, on='Date_Str', how='left')
                    df.set_index('Datetime' if 'Datetime' in df.columns else 'Date', inplace=True)
                    
                    df['MA5'] = df['MA5_Daily'].ffill()
                    df['MA20'] = df['MA20_Daily'].ffill()
                    df['Vol90_MA'] = df['Vol90_MA_Daily'].ffill()
                    df['POC'] = df['POC_Daily'].ffill()
                    
                    df.dropna(subset=['MA20', 'BB_Lower'], inplace=True)
                    
                    if len(df) < 25:
                        st.error("데이터 개수가 너무 적습니다. 조회 기간을 넓혀주세요.")
                    else:
                        # ==========================================
                        # 3. 모의투자 시뮬레이션
                        # ==========================================
                        INITIAL_CAPITAL = 10000000
                        cash = INITIAL_CAPITAL
                        shares = 0.0
                        avg_price = 0.0
                        bb_step = 0
                        
                        first_buy_date = None
                        last_buy_date = None
                        current_buy_reason = "" # 현재 포지션 진입 사유 저장
                        
                        trade_log = []
                        buy_markers = []
                        sell_markers = []
                        
                        for i in range(20, len(df)):
                            date = df.index[i].strftime('%Y-%m-%d %H:%M')
                            close = float(df['Close'].iloc[i])
                            low = float(df['Low'].iloc[i])
                            open_price = float(df['Open'].iloc[i])
                            vol = float(df['Volume'].iloc[i])
                            
                            ma5 = float(df['MA5'].iloc[i])
                            ma20 = float(df['MA20'].iloc[i])
                            vol90 = float(df['Vol90_MA'].iloc[i])
                            poc = float(df['POC'].iloc[i])
                            bb_lower = float(df['BB_Lower'].iloc[i])
                            bb_width = float(df['BB_Width'].iloc[i])
                            prev_price = float(df['Close'].iloc[i-1])
                            
                            is_bb_touch = low <= bb_lower
                            
                            # 매수 4가지 조건 검사 및 텍스트 매핑
                            cond1 = (prev_price < poc) and (close > poc) and (ma5 > ma20)
                            cond2 = (vol > vol90 * 1.5) and (ma5 > ma20)
                            cond3 = (bb_width < 0.05) and (vol > vol90 * 3.0)
                            cond4 = (abs(close - poc) / poc < 0.02) and (close > open_price)
                            
                            is_buy_condition_met = False
                            buy_condition_text = ""
                            
                            if is_bb_touch:
                                if cond1:
                                    is_buy_condition_met = True
                                    buy_condition_text = "조건1(매물대 돌파)"
                                elif cond2:
                                    is_buy_condition_met = True
                                    buy_condition_text = "조건2(거래량 1.5배)"
                                elif cond3:
                                    is_buy_condition_met = True
                                    buy_condition_text = "조건3(VCP)"
                                elif cond4:
                                    is_buy_condition_met = True
                                    buy_condition_text = "조건4(매물대 지지)"
                            
                            # 포지션 없을 때 -> 신규 진입 (1차: 30%)
                            if shares == 0:
                                if is_buy_condition_met:
                                    bb_step = 1
                                    invest_amount = cash * 0.30
                                    bought_shares = invest_amount / close
                                    
                                    shares += bought_shares
                                    cash -= invest_amount
                                    avg_price = close
                                    
                                    first_buy_date = date
                                    last_buy_date = date.split()[0]
                                    current_buy_reason = f"1차 매수 [{buy_condition_text}]"
                                    
                                    buy_markers.append({'date': date, 'price': low, 'reason': current_buy_reason})
                            
                            # 포지션 있을 때 -> 분할 매수 또는 청산
                            elif shares > 0:
                                # 익절 목표가 설정
                                if bb_step == 1: target_pct = 0.03
                                elif bb_step == 2: target_pct = 0.01
                                else: target_pct = 0.0
                                
                                target_price = avg_price * (1 + target_pct)
                                
                                is_sold = False
                                sell_reason = ""
                                
                                # 매도 조건 1: 목표 수익률 도달
                                if close >= target_price:
                                    is_sold = True
                                    sell_reason = f"목표 달성 (+{target_pct*100:.0f}%)"
                                
                                # 매도 조건 2: 20일선 > 5일선일 때 (수익 중일 때만)
                                elif (ma20 > ma5) and (close > avg_price):
                                    is_sold = True
                                    sell_reason = "20일선 > 5일선 데드크로스 (수익실현)"
                                    
                                # 추가 매수 조건: 익절/매도 실패하고 다시 조건 만족 + BB하단 터치 시
                                elif is_buy_condition_met and date.split()[0] != last_buy_date and bb_step < 3:
                                    bb_step += 1
                                    invest_amount = cash * 0.50
                                    bought_shares = invest_amount / close
                                    
                                    total_cost = (shares * avg_price) + invest_amount
                                    shares += bought_shares
                                    cash -= invest_amount
                                    avg_price = total_cost / shares
                                    
                                    last_buy_date = date.split()[0]
                                    current_buy_reason += f" / {bb_step}차 [{buy_condition_text}]"
                                    
                                    buy_markers.append({'date': date, 'price': low, 'reason': f"{bb_step}차 추가매수 [{buy_condition_text}]"})
                                
                                # 매도 실행
                                if is_sold:
                                    sell_amount = shares * close
                                    cash += sell_amount
                                    trade_return = (close - avg_price) / avg_price * 100
                                    
                                    trade_log.append({
                                        '최초 진입일': first_buy_date,
                                        '진입 사유': current_buy_reason,
                                        '최종 청산일': date,
                                        '평균 매수단가': round(avg_price, 2),
                                        '매도단가': round(close, 2),
                                        '수익률(%)': round(trade_return, 2),
                                        '매도사유': sell_reason
                                    })
                                    sell_markers.append({'date': date, 'price': close, 'reason': sell_reason})
                                    
                                    shares = 0.0
                                    avg_price = 0.0
                                    bb_step = 0
                                    current_buy_reason = ""

                        final_equity = cash + (shares * float(df['Close'].iloc[-1]))
                        total_cumulative_return = ((final_equity / INITIAL_CAPITAL) - 1) * 100

                        # ==========================================
                        # 4. 결과 UI 대시보드
                        # ==========================================
                        log_df = pd.DataFrame(trade_log)
                        
                        st.subheader(f"🎯 [{ticker}] 모의투자 결과 요약 (다중 조건)")
                        
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
                                # 컬럼 순서 재배치
                                log_df = log_df[['최초 진입일', '최종 청산일', '진입 사유', '평균 매수단가', '매도단가', '수익률(%)', '매도사유']]
                                st.dataframe(log_df, use_container_width=True)
                            
                        st.subheader(f"📊 매수/매도 타점 차트 ({selected_tf} 주가 + BB 하단 + 일봉 이평선)")
                        
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, subplot_titles=(f'{ticker} 주가 흐름 및 지표', '거래량'),
                                            row_width=[0.2, 0.7])
                        
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='purple', width=1, dash='dot'), name='BB 하단'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='gray', width=1), name='일봉 20선'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='일봉 5선'), row=1, col=1)
                        
                        if buy_markers:
                            fig.add_trace(go.Scatter(x=[m['date'] for m in buy_markers], y=[m['price'] for m in buy_markers], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')), name='Buy', text=[m['reason'] for m in buy_markers], hoverinfo='text'), row=1, col=1)
                        if sell_markers:
                            fig.add_trace(go.Scatter(x=[m['date'] for m in sell_markers], y=[m['price'] for m in sell_markers], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')), name='Sell', text=[m['reason'] for m in sell_markers], hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
