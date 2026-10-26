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
    """지정된 기간 동안의 매물대(POC) 계산"""
    poc_series = pd.Series(index=df.index, dtype=float)
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        hist, bin_edges = np.histogram(window_df['Close'], bins=bins, weights=window_df['Volume'])
        max_vol_idx = np.argmax(hist)
        poc_series.iloc[i] = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
    return poc_series

st.set_page_config(page_title="주식 백테스트 대시보드", layout="wide")
st.title("📈 혼합 전략: 돌파 & 볼린저 밴드 분할 매수")

with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    ticker = st.text_input("종목 코드 입력 (예: 005930.KS, AAPL)", "005930.KS")
    
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=365 * 5)
    date_range = st.date_input("조회 기간 선택", value=(five_years_ago, today), max_value=today)
    
    st.markdown("---")
    atr_mult = st.number_input("돌파전략 ATR 스탑 배수", value=2.0, step=0.1)
    
    run_btn = st.button("🚀 백테스트 실행", use_container_width=True)

if run_btn:
    if len(date_range) != 2:
        st.warning("시작일과 종료일을 모두 선택해 주세요.")
    else:
        start_date, end_date = date_range
        
        with st.spinner(f"'{ticker}' 데이터를 분석 중입니다..."):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    if ticker in df.columns.get_level_values(1):
                        df = df.xs(ticker, level=1, axis=1)
                    else:
                        df = df.droplevel(1, axis=1)
                        
                df.dropna(inplace=True)
                
                if len(df) < 120:
                    st.error("최소 120일 이상의 데이터가 필요합니다.")
                else:
                    # 1. 지표 계산
                    df['MA5'] = df['Close'].rolling(window=5).mean()
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['Vol90_MA'] = df['Volume'].rolling(window=90).mean()
                    
                    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
                    df['BB_Upper'] = bb.bollinger_hband()
                    df['BB_Lower'] = bb.bollinger_lband()
                    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] 
                    
                    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
                    df['POC'] = calculate_volume_profile(df)
                    
                    # 2. 백테스트 상태 관리 변수 (모의투자 잔고 로직 제거, 비중/평단가 수식화)
                    position_weight = 0.0 # 보유 비중 (0.0 ~ 1.0)
                    avg_price = 0.0       # 평균 매수 단가
                    
                    current_strategy = None
                    current_buy_reason = ""
                    bb_step = 0
                    last_buy_date = None
                    first_buy_date = None
                    highest_since_entry = 0
                    
                    trade_log = []
                    buy_markers = []
                    sell_markers = []
                    
                    for i in range(120, len(df)):
                        date = df.index[i].strftime('%Y-%m-%d')
                        close = float(df['Close'].iloc[i])
                        low = float(df['Low'].iloc[i])
                        open_price = float(df['Open'].iloc[i])
                        vol = float(df['Volume'].iloc[i])
                        ma5 = float(df['MA5'].iloc[i])
                        ma20 = float(df['MA20'].iloc[i])
                        vol90 = float(df['Vol90_MA'].iloc[i])
                        poc = float(df['POC'].iloc[i])
                        bb_width = float(df['BB_Width'].iloc[i])
                        bb_lower = float(df['BB_Lower'].iloc[i])
                        prev_price = float(df['Close'].iloc[i-1])
                        
                        # [조건 파악]
                        is_bb_touch = low <= bb_lower
                        
                        cond1 = (prev_price < poc) and (close > poc) and (ma5 > ma20)
                        cond2 = (vol > vol90 * 1.5) and (ma5 > ma20)
                        cond3 = (bb_width < 0.05) and (vol > vol90 * 3.0)
                        cond4 = (abs(close - poc) / poc < 0.02) and (close > open_price)
                        is_breakout_buy = cond1 or cond2 or cond3 or cond4
                        
                        # 3-A. 신규 매수 진입 (포지션 0일 때)
                        if position_weight == 0.0:
                            if is_bb_touch:
                                current_strategy = "BB_Bottom"
                                bb_step = 1
                                position_weight = 0.30 # 1차: 30%
                                avg_price = close
                                
                                first_buy_date = date
                                last_buy_date = date
                                current_buy_reason = "조건5: 볼린저밴드 하단 터치"
                                buy_markers.append({'date': date, 'price': low, 'reason': "BB 1차 매수 (30%)"})
                                
                            elif is_breakout_buy:
                                current_strategy = "Breakout"
                                position_weight = 1.0 # 돌파매매는 100% 진입
                                avg_price = close
                                highest_since_entry = close
                                
                                if cond1: current_buy_reason = "조건1: 매물대 돌파"
                                elif cond2: current_buy_reason = "조건2: 거래량 1.5배 돌파"
                                elif cond3: current_buy_reason = "조건3: VCP 수축 폭발"
                                elif cond4: current_buy_reason = "조건4: 매물대 지지 반등"
                                
                                first_buy_date = date
                                buy_markers.append({'date': date, 'price': close, 'reason': "돌파 신규매수"})
                                
                        # 3-B. 포지션 보유 중 (추가 매수 또는 매도 청산)
                        elif position_weight > 0.0:
                            is_sold = False
                            sell_reason = ""
                            
                            # 돌파 매매 보유 중 -> 트레일링 스탑 또는 20일선 이탈
                            if current_strategy == "Breakout":
                                highest_since_entry = max(highest_since_entry, close)
                                trailing_stop = highest_since_entry - (atr_mult * df['ATR'].iloc[i])
                                
                                if close < trailing_stop:
                                    is_sold = True
                                    sell_reason = "ATR 트레일링 스탑 이탈"
                                elif close < ma20:
                                    is_sold = True
                                    sell_reason = "20일선 하향 이탈"
                                    
                            # 볼린저 밴드 매매 보유 중 -> 익절 판단 또는 물타기
                            elif current_strategy == "BB_Bottom":
                                # step에 따른 익절 목표가
                                if bb_step == 1: target_pct = 0.03
                                elif bb_step == 2: target_pct = 0.01
                                else: target_pct = 0.0
                                
                                target_price = avg_price * (1 + target_pct)
                                
                                # 1) 목표 달성 시 전량 매도
                                if close >= target_price:
                                    is_sold = True
                                    sell_reason = f"BB {bb_step}차 목표 익절 (+{target_pct*100:.0f}%)"
                                
                                # 2) 목표 달성 못하고 또 하단에 닿았을 때 (당일 중복 매수 방지 및 3차 제한)
                                elif is_bb_touch and date != last_buy_date and bb_step < 3:
                                    bb_step += 1
                                    if bb_step == 2:
                                        add_weight = 0.35 # 남은 비중(70%)의 절반(50%) = 전체의 35%
                                    elif bb_step == 3:
                                        add_weight = 0.175 # 남은 비중(35%)의 절반(50%) = 전체의 17.5%
                                        
                                    # 가중 평단가 재계산
                                    avg_price = ((avg_price * position_weight) + (close * add_weight)) / (position_weight + add_weight)
                                    position_weight += add_weight
                                    last_buy_date = date
                                    buy_markers.append({'date': date, 'price': low, 'reason': f"BB {bb_step}차 추가매수"})
                            
                            # 매도 실행 및 일지 기록
                            if is_sold:
                                trade_return = (close - avg_price) / avg_price * 100
                                
                                trade_log.append({
                                    '최초 진입일': first_buy_date,
                                    '최종 청산일': date,
                                    '매수조건': current_buy_reason,
                                    '평균 매수단가': round(avg_price, 2),
                                    '매도단가': round(close, 2),
                                    '수익률(%)': round(trade_return, 2),
                                    '매도사유': sell_reason
                                })
                                sell_markers.append({'date': date, 'price': close, 'reason': sell_reason})
                                
                                # 변수 초기화
                                position_weight = 0.0
                                avg_price = 0.0
                                bb_step = 0
                                current_strategy = None
                                current_buy_reason = ""

                    # 4. 결과 UI 출력
                    log_df = pd.DataFrame(trade_log)
                    
                    if log_df.empty:
                        st.warning("조건에 맞는 매매가 발생하지 않았습니다.")
                    else:
                        st.subheader(f"🎯 [{ticker}] 백테스트 결과 요약")
                        
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
                        
                        # --- 매수 조건별 & 매도 사유별 통계 표 복구 ---
                        col_tbl1, col_tbl2 = st.columns(2)
                        with col_tbl1:
                            st.subheader("🛠️ 매수 조건별 성과")
                            # 매수조건별 횟수, 승률, 단순합산수익률 계산
                            cond_stats = log_df.groupby('매수조건')['수익률(%)'].agg(
                                매매횟수='count',
                                승률=lambda x: (x > 0).mean() * 100,
                                누적수익률_합산='sum'
                            ).reset_index()
                            cond_stats = cond_stats.round(2)
                            cond_stats.columns = ['매수 조건', '매매 횟수', '승률(%)', '단순 합산 수익률(%)']
                            st.dataframe(cond_stats, use_container_width=True)
                            
                        with col_tbl2:
                            st.subheader("📝 매도 사유별 통계")
                            sell_stats = log_df['매도사유'].value_counts().reset_index()
                            sell_stats.columns = ['매도 사유', '발생 횟수']
                            st.dataframe(sell_stats, use_container_width=True)
                        
                        st.divider()
                        
                        st.subheader("📝 상세 매매 일지")
                        st.dataframe(log_df, use_container_width=True)
                        
                        # --- 차트 시각화 ---
                        st.subheader("📊 매수/매도 타점 차트 시각화")
                        
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, subplot_titles=(f'{ticker} 주가 흐름 및 BB하단', '거래량'),
                                            row_width=[0.2, 0.7])
                        
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='purple', width=1, dash='dot'), name='BB 하단'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='red', width=1), name='20일선'), row=1, col=1)
                        
                        if buy_markers:
                            fig.add_trace(go.Scatter(x=[m['date'] for m in buy_markers], y=[m['price'] for m in buy_markers], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')), name='Buy', text=[m['reason'] for m in buy_markers], hoverinfo='text'), row=1, col=1)
                        if sell_markers:
                            fig.add_trace(go.Scatter(x=[m['date'] for m in sell_markers], y=[m['price'] for m in sell_markers], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')), name='Sell', text=[m['reason'] for m in sell_markers], hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
