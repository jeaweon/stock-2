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
                st.error("데이터를 불러오지 못했습니다. 종목 코드를 다시 확인해 주세요.")
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
                    # 지표 계산
                    df['MA5'] = df['Close'].rolling(window=5).mean()
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['Vol90_MA'] = df['Volume'].rolling(window=90).mean()
                    
                    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
                    df['BB_Upper'] = bb.bollinger_hband()
                    df['BB_Lower'] = bb.bollinger_lband()
                    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] 
                    
                    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
                    df['POC'] = calculate_volume_profile(df)
                    
                    # 가상 계좌(포트폴리오) 세팅
                    INITIAL_CAPITAL = 10000000 # 1천만 원 시작
                    cash = INITIAL_CAPITAL
                    shares = 0
                    avg_price = 0.0
                    
                    # 상태 관리 변수
                    current_strategy = None
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
                        
                        # 엄격한 원래 4가지 조건
                        cond1 = (prev_price < poc) and (close > poc) and (ma5 > ma20)
                        cond2 = (vol > vol90 * 1.5) and (ma5 > ma20)
                        cond3 = (bb_width < 0.05) and (vol > vol90 * 3.0)
                        cond4 = (abs(close - poc) / poc < 0.02) and (close > open_price)
                        is_breakout_buy = cond1 or cond2 or cond3 or cond4
                        
                        # 1. 포지션이 없을 때 (신규 매수 진입)
                        if shares == 0:
                            if is_bb_touch:
                                # 볼린저 밴드 하단 1차 매수 (잔고의 30%)
                                current_strategy = "BB_Bottom"
                                bb_step = 1
                                invest_amount = cash * 0.30
                                bought_shares = invest_amount / close
                                
                                shares += bought_shares
                                cash -= invest_amount
                                avg_price = close
                                
                                first_buy_date = date
                                last_buy_date = date
                                
                                buy_markers.append({'date': date, 'price': low, 'reason': "BB 1차 매수 (30%)"})
                                
                            elif is_breakout_buy:
                                # 돌파 매매 (잔고의 100%)
                                current_strategy = "Breakout"
                                invest_amount = cash * 1.0
                                bought_shares = invest_amount / close
                                
                                shares += bought_shares
                                cash -= invest_amount
                                avg_price = close
                                highest_since_entry = close
                                
                                first_buy_date = date
                                buy_markers.append({'date': date, 'price': close, 'reason': "돌파 매수 (100%)"})
                                
                        # 2. 포지션 보유 중일 때 (추가 매수 또는 매도 청산)
                        elif shares > 0:
                            is_sold = False
                            sell_reason = ""
                            
                            # A. 돌파 매매 보유 중 -> 트레일링 스탑 적용
                            if current_strategy == "Breakout":
                                highest_since_entry = max(highest_since_entry, close)
                                trailing_stop = highest_since_entry - (atr_mult * df['ATR'].iloc[i])
                                
                                if close < trailing_stop:
                                    is_sold = True
                                    sell_reason = "ATR 트레일링 스탑"
                                elif close < ma20:
                                    is_sold = True
                                    sell_reason = "20일선 이탈"
                                    
                            # B. 볼린저 밴드 매매 보유 중 -> 익절 또는 물타기
                            elif current_strategy == "BB_Bottom":
                                # 현재 step에 따른 익절 목표가 설정
                                if bb_step == 1: target_pct = 0.03   # 3% 익절
                                elif bb_step == 2: target_pct = 0.01 # 1% 익절
                                else: target_pct = 0.00              # 0% 본전 탈출
                                
                                target_price = avg_price * (1 + target_pct)
                                
                                # 익절 조건 달성 시 매도
                                if close >= target_price:
                                    is_sold = True
                                    sell_reason = f"BB {bb_step}차 목표 익절 (+{target_pct*100:.0f}%)"
                                
                                # 익절 실패하고 또 하단에 닿았을 때 (당일 중복 매수 방지)
                                elif is_bb_touch and date != last_buy_date and bb_step < 3:
                                    bb_step += 1
                                    invest_amount = cash * 0.50 # 남은 잔고의 50% 추가 매수
                                    bought_shares = invest_amount / close
                                    
                                    # 평단가 재계산
                                    total_cost = (shares * avg_price) + invest_amount
                                    shares += bought_shares
                                    cash -= invest_amount
                                    avg_price = total_cost / shares
                                    last_buy_date = date
                                    
                                    buy_markers.append({'date': date, 'price': low, 'reason': f"BB {bb_step}차 매수 (잔고 50%)"})
                            
                            # C. 매도 실행 및 일지 기록
                            if is_sold:
                                sell_amount = shares * close
                                cash += sell_amount
                                trade_return = (close - avg_price) / avg_price * 100
                                
                                trade_log.append({
                                    '최초 진입일': first_buy_date,
                                    '최종 청산일': date,
                                    '평균 매수단가': round(avg_price, 2),
                                    '매도단가': round(close, 2),
                                    '수익률(%)': round(trade_return, 2),
                                    '전략': current_strategy,
                                    '청산사유': sell_reason
                                })
                                
                                sell_markers.append({'date': date, 'price': close, 'reason': sell_reason})
                                
                                # 포지션 초기화
                                shares = 0
                                avg_price = 0.0
                                bb_step = 0
                                current_strategy = None

                    # 백테스트 종료 후 남은 주식 현금화
                    final_equity = cash + (shares * float(df['Close'].iloc[-1]))
                    total_cumulative_return = ((final_equity / INITIAL_CAPITAL) - 1) * 100

                    # --- 결과 출력 ---
                    log_df = pd.DataFrame(trade_log)
                    
                    if log_df.empty:
                        st.warning("조건에 맞는 매매가 발생하지 않았습니다.")
                    else:
                        st.subheader(f"🎯 [{ticker}] 포트폴리오 백테스트 요약 (초기: 1천만원)")
                        
                        win_trades = log_df[log_df['수익률(%)'] > 0]
                        win_rate = (len(win_trades) / len(log_df)) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("총 매매 완료(Round-trip)", f"{len(log_df)}회")
                        col2.metric("승률 (Win Rate)", f"{win_rate:.2f}%")
                        col3.metric("최종 자산 평가액", f"{int(final_equity):,}원")
                        col4.metric("총 누적 수익률", f"{total_cumulative_return:.2f}%")
                        
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
                            b_dates = [m['date'] for m in buy_markers]
                            b_prices = [m['price'] for m in buy_markers]
                            b_texts = [m['reason'] for m in buy_markers]
                            fig.add_trace(go.Scatter(x=b_dates, y=b_prices, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')), name='Buy', text=b_texts, hoverinfo='text'), row=1, col=1)
                            
                        if sell_markers:
                            s_dates = [m['date'] for m in sell_markers]
                            s_prices = [m['price'] for m in sell_markers]
                            s_texts = [m['reason'] for m in sell_markers]
                            fig.add_trace(go.Scatter(x=s_dates, y=s_prices, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')), name='Sell', text=s_texts, hoverinfo='text'), row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
