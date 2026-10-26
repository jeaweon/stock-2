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
st.title("📈 90분 봉 타점 + 일봉 이평선 합성 백테스트")

with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    ticker = st.text_input("종목 코드 입력 (예: 005930.KS, AAPL)", "005930.KS")
    
    # 시간 봉 옵션에 90분 봉을 기본값으로 추가
    timeframe_option = st.selectbox("시간 봉 선택", ["90분 봉 (90m)", "1시간 봉 (1h)", "2시간 봉 (2h)", "4시간 봉 (4h)"], index=0)
    tf_code_map = {"90분 봉 (90m)": "90m", "1시간 봉 (1h)": "1h", "2시간 봉 (2h)": "2h", "4시간 봉 (4h)": "4h"}
    selected_tf = tf_code_map[timeframe_option]
    
    # 야후 파이낸스 정책: 90m은 최대 60일, 1h 기반 리샘플링은 730일 제공
    max_days_allowed = 60 if selected_tf == "90m" else 730
    if selected_tf == "90m":
        st.warning("⚠️ 야후 무료 데이터 정책상 '90분 봉'은 최근 60일까지만 조회가 가능합니다.")
        
    today = datetime.date.today()
    limit_ago = today - datetime.timedelta(days=max_days_allowed)
    
    date_range = st.date_input(
        "조회 기간 선택", 
        value=(limit_ago, today), 
        max_value=today, 
        min_value=today - datetime.timedelta(days=730)
    )
    
    st.markdown("---")
    atr_mult = st.number_input("돌파전략 ATR 스탑 배수", value=2.0, step=0.1)
    
    run_btn = st.button("🚀 백테스트 실행", use_container_width=True)

if run_btn:
    if len(date_range) != 2:
        st.warning("시작일과 종료일을 모두 선택해 주세요.")
    else:
        start_date, end_date = date_range
        
        # 90m 선택 시 60일 초과 범위 방어 로직
        if selected_tf == "90m" and (end_date - start_date).days > 60:
            st.error("90분 봉은 최대 60일까지만 조회 가능합니다. 기간을 줄여주세요.")
        elif (end_date - start_date).days > 730:
            st.error("시간 봉 데이터는 최대 최근 730일까지만 불러올 수 있습니다.")
        else:
            with st.spinner(f"'{ticker}' 데이터를 수집하고 일봉 이평선을 맵핑 중입니다..."):
                # ==========================================
                # 1. 일봉 데이터 다운로드 및 일봉 기준 지표 계산
                # ==========================================
                df_daily = yf.download(ticker, start=start_date - datetime.timedelta(days=200), end=end_date, interval='1d', progress=False)
                
                if df_daily.empty:
                    st.error("일봉 데이터를 불러오지 못했습니다.")
                else:
                    if isinstance(df_daily.columns, pd.MultiIndex):
                        if ticker in df_daily.columns.get_level_values(1):
                            df_daily = df_daily.xs(ticker, level=1, axis=1)
                        else:
                            df_daily = df_daily.droplevel(1, axis=1)
                    df_daily.dropna(inplace=True)
                    
                    df_daily['MA5_Daily'] = df_daily['Close'].rolling(window=5).mean()
                    df_daily['MA20_Daily'] = df_daily['Close'].rolling(window=20).mean()
                    df_daily['Vol90_MA_Daily'] = df_daily['Volume'].rolling(window=90).mean()
                    df_daily['POC_Daily'] = calculate_volume_profile(df_daily)
                    
                    df_daily['Date_Str'] = df_daily.index.strftime('%Y-%m-%d')
                    daily_indicators = df_daily[['Date_Str', 'MA5_Daily', 'MA20_Daily', 'Vol90_MA_Daily', 'POC_Daily']].copy()
                    
                    # ==========================================
                    # 2. 시간 봉 데이터 다운로드 및 리샘플링
                    # ==========================================
                    fetch_interval = '90m' if selected_tf == '90m' else '1h'
                    df_intraday = yf.download(ticker, start=start_date, end=end_date, interval=fetch_interval, progress=False)
                    
                    if df_intraday.empty:
                        st.error("해당 기간의 시간 봉 데이터를 가져올 수 없습니다. 장이 열리는 평일로 설정되어 있는지 확인하세요.")
                    else:
                        if isinstance(df_intraday.columns, pd.MultiIndex):
                            if ticker in df_intraday.columns.get_level_values(1):
                                df_intraday = df_intraday.xs(ticker, level=1, axis=1)
                            else:
                                df_intraday = df_intraday.droplevel(1, axis=1)
                        df_intraday.dropna(inplace=True)
                        
                        if selected_tf in ['2h', '4h']:
                            df_intraday = df_intraday.resample(selected_tf).agg({
                                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                            }).dropna()
                            
                        # 단기 진입용 볼린저 밴드 및 ATR은 시간 봉 기준으로 산출
                        bb = ta.volatility.BollingerBands(close=df_intraday['Close'], window=20, window_dev=2)
                        df_intraday['BB_Lower'] = bb.bollinger_lband()
                        df_intraday['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df_intraday['Close'].rolling(window=20).mean()
                        df_intraday['ATR'] = ta.volatility.AverageTrueRange(high=df_intraday['High'], low=df_intraday['Low'], close=df_intraday['Close'], window=14).average_true_range()
                        
                        # ==========================================
                        # 3. 시간 봉 데이터에 일봉 지표 병합 (Merge & ffill)
                        # ==========================================
                        df_intraday['Date_Str'] = df_intraday.index.strftime('%Y-%m-%d')
                        df = pd.merge(df_intraday.reset_index(), daily_indicators, on='Date_Str', how='left')
                        df.set_index('Datetime' if 'Datetime' in df.columns else 'Date', inplace=True)
                        
                        df['MA5'] = df['MA5_Daily'].ffill()
                        df['MA20'] = df['MA20_Daily'].ffill()
                        df['Vol90_MA'] = df['Vol90_MA_Daily'].ffill()
                        df['POC'] = df['POC_Daily'].ffill()
                        
                        df.dropna(subset=['MA20', 'BB_Lower'], inplace=True)
                        
                        # 일봉 지표는 이미 계산되었으므로 시간봉 밴드 생성을 위한 20개만 있으면 됨
                        if len(df) < 25:
                            st.error("데이터 개수가 부족하여 백테스트를 실행할 수 없습니다.")
                        else:
                            position_weight = 0.0
                            avg_price = 0.0
                            current_strategy = None
                            current_buy_reason = ""
                            bb_step = 0
                            last_buy_date = None
                            first_buy_date = None
                            highest_since_entry = 0
                            
                            trade_log = []
                            buy_markers = []
                            sell_markers = []
                            
                            for i in range(20, len(df)):
                                date = df.index[i].strftime('%Y-%m-%d %H:%M')
                                close = float(df['Close'].iloc[i])
                                low = float(df['Low'].iloc[i])
                                open_price = float(df['Open'].iloc[i])
                                vol = float(df['Volume'].iloc[i])
                                
                                # 일봉 기준 지표 판별
                                ma5 = float(df['MA5'].iloc[i])
                                ma20 = float(df['MA20'].iloc[i])
                                vol90 = float(df['Vol90_MA'].iloc[i])
                                poc = float(df['POC'].iloc[i])
                                
                                # 시간 봉 기준 지표 판별
                                bb_width = float(df['BB_Width'].iloc[i])
                                bb_lower = float(df['BB_Lower'].iloc[i])
                                prev_price = float(df['Close'].iloc[i-1])
                                
                                is_bb_touch = low <= bb_lower
                                
                                cond1 = (prev_price < poc) and (close > poc) and (ma5 > ma20)
                                cond2 = (vol > vol90 * 1.5) and (ma5 > ma20)
                                cond3 = (bb_width < 0.05) and (vol > vol90 * 3.0)
                                cond4 = (abs(close - poc) / poc < 0.02) and (close > open_price)
                                is_breakout_buy = cond1 or cond2 or cond3 or cond4
                                
                                if position_weight == 0.0:
                                    if is_bb_touch:
                                        current_strategy = "BB_Bottom"
                                        bb_step = 1
                                        position_weight = 0.30
                                        avg_price = close
                                        first_buy_date = date
                                        last_buy_date = date
                                        current_buy_reason = "조건5: BB 하단 터치"
                                        buy_markers.append({'date': date, 'price': low, 'reason': "BB 1차 매수 (30%)"})
                                        
                                    elif is_breakout_buy:
                                        current_strategy = "Breakout"
                                        position_weight = 1.0
                                        avg_price = close
                                        highest_since_entry = close
                                        
                                        if cond1: current_buy_reason = "조건1: 매물대 돌파"
                                        elif cond2: current_buy_reason = "조건2: 거래량 돌파"
                                        elif cond3: current_buy_reason = "조건3: VCP 수축 폭발"
                                        elif cond4: current_buy_reason = "조건4: 매물대 지지"
                                        
                                        first_buy_date = date
                                        buy_markers.append({'date': date, 'price': close, 'reason': "돌파 신규매수"})
                                        
                                elif position_weight > 0.0:
                                    is_sold = False
                                    sell_reason = ""
                                    
                                    if current_strategy == "Breakout":
                                        highest_since_entry = max(highest_since_entry, close)
                                        trailing_stop = highest_since_entry - (atr_mult * df['ATR'].iloc[i])
                                        
                                        if close < trailing_stop:
                                            is_sold = True
                                            sell_reason = "ATR 트레일링 스탑 이탈"
                                        elif close < ma20:
                                            is_sold = True
                                            sell_reason = "일봉 20일선 이탈"
                                            
                                    elif current_strategy == "BB_Bottom":
                                        if bb_step == 1: target_pct = 0.03
                                        elif bb_step == 2: target_pct = 0.01
                                        else: target_pct = 0.0
                                        
                                        target_price = avg_price * (1 + target_pct)
                                        
                                        if close >= target_price:
                                            is_sold = True
                                            sell_reason = f"BB {bb_step}차 목표 달성 (+{target_pct*100:.0f}%)"
                                        
                                        elif is_bb_touch and date.split()[0] != last_buy_date.split()[0] and bb_step < 3:
                                            # 분할매수 횟수 제한 및 당일 중복매수 방지 로직 (날짜 비교)
                                            bb_step += 1
                                            add_weight = 0.35 if bb_step == 2 else 0.175
                                            avg_price = ((avg_price * position_weight) + (close * add_weight)) / (position_weight + add_weight)
                                            position_weight += add_weight
                                            last_buy_date = date
                                            buy_markers.append({'date': date, 'price': low, 'reason': f"BB {bb_step}차 추가매수"})
                                    
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
                                        
                                        position_weight = 0.0
                                        avg_price = 0.0
                                        bb_step = 0
                                        current_strategy = None
                                        current_buy_reason = ""

                            log_df = pd.DataFrame(trade_log)
                            
                            if log_df.empty:
                                st.warning(f"해당 기간 내 설정된 매매 조건이 발생하지 않았습니다.")
                            else:
                                st.subheader(f"🎯 [{ticker}] {selected_tf} 타점 + 일봉 이평선 결과")
                                
                                win_trades = log_df[log_df['수익률(%)'] > 0]
                                win_rate = (len(win_trades) / len(log_df)) * 100
                                
                                cumulative_return = 1.0
                                for ret in log_df['수익률(%)']:
                                    cumulative_return *= (1 + ret / 100)
                                cumulative_return = (cumulative_return - 1) * 100

                                col1, col2, col3 = st.columns(3)
                                col1.metric("총 매매 횟수", f"{len(log_df)}회")
                                col2.metric("전체 승률", f"{win_rate:.2f}%")
                                col3.metric("전체 누적 수익률", f"{cumulative_return:.2f}%")
                                
                                st.divider()
                                
                                col_tbl1, col_tbl2 = st.columns(2)
                                with col_tbl1:
                                    st.subheader("🛠️ 매수 조건별 성과")
                                    cond_stats = log_df.groupby('매수조건')['수익률(%)'].agg(
                                        매매횟수='count',
                                        승률=lambda x: (x > 0).mean() * 100,
                                        누적수익률_합산='sum'
                                    ).reset_index().round(2)
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
                                
                                st.subheader(f"📊 매수/매도 타점 차트 ({selected_tf} 주가 + 일봉 20일선)")
                                
                                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                    vertical_spacing=0.03, subplot_titles=(f'{ticker} 주가 흐름 및 지표', '거래량'),
                                                    row_width=[0.2, 0.7])
                                
                                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='purple', width=1, dash='dot'), name='BB 하단'), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='red', width=1.5), name='진짜 일봉 20일선'), row=1, col=1)
                                
                                if buy_markers:
                                    fig.add_trace(go.Scatter(x=[m['date'] for m in buy_markers], y=[m['price'] for m in buy_markers], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='DarkSlateGrey')), name='Buy', text=[m['reason'] for m in buy_markers], hoverinfo='text'), row=1, col=1)
                                if sell_markers:
                                    fig.add_trace(go.Scatter(x=[m['date'] for m in sell_markers], y=[m['price'] for m in sell_markers], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='DarkSlateGrey')), name='Sell', text=[m['reason'] for m in sell_markers], hoverinfo='text'), row=1, col=1)
                                
                                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
                                fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)
