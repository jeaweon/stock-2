import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings

# yfinance 및 pandas 연산 시 발생하는 불필요한 경고문구 숨김
warnings.filterwarnings('ignore')

def calculate_volume_profile(df, window=120, bins=50):
    """지정된 기간 동안의 매물대(Point of Control) 계산"""
    poc_series = pd.Series(index=df.index, dtype=float)
    
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        # 가격대별 거래량 히스토그램 생성
        hist, bin_edges = np.histogram(window_df['Close'], bins=bins, weights=window_df['Volume'])
        max_vol_idx = np.argmax(hist)
        # 가장 거래량이 많은 가격대의 중간값(POC) 추출
        poc_series.iloc[i] = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
        
    return poc_series

def run_custom_backtest(ticker):
    print(f"\n📊 [{ticker}] 5년 주가 데이터 기반 백테스트 시작...")
    
    # 1. 데이터 수집 (최근 5년)
    df = yf.download(ticker, period="5y", progress=False)
    if df.empty:
        print("데이터를 불러오지 못했습니다. 티커(종목코드)를 정확히 입력했는지 확인해주세요.")
        return

    # 최신 yfinance 다중 인덱스 컬럼 구조 평탄화 처리
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, level=1, axis=1)
        else:
            df = df.droplevel(1, axis=1)
            
    # 데이터 결측치 제거
    df.dropna(inplace=True)

    # 2. 보조지표 계산
    print("지표 및 매물대(Volume Profile) 계산 중... (잠시만 기다려주세요)")
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Vol90_MA'] = df['Volume'].rolling(window=90).mean()
    
    # 볼린저 밴드
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    # 밴드폭 (수축 확인용)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] 
    
    # 매물대(POC) 계산
    df['POC'] = calculate_volume_profile(df)
    
    # 3. 시간 순서대로 매매 시뮬레이션 및 일지 기록
    position = 0
    buy_date, buy_price = None, 0
    trade_log = []
    
    # 지표 계산에 필요한 초기 기간(120일) 이후부터 백테스트 시작
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
        
        # --- 매수 조건 파악 ---
        # 1. 매물대 상향 돌파 & 5일선 > 20일선
        cond1 = (prev_price < poc) and (price > poc) and (ma5 > ma20)
        # 2. 거래량 90일 평균 1.5배 돌파 & 5일선 > 20일선
        cond2 = (vol > vol90 * 1.5) and (ma5 > ma20)
        # 3. VCP: 밴드폭 극도 수축(5% 이내 가정) 후 거래량 3배 폭발
        cond3 = (bb_width < 0.05) and (vol > vol90 * 3.0)
        # 4. 매물대 돌파 후 지지 (현재가가 매물대 2% 이내에 위치하며 양봉 마감)
        cond4 = (abs(price - poc) / poc < 0.02) and (price > open_price)
        
        is_buy_signal = cond1 or cond2 or cond3 or cond4
        
        # --- 매도 조건 파악 ---
        # 20일선이 5일선 위로 올라갈 때 (데드크로스 발생)
        is_sell_signal = ma20 > ma5
        
        # 매매 실행 로직
        if is_buy_signal and position == 0:
            position = 1
            buy_date = date
            buy_price = price
            
        elif is_sell_signal and position == 1:
            sell_price = price
            trade_return = (sell_price - buy_price) / buy_price * 100
            
            # 매매 일지 기록
            trade_log.append({
                '매수일': buy_date,
                '매수단가': round(buy_price, 2),
                '매도일': date,
                '매도단가': round(sell_price, 2),
                '수익률(%)': round(trade_return, 2)
            })
            # 포지션 초기화 (현금 보유)
            position = 0
            buy_date, buy_price = None, 0

    # 4. 매매일지 출력 및 결과 요약
    log_df = pd.DataFrame(trade_log)
    
    if log_df.empty:
        print("\n조건에 맞는 매매 내역이 발생하지 않았습니다.")
        return

    # 승률 계산
    win_trades = log_df[log_df['수익률(%)'] > 0]
    win_rate = (len(win_trades) / len(log_df)) * 100
    
    # 복리 누적 수익률 계산
    cumulative_return = 1.0
    for ret in log_df['수익률(%)']:
        cumulative_return *= (1 + ret / 100)
    cumulative_return = (cumulative_return - 1) * 100

    print("\n📝 [상세 매매 일지]")
    print(log_df.to_string(index=False))
    
    print("\n🎯 [백테스트 결과 요약]")
    print(f"총 매매 횟수 : {len(log_df)}회")
    print(f"승률 (Win Rate) : {win_rate:.2f}%")
    print(f"누적 수익률 : {cumulative_return:.2f}%")

if __name__ == "__main__":
    # 사용자로부터 티커 입력 받기
    target_ticker = input("분석할 주식의 티커를 입력하세요 (미국: AAPL, 한국 코스피: 005930.KS 등): ")
    run_custom_backtest(target_ticker.strip())
