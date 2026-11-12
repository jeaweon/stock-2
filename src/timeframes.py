"""1시간 봉 체제의 이평선(EMA) 기간 정의 및 용어 규약.

미국 정규장은 하루 6.5시간(09:30~16:00 ET)이므로 1시간 봉 데이터에서
'N일(Day) 거시 EMA'는 반드시 'N * 6.5 시간봉' span으로 환산해야 합니다.

용어 규약 (프롬프트 / 변수명 / 주석 공통):
    - "N일 거시 EMA"   (day 기준)    -> span = round(N * 6.5) 시간봉
        * 20일 거시 EMA  = 130시간봉 EMA  -> 변수명: ema_20day_130h
        * 50일 거시 EMA  = 325시간봉 EMA  -> 변수명: ema_50day_325h
        * 100일 거시 EMA = 650시간봉 EMA  -> 변수명: ema_100day_650h
    - "N시간봉 EMA"     (candle 기준) -> span = N 시간봉 (환산 없음)
        * 20시간봉 EMA  = 단기 20 EMA     -> 변수명: ema_20h
        * 200시간봉 EMA = 약 30.8거래일    -> 변수명: ema_200h
"""
import pandas as pd

HOURS_PER_TRADING_DAY: float = 6.5


def day_to_hour_bars(days: float) -> int:
    """N 거래일을 1시간 봉 개수로 환산 (예: 20일 -> 130봉)."""
    return int(round(days * HOURS_PER_TRADING_DAY))


# 표준 span 상수
SPAN_20DAY = day_to_hour_bars(20)    # 130
SPAN_50DAY = day_to_hour_bars(50)    # 325
SPAN_100DAY = day_to_hour_bars(100)  # 650
SPAN_20H = 20
SPAN_200H = 200


def ema_days(close: pd.Series, days: float) -> pd.Series:
    """'N일 거시 EMA' (N * 6.5 시간봉 span)."""
    return close.ewm(span=day_to_hour_bars(days), adjust=False).mean()


def ema_hours(close: pd.Series, hours: int) -> pd.Series:
    """'N시간봉 EMA' (환산 없이 N 시간봉 span)."""
    return close.ewm(span=int(hours), adjust=False).mean()


TERMINOLOGY_PROMPT = (
    "[이평선 용어 규약 - 1시간 봉 체제, 정규장 하루 6.5시간]\n"
    "- 'N일 거시 EMA'(Day 기준)는 반드시 span = N*6.5 시간봉으로 계산: "
    "20일 = ewm(span=130), 50일 = ewm(span=325), 100일 = ewm(span=650).\n"
    "- 'N시간봉 EMA'(Candle 기준)는 span = N 그대로: 단기 20시간봉 EMA = ewm(span=20), 200시간봉 EMA = ewm(span=200).\n"
    "- 변수명은 ema_20day_130h(일 기준) / ema_20h(시간봉 기준)처럼 단위를 접미사로 명시하고, "
    "'20 EMA'처럼 단위가 모호한 표현은 사용하지 마십시오.\n"
)
