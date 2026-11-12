"""Stock data loader with caching for hourly (1h) historical data.

yfinance Rate Limit 방어:
    1. 로컬 캐시(data/cache/{ticker}_1h_730d.csv) 최우선 사용 -> 네트워크 요청 0회
    2. 신규 다운로드는 지수 백오프 재시도 + 요청 간 time.sleep() 간격
    3. 대량 종목은 load_many_stock_data()로 배치 처리 (배치 사이 휴식)
"""
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/cache")

# Rate limit 방어 파라미터
REQUEST_INTERVAL_SEC: float = 1.0     # 신규 다운로드 요청 간 최소 간격
MAX_RETRIES: int = 3                  # 요청 실패/빈 응답 시 재시도 횟수
BACKOFF_BASE_SEC: float = 2.0         # 재시도 대기 = BACKOFF_BASE_SEC * 2**attempt
DEFAULT_BATCH_SIZE: int = 10          # 배치당 종목 수
BATCH_PAUSE_SEC: float = 5.0          # 배치 사이 휴식

_last_request_ts: float = 0.0


def _throttle(sleep_fn: Callable[[float], None] = time.sleep) -> None:
    """직전 네트워크 요청 이후 REQUEST_INTERVAL_SEC 이 지나도록 대기."""
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < REQUEST_INTERVAL_SEC:
        sleep_fn(REQUEST_INTERVAL_SEC - elapsed)
    _last_request_ts = time.time()


def _fetch_with_retry(
    fetch_fn: Callable[[], pd.DataFrame],
    max_retries: int = MAX_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """요청 간격 조절 + 지수 백오프 재시도. 모든 시도 실패 시 빈 DataFrame 반환."""
    for attempt in range(max_retries):
        _throttle(sleep_fn)
        try:
            df = fetch_fn()
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        if attempt < max_retries - 1:
            sleep_fn(BACKOFF_BASE_SEC * (2 ** attempt))
    return pd.DataFrame()


def _read_cache(cache_file: Path) -> Optional[pd.DataFrame]:
    if not cache_file.exists():
        return None
    try:
        df = pd.read_csv(cache_file, index_col=0)
        if not df.empty and "Close" in df.columns:
            df.index = pd.to_datetime(df.index, utc=True)
            return df
    except Exception:
        pass
    return None


def get_cache_path(ticker: str, period: str = "730d", interval: str = "1h") -> Path:
    safe_ticker = ticker.upper().strip()
    return CACHE_DIR / f"{safe_ticker}_{interval}_{period}.csv"


def load_stock_data(
    ticker: str,
    period: str = "730d",
    interval: str = "1h",
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load stock data from Yahoo Finance or local cache.

    Yahoo Finance supports up to 730 days for 1h interval.
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'NVDA', 'SPY')
        period: Period string (default '730d' = ~2 years)
        interval: Data interval ('1h')
        force_reload: If True, bypass cache and re-download.
    Returns:
        pd.DataFrame with DatetimeIndex and ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = get_cache_path(ticker, period, interval)

    # 1) 로컬 캐시 최우선 (네트워크 요청 없음)
    if not force_reload:
        cached = _read_cache(cache_file)
        if cached is not None:
            return cached

    # 2) 신규 다운로드: 요청 간격 + 지수 백오프 재시도
    ticker_sym = ticker.upper().strip()
    yf_ticker = yf.Ticker(ticker_sym)
    df = _fetch_with_retry(lambda: yf_ticker.history(period=period, interval=interval, auto_adjust=True))

    if df.empty:
        # 상장 기간이 730일보다 짧은 종목(ETHU 등) 대비
        df = _fetch_with_retry(
            lambda: yf_ticker.history(period="max", interval=interval, auto_adjust=True), max_retries=1
        )

    if df.empty:
        df = _fetch_with_retry(
            lambda: yf.download(tickers=ticker_sym, period=period, interval=interval, auto_adjust=True, progress=False),
            max_retries=1,
        )

    if df.empty:
        raise ValueError(f"'{ticker_sym}' 종목의 {interval} 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")

    # Flatten multi-level columns if present (common in recent yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Standardize column names
    col_mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns={c: col_mapping[c.lower()] for c in df.columns if c.lower() in col_mapping})

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"데이터에 필수 열({col})이 누락되었습니다: {df.columns.tolist()}")

    df = df[required_cols].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "Datetime"
    df.sort_index(inplace=True)

    # Cache locally
    try:
        df.to_csv(cache_file)
    except Exception:
        pass

    return df


def load_many_stock_data(
    tickers: Iterable[str],
    period: str = "730d",
    interval: str = "1h",
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_pause_sec: float = BATCH_PAUSE_SEC,
    force_reload: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """대량 종목 배치 로드. 캐시 적중 종목은 즉시 반환하고, 신규 다운로드 종목만 배치로 나눠
    배치 사이 batch_pause_sec 만큼 휴식하여 서버 차단(Rate Limit)을 방지합니다.

    Returns: (성공 {ticker: df}, 실패 {ticker: 에러 메시지})
    """
    syms: List[str] = []
    for t in tickers:
        sym = t.upper().strip()
        if sym and sym not in syms:
            syms.append(sym)

    loaded: Dict[str, pd.DataFrame] = {}
    failed: Dict[str, str] = {}
    to_download: List[str] = []

    for sym in syms:
        cached = None if force_reload else _read_cache(get_cache_path(sym, period, interval))
        if cached is not None:
            loaded[sym] = cached
        else:
            to_download.append(sym)

    for b_start in range(0, len(to_download), max(1, batch_size)):
        if b_start > 0:
            sleep_fn(batch_pause_sec)
        for sym in to_download[b_start:b_start + batch_size]:
            try:
                loaded[sym] = load_stock_data(sym, period=period, interval=interval, force_reload=True)
            except Exception as e:
                failed[sym] = str(e)

    return loaded, failed
