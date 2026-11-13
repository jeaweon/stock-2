"""Stock data loader with caching for hourly (1h) historical data.

yfinance Rate Limit 방어:
    1. 로컬 캐시(data/cache/{ticker}_1h_730d.csv) 최우선 사용 -> 네트워크 요청 0회
    2. 신규 다운로드는 지수 백오프 재시도 + 요청 간 time.sleep() 간격
    3. 대량 종목은 load_many_stock_data()로 배치 처리 (배치 사이 휴식)
"""
import os
import time
from datetime import datetime, timezone
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

    df = _standardize_ohlcv_df(df)

    # Cache locally
    try:
        df.to_csv(cache_file)
    except Exception:
        pass

    return df


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    """Deduplicate and uppercase ticker symbols."""
    syms: List[str] = []
    for t in tickers:
        sym = str(t).upper().strip()
        if sym and sym not in syms:
            syms.append(sym)
    return syms


def _standardize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance OHLCV columns, UTC DatetimeIndex, and sorting."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [col[0] for col in out.columns]

    col_mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = out.rename(columns={c: col_mapping[str(c).lower()] for c in out.columns if str(c).lower() in col_mapping})

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in out.columns:
            raise ValueError(f"데이터에 필수 열({col})이 누락되었습니다: {out.columns.tolist()}")

    out = out[required_cols].dropna(subset=["Close"]).copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "Datetime"
    out = out.sort_index()
    return out


def _extract_ticker_df_from_batch(
    raw_df: pd.DataFrame, sym: str, is_single_ticker: bool
) -> Optional[pd.DataFrame]:
    """Extract a single ticker's DataFrame from yf.download group_by='ticker' output."""
    if raw_df is None or raw_df.empty:
        return None
    if isinstance(raw_df.columns, pd.MultiIndex):
        lvl0 = raw_df.columns.get_level_values(0)
        lvl1 = raw_df.columns.get_level_values(1)
        if sym in lvl0:
            sub = raw_df[sym]
        elif sym in lvl1:
            sub = raw_df.xs(sym, axis=1, level=1)
        else:
            return None
    elif is_single_ticker:
        sub = raw_df
    else:
        return None

    try:
        std = _standardize_ohlcv_df(sub)
        return std if not std.empty else None
    except Exception:
        return None


def fetch_live_usd_krw_rate(fallback: float = 1400.0) -> Tuple[float, str]:
    """Fetch live USD/KRW exchange rate via Yahoo Finance (KRW=X).

    Returns:
        (fx_rate, timestamp_iso_utc)
    """
    now_str = datetime.now(timezone.utc).isoformat()
    try:
        fx = yf.Ticker("KRW=X")
        df = fx.history(period="5d", interval="1h")
        if df is None or df.empty:
            df = fx.history(period="5d", interval="1d")
        if df is not None and not df.empty and "Close" in df.columns:
            closes = df["Close"].dropna()
            if not closes.empty:
                rate = float(closes.iloc[-1])
                if 800.0 <= rate <= 3000.0:
                    ts = pd.to_datetime(closes.index[-1], utc=True).isoformat()
                    return round(rate, 2), ts
    except Exception:
        pass
    return float(fallback), now_str


def fetch_live_hourly_batch(
    tickers: Iterable[str],
    period: str = "90d",
    interval: str = "1h",
    chunk_size: int = 25,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Fetch fresh live 1h candle data directly from Yahoo Finance API (bypasses old cache).

    Uses chunked multi-ticker downloads for high speed across 100+ stocks, with
    single-ticker fallback retry for any symbol missed in a chunk.
    """
    syms = _normalize_tickers(tickers)
    total = len(syms)
    loaded: Dict[str, pd.DataFrame] = {}
    failed: Dict[str, str] = {}
    if total == 0:
        return loaded, failed

    step = max(1, chunk_size)
    processed = 0

    for c_start in range(0, total, step):
        chunk = syms[c_start:c_start + step]
        if c_start > 0:
            sleep_fn(0.4)

        if progress_callback:
            progress_callback(processed, total, f"{chunk[0]} 등 {len(chunk)}종목 API 수신 중...")

        raw_chunk = pd.DataFrame()
        try:
            raw_chunk = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            raw_chunk = pd.DataFrame()

        missed: List[str] = []
        is_single = len(chunk) == 1
        for sym in chunk:
            sub_df = _extract_ticker_df_from_batch(raw_chunk, sym, is_single)
            if sub_df is not None and not sub_df.empty:
                loaded[sym] = sub_df
            else:
                missed.append(sym)

        # Fallback retry individually for any symbols missed in the batch download
        for sym in missed:
            try:
                tk = yf.Ticker(sym)
                single_df = _fetch_with_retry(
                    lambda s=tk: s.history(period=period, interval=interval, auto_adjust=True),
                    max_retries=2,
                    sleep_fn=sleep_fn,
                )
                if single_df.empty:
                    single_df = _fetch_with_retry(
                        lambda s=tk: s.history(period="max", interval=interval, auto_adjust=True),
                        max_retries=1,
                        sleep_fn=sleep_fn,
                    )
                if not single_df.empty:
                    loaded[sym] = _standardize_ohlcv_df(single_df)
                else:
                    failed[sym] = "API에서 1시간봉 데이터를 반환하지 않았습니다."
            except Exception as e:
                failed[sym] = str(e)

        processed += len(chunk)
        if progress_callback:
            progress_callback(processed, total, f"완료 ({processed}/{total})")

    return loaded, failed


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
    syms = _normalize_tickers(tickers)

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
