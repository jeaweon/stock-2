"""Curated Stock Universes for Real-Time 1-Hour Live Paper Trading Scans."""
from typing import Dict, List


# 130+ Liquidity-Verified US Stocks & ETFs (S&P500 / NASDAQ100 / High-Growth Themes)
LIVE_SCAN_UNIVERSE_130: List[str] = [
    # Mega Tech & Software / Cloud / AI
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "PLTR", "CRM", "ORCL",
    "ADBE", "INTU", "NOW", "IBM", "CSCO", "FTNT", "DDOG", "NET", "SNOW", "PANW",
    "CRWD", "ZS", "MDB", "WDAY", "APP", "RBLX", "SPOT", "UBER", "ABNB", "DASH",
    # Semiconductors & Hardware / Data Center Infrastructure
    "AMD", "AVGO", "TSM", "MU", "QCOM", "INTC", "ARM", "SMCI", "MRVL", "LRCX",
    "AMAT", "ADI", "NXPI", "MCHP", "ON", "STM", "TER", "WDC", "STX", "SWKS",
    "QRVO", "ANET", "DELL", "HPQ", "VRT", "ALAB", "SOXL",
    # Nuclear / Clean Energy / Space / Quantum / High-Beta Growth
    "OKLO", "SMR", "NNE", "BE", "CEG", "TEM", "IONQ", "RGTI", "QBTS", "RKLB",
    "ASTS", "LUNR", "JOBY", "ACHR", "FSLR", "ENPH", "CCJ", "UEC", "NXE",
    # Fintech / Crypto Proxies & High-Momentum Financials
    "HOOD", "SOFI", "COIN", "MARA", "RIOT", "CLSK", "MSTR", "BITO", "PYPL", "UPST",
    "AFRM", "TOST", "NU", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "V", "MA",
    # Healthcare / Biotech & Consumer Growth
    "MRNA", "LLY", "PFE", "JNJ", "ABBV", "GILD", "BMY", "MRK", "CVS", "BSX",
    "EW", "DXCM", "PODD", "NKE", "SBUX", "TGT", "WMT", "DIS", "GM", "F",
    "RIVN", "LCID", "CCL", "RCL", "DAL", "UAL", "LUV", "GRAB", "SE", "BABA", "PDD",
    # Industrials / Defense / Energy / Materials & Core ETFs
    "GE", "BA", "LMT", "RTX", "CAT", "DE", "ETN", "EMR", "PWR", "XOM",
    "CVX", "OXY", "SLB", "HAL", "KMI", "FCX", "NEM", "AA", "CLF",
    "SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "TLT", "GLD", "SLV", "EWY"
]


# Under ~$350 (~500,000 KRW) friendly liquid stocks for 20-slot integer-share portfolios
AFFORDABLE_20SLOT_UNIVERSE: List[str] = [
    "NVDA", "PLTR", "AMD", "MU", "INTC", "ARM", "SMCI", "MRVL", "WDC", "STX",
    "ON", "STM", "TER", "SWKS", "QRVO", "HPQ", "DELL", "VRT", "ALAB", "SOXL",
    "OKLO", "SMR", "NNE", "BE", "TEM", "IONQ", "RGTI", "QBTS", "RKLB", "ASTS",
    "LUNR", "JOBY", "ACHR", "FSLR", "ENPH", "CCJ", "UEC", "NXE", "DDOG", "NET",
    "SNOW", "ZS", "MDB", "RBLX", "UBER", "ABNB", "CRM", "ORCL", "FTNT", "CSCO",
    "HOOD", "SOFI", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PYPL", "UPST", "AFRM",
    "TOST", "NU", "BAC", "WFC", "C", "SCHW", "MRNA", "PFE", "BMY", "GILD",
    "CVS", "BSX", "EW", "DXCM", "NKE", "SBUX", "TGT", "WMT", "DIS", "GM",
    "F", "RIVN", "LCID", "CCL", "DAL", "UAL", "LUV", "GRAB", "SE", "BABA",
    "PDD", "XOM", "CVX", "OXY", "SLB", "HAL", "KMI", "FCX", "NEM", "AA",
    "CLF", "TQQQ", "SQQQ", "IWM", "TLT", "SLV", "EWY"
]


HIGH_GROWTH_THEME_UNIVERSE: List[str] = [
    "NVDA", "PLTR", "APP", "ALAB", "VRT", "OKLO", "SMR", "NNE", "BE", "TEM",
    "IONQ", "RGTI", "QBTS", "RKLB", "ASTS", "LUNR", "JOBY", "ACHR", "HOOD", "SOFI",
    "COIN", "MSTR", "MARA", "CLSK", "MU", "AMD", "AVGO", "ARM", "MRVL", "LRCX",
    "AMAT", "TSLA", "SNOW", "NET", "CRWD", "DDOG", "RBLX", "TOST", "SOXL", "TQQQ"
]


def _dedupe(tickers: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in tickers:
        sym = t.upper().strip()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


SCAN_UNIVERSES: Dict[str, List[str]] = {
    "🌐 전체 대량 API 스캔 풀 (140+ 종목: S&P/나스닥/고성장주 통합)": _dedupe(LIVE_SCAN_UNIVERSE_130),
    "💵 20분할 정수매수 최적화 풀 (100+ 종목: 1주 50만원 이하 주도주 중심)": _dedupe(AFFORDABLE_20SLOT_UNIVERSE),
    "🚀 AI · 반도체 · 원전 · 우주항공 고성장 테마 (40종목)": _dedupe(HIGH_GROWTH_THEME_UNIVERSE),
}
