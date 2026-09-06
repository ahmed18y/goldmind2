"""
live_quotes.py
---------------
Fetches a LIVE quote for every market series the model uses as a
feature (not just the Egypt gold price - see live_price.py for that
one specifically). This is what lets the deployed app recompute
"today"'s features from the real current market state instead of
whatever the last daily batch job saw.

Uses Yahoo Finance's public chart endpoint (the same data source
yfinance itself wraps) with a plain `requests` call, so we don't need
the full yfinance dependency (and its slower startup) in the request
path of a serverless function.

Every ticker is fetched independently and failures are isolated - if
DXY's endpoint is briefly down, we still get everything else, and the
one missing value just falls back to its last known value from
recent_market_window.csv (see predictor.py).
"""

from concurrent.futures import ThreadPoolExecutor

import requests

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
}

# Maps our internal feature column name -> Yahoo ticker symbol.
# (US10Y / DGS10 has no simple free intraday Yahoo ticker and doesn't
# move meaningfully within a day anyway, so it intentionally stays on
# the daily-refresh path rather than live.)
TICKERS = {
    "Gold_World": "GC=F",
    "USD_EGP": "EGP=X",
    "DXY": "DX-Y.NYB",
    "WTI_Oil": "CL=F",
    "Silver": "SI=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
}


def fetch_live_quote(ticker: str, timeout: float = 4.0) -> float | None:
    """Returns the latest live price for one Yahoo ticker, or None on failure."""
    try:
        resp = requests.get(
            YAHOO_CHART_URL.format(ticker=ticker),
            headers=HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        price = result["meta"].get("regularMarketPrice")
        return float(price) if price is not None else None
    except Exception:
        return None


def fetch_all_live_quotes() -> dict:
    """Returns {feature_col_name: live_price_or_None} for every tracked series.

    Runs all 7 Yahoo requests concurrently instead of one after another -
    these are pure I/O waits (network round-trips), so there's no reason to
    pay for them sequentially. Before this, a worst case where every ticker
    timed out slowly could take up to ~7 * 4s = 28s in a single request
    (stacked on top of the gold-price fetch itself); now the worst case is
    just ~4s total since they all wait on the network at the same time.
    """
    with ThreadPoolExecutor(max_workers=len(TICKERS)) as pool:
        futures = {col: pool.submit(fetch_live_quote, ticker) for col, ticker in TICKERS.items()}
        return {col: fut.result() for col, fut in futures.items()}
