"""
live_price.py
--------------
Fetches the current Egypt 21K gold price live, so the app is always
"synced" with the real market instead of only showing the last row of
the training dataset.

Uses the same GoldRate24 (elomla.com) endpoint as the user's own
update_gold_project.py, requesting just the most recent observation.
If the request fails for any reason (offline dev environment, API
downtime, rate limiting) we fall back to the last known price from the
trained dataset so the app never crashes - it just shows a small
"(latest cached price)" note instead of a live one.
"""

import requests

GOLDRATE24_URL = "https://www.elomla.com/api/historical/lastdays"

GOLDRATE24_PARAMS = {
    "base": "XAU",
    "to": "EGP",
    "days": 2,
    "premium": "true",
    "format": "highcharts",
    "factor": 0.02813125,
}

GOLDRATE24_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def fetch_live_price(timeout: float = 6.0) -> float | None:
    """Returns the latest live Gold_21K price in EGP, or None on failure."""
    try:
        resp = requests.get(
            GOLDRATE24_URL,
            params=GOLDRATE24_PARAMS,
            headers=GOLDRATE24_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            return None

        rates = data["data"]["rates"]
        if not rates:
            return None

        # rates entries look like [timestamp_ms, value] - take the last one.
        last_value = rates[-1][1]
        return float(last_value)

    except Exception:
        return None
