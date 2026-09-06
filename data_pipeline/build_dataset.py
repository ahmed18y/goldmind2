"""
build_dataset.py
-----------------
Merges every raw source CSV into one clean, feature-engineered daily
time series table that the model can train on.

INPUT  (read-only, never modified): data/raw/<CATEGORY>/*.csv
OUTPUT: data/processed/merged_dataset.csv

Design notes (read this before touching the feature list):
- The TARGET series (Egypt 21K gold price) is a retail price feed and
  has an observation for every single calendar day, including
  weekends. All the market series (Yahoo/FRED) only have observations
  on trading days. So the master calendar is built from the TARGET's
  date range, and every other series is reindexed onto that calendar
  and forward-filled (the last known trading price "carries over"
  through the weekend/holiday, which is standard practice for mixing
  a continuous retail feed with market data).
- Every price-like column gets: log price, log return, N-day lag
  returns, rolling mean/std of returns, and momentum vs N days ago.
  This is what lets a plain gradient-boosting model act like a time
  series model without needing recurrent architectures.
- If data/raw/06_EVENTS exists (EPU / GPR files produced by the
  updated update_gold_project.py) they are merged in automatically.
  Their absence does not break the pipeline - the model just trains
  without them.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feature_utils import price_features, composite_risk_features  # noqa: E402  (shared with model/predictor.py)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "merged_dataset.csv")
HISTORY_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "price_history.csv")
RECENT_WINDOW_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "recent_market_window.csv")

# How many trailing raw-price days to ship to the deployed app so it can
# recompute "today"'s features live (needs enough days to cover the
# longest lag/rolling window used in feature_utils.py, i.e. 90+60=150,
# plus a safety margin).
RECENT_WINDOW_DAYS = 200


def _read_price_csv(path: str, value_col: str = "Close") -> pd.Series:
    """Read a Yahoo/FRED-style CSV and return a single Date-indexed Series."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    date_col = "Date" if "Date" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if value_col not in df.columns:
        # FRED single-value files (e.g. DGS10) -> take the only non-date column
        candidates = [c for c in df.columns if c != date_col]
        value_col = candidates[0]

    series = pd.to_numeric(df[value_col], errors="coerce")
    series.index = df[date_col]
    series = series[~series.index.duplicated(keep="last")]
    return series.rename(value_col)


def _price_features(df: pd.DataFrame, col: str) -> dict:
    """Thin wrapper kept for readability at the call site below - the
    actual formulas live in feature_utils.py so build_dataset.py (offline)
    and predictor.py (live, per-request) can never drift out of sync."""
    return price_features(df[col].ffill(), col)


def main():
    print("=" * 70)
    print("BUILDING MERGED FEATURE DATASET")
    print("=" * 70)

    # ---- 1. Target series defines the master calendar -----------------
    target = _read_price_csv(os.path.join(RAW_DIR, "01_TARGET", "GoldRate24_Egypt_21K.csv"), "Gold_21K")
    calendar = pd.date_range(target.index.min(), target.index.max(), freq="D")

    df = pd.DataFrame(index=calendar)
    df.index.name = "Date"
    df["Gold_21K"] = target.reindex(calendar)

    # Target should not be forward filled blindly (it's what we predict),
    # but a handful of missing days (source hiccups) are safe to ffill.
    df["Gold_21K"] = df["Gold_21K"].ffill()

    # ---- 2. Market series (forward-filled onto the daily calendar) ----
    sources = {
        "Gold_World":  ("02_CORE", "Gold_World.csv", "Close"),
        "USD_EGP":     ("02_CORE", "USD_EGP.csv", "Close"),
        "DXY":         ("03_STRONG", "US_Dollar_Index_DXY.csv", "Close"),
        "WTI_Oil":     ("03_STRONG", "WTI_Oil.csv", "Close"),
        "US10Y":       ("03_STRONG", "US_10Y_Treasury_Yield.csv", "DGS10"),
        "Silver":      ("04_EXPERIMENTAL", "Silver.csv", "Close"),
        "SP500":       ("04_EXPERIMENTAL", "SP500.csv", "Close"),
        "VIX":         ("04_EXPERIMENTAL", "VIX.csv", "Close"),
    }

    for col_name, (folder, fname, value_col) in sources.items():
        path = os.path.join(RAW_DIR, folder, fname)
        if not os.path.exists(path):
            print(f"[SKIP] missing source file: {path}")
            continue
        s = _read_price_csv(path, value_col)
        df[col_name] = s.reindex(calendar).ffill()
        print(f"[OK] merged {col_name} ({s.dropna().shape[0]} raw obs)")

    # ---- 2b. Optional event/risk indices (06_EVENTS), if present -------
    events_dir = os.path.join(RAW_DIR, "06_EVENTS")
    event_cols = []
    if os.path.isdir(events_dir):
        for fname in os.listdir(events_dir):
            if not fname.lower().endswith(".csv"):
                continue
            path = os.path.join(events_dir, fname)
            try:
                s = _read_price_csv(path)
                col_name = os.path.splitext(fname)[0]
                df[col_name] = s.reindex(calendar).ffill()
                event_cols.append(col_name)
                print(f"[OK] merged event feature {col_name}")
            except Exception as exc:
                print(f"[WARN] could not merge {fname}: {exc}")

    # ---- 3. USD/EGP-implied fair value gap ------------------------------
    # Gold_21K should roughly track Gold_World * USD_EGP (unit-adjusted).
    # The gap between the actual local price and this implied value is a
    # genuinely informative feature (captures local premium/scarcity).
    if "Gold_World" in df.columns and "USD_EGP" in df.columns:
        implied = df["Gold_World"] * df["USD_EGP"]
        df["local_premium_ratio"] = df["Gold_21K"] / implied

    # ---- 4. Feature engineering per price column -------------------------
    price_cols = [c for c in df.columns if c not in ("local_premium_ratio",)]
    all_new_feats = {}
    for col in price_cols:
        all_new_feats.update(_price_features(df, col))

    # ---- 4b. Composite risk/trend proxy features --------------------------
    # Stand-ins for "political/economic risk" until the real news-based
    # GPR/EPU indices (see update_gold_project.py's 06_EVENTS download
    # functions) are available - see feature_utils.composite_risk_features
    # for exactly what these mean and why they're a reasonable proxy.
    all_new_feats["market_stress_proxy"], all_new_feats["gold_safehaven_signal"] = composite_risk_features(
        all_new_feats["VIX_rollstd30"], all_new_feats["WTI_Oil_rollstd30"], all_new_feats["DXY_rollstd30"],
        all_new_feats["Gold_21K_rollmean7"], all_new_feats["SP500_rollmean7"],
    )

    # ---- 5. Calendar features --------------------------------------------
    all_new_feats["day_of_week"] = df.index.dayofweek
    all_new_feats["day_of_month"] = df.index.day
    all_new_feats["month"] = df.index.month
    all_new_feats["day_of_year"] = df.index.dayofyear
    all_new_feats["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    df = pd.concat([df, pd.DataFrame(all_new_feats, index=df.index)], axis=1)
    df = df.reset_index()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Slim companion file: just Date + actual Gold_21K price. This is the
    # only thing the deployed web app needs at runtime for the "actual
    # history" charts, so shipping this ~150KB file instead of the full
    # ~20MB feature dataset keeps the Vercel deployment lean and fast.
    df[["Date", "Gold_21K"]].to_csv(HISTORY_OUT_PATH, index=False)

    # Slim companion file #2: the trailing RAW (pre-feature-engineering)
    # prices for every market series. This is what lets the deployed app
    # recompute "today"'s features live, by swapping in a freshly-fetched
    # live quote for the last row and re-running feature_utils on just
    # this small window - instead of being stuck with whatever the last
    # daily batch job happened to see.
    raw_cols = ["Date", "Gold_21K"] + list(sources.keys()) + event_cols
    raw_cols = [c for c in raw_cols if c in df.columns]
    df[raw_cols].tail(RECENT_WINDOW_DAYS).to_csv(RECENT_WINDOW_OUT_PATH, index=False)

    print("-" * 70)
    print(f"[DONE] Saved merged dataset -> {OUT_PATH}")
    print(f"[DONE] Saved slim price history -> {HISTORY_OUT_PATH}")
    print(f"[DONE] Saved recent market window ({RECENT_WINDOW_DAYS}d) -> {RECENT_WINDOW_OUT_PATH}")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]:,}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")


if __name__ == "__main__":
    main()
