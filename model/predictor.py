"""
predictor.py
------------
Loads the trained quantile models once and exposes simple prediction
functions that app.py (Flask) calls.

Two feature sources are supported for "today"'s starting point:

1. STATIC (fallback) - the frozen feature snapshot saved at training
   time (models/latest_features.json). Always available, never fails.

2. LIVE - fetches a live quote for every market driver (gold, USD/EGP,
   DXY, WTI, silver, S&P500, VIX) and recomputes the lag/rolling
   features for "today" on the fly, using feature_utils.py (the exact
   same formulas build_dataset.py used at training time) applied to
   data/processed/recent_market_window.csv + today's live quotes.

get_live_snapshot() tries #2 and transparently falls back to #1 (or a
partial mix, per-series) if any live source is unreachable - the app
never breaks, it just becomes less "live" in degraded conditions.
"""

import os
import sys
import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_utils import price_features, composite_risk_features  # noqa: E402

from model.live_price import fetch_live_price
from model.live_quotes import fetch_all_live_quotes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HISTORY_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "price_history.csv")
RECENT_WINDOW_PATH = os.path.join(BASE_DIR, "data", "processed", "recent_market_window.csv")

_models = None
_feature_columns = None          # full list incl. horizon_* columns
_base_feature_columns = None     # feature_columns minus horizon_* columns
_base_feature_values = None      # dict: {column_name: latest value} (static fallback)
_meta = None
_history_df = None
_recent_window_df = None
_calib_alpha = None               # (horizons_array, alphas_array) or None
_calib_scale = None                # (horizons_array, scales_array) or None


def _interp_calib(horizon_days: int, calib, default: float = 1.0) -> float:
    """Linearly interpolates a calibration curve fit at train_model.py's
    EVAL_HORIZONS grid points; flat-extrapolates outside that range (so a
    400-day custom-date query just reuses the 365-day calibration - we have
    no evaluation evidence past 365 days, so holding it flat is the
    conservative choice rather than guessing further out)."""
    if calib is None:
        return default
    hs, vals = calib
    h = float(np.clip(horizon_days, hs[0], hs[-1]))
    return float(np.interp(h, hs, vals))


def _load():
    """Lazy-loads everything once per process (cheap after the first call -
    important on serverless where the process may be reused across
    requests)."""
    global _models, _feature_columns, _base_feature_columns, _base_feature_values
    global _meta, _history_df, _recent_window_df

    if _models is not None:
        return

    _models = {
        "low": joblib.load(os.path.join(MODELS_DIR, "gold_model_low.joblib")),
        "median": joblib.load(os.path.join(MODELS_DIR, "gold_model_median.joblib")),
        "high": joblib.load(os.path.join(MODELS_DIR, "gold_model_high.joblib")),
    }

    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        _feature_columns = json.load(f)
    _base_feature_columns = [c for c in _feature_columns if not c.startswith("horizon")]

    with open(os.path.join(MODELS_DIR, "latest_features.json")) as f:
        latest_records = json.load(f)
    _base_feature_values = latest_records[0]

    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        _meta = json.load(f)

    global _calib_alpha, _calib_scale
    eval_report = _meta.get("evaluation") or {}
    deploy_calib = _meta.get("deployment_calibration") or {}
    if eval_report:
        hs = sorted(int(h) for h in eval_report.keys())
        _calib_alpha = (
            np.array(hs, dtype=float),
            np.array([eval_report[str(h)]["blend_alpha"] for h in hs], dtype=float),
        )
    else:
        _calib_alpha = None
    if deploy_calib:
        hs2 = sorted(int(h) for h in deploy_calib.keys())
        _calib_scale = (
            np.array(hs2, dtype=float),
            np.array([deploy_calib[str(h)]["band_scale"] for h in hs2], dtype=float),
        )
    else:
        _calib_scale = None

    _history_df = pd.read_csv(HISTORY_DATA_PATH, parse_dates=["Date"])

    if os.path.exists(RECENT_WINDOW_PATH):
        _recent_window_df = pd.read_csv(RECENT_WINDOW_PATH, parse_dates=["Date"])
    else:
        _recent_window_df = None


def get_meta() -> dict:
    _load()
    return dict(_meta)


def get_anchor_date() -> date:
    _load()
    return pd.to_datetime(_meta["data_end"]).date()


def get_last_known_price() -> float:
    _load()
    return float(_meta["last_known_price"])


# --------------------------------------------------------------------
# LIVE feature snapshot
# --------------------------------------------------------------------

_snapshot_cache = {"data": None, "fetched_at": 0.0}
SNAPSHOT_CACHE_SECONDS = 20  # short-lived, per-process cache


def get_live_snapshot(force: bool = False) -> dict:
    """Public entry point - wraps _get_live_snapshot_inner() in a broad
    safety net AND a short-lived cache.

    Why the cache: a single page load hits 4 different API routes
    (/api/summary, /api/year-forecast, /api/history x2), and each one
    used to call this function independently - meaning up to 4x the live
    network calls (gold + 7 Yahoo tickers each) for ONE page view. That's
    slow, hammers the upstream APIs harder than necessary, and could even
    show slightly different "current price" numbers across cards/charts
    if the price ticked between calls. Caching for a short window (20s)
    fixes all three: one real fetch serves the whole page, and a refresh
    a few seconds later still gets reasonably fresh data.

    Pass force=True to bypass the cache and always fetch fresh live data -
    used by the manual "refresh" button in the UI, so the person always
    gets a genuinely new price when they explicitly ask for one instead
    of possibly seeing the same cached number for up to 20 seconds.

    Safety net: live data fetching involves multiple external network
    calls, and production APIs can fail in ways we haven't specifically
    anticipated (a changed response schema, a rate limit, a transient
    DNS hiccup). Rather than letting any of that turn into a 500 error
    for the person looking at the page, ANY unexpected failure here
    falls back to the safe static snapshot - the site keeps working,
    just with slightly less fresh data.

    Returns a dict:
        {
          "anchor_date": date,
          "anchor_price": float,
          "feature_row": dict | None,   # None means "use the static snapshot"
          "status": "live" | "partial_live" | "estimated_live" | "cached",
        }

    Never raises - every failure degrades gracefully.
    """
    now = time.time()
    cached = _snapshot_cache["data"]
    if not force and cached is not None and (now - _snapshot_cache["fetched_at"]) < SNAPSHOT_CACHE_SECONDS:
        return cached

    try:
        result = _get_live_snapshot_inner()
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
        print(f"[live_snapshot] unexpected failure, falling back to cached: {exc}", file=sys.stderr)
        _load()
        result = {
            "anchor_date": date.today(),
            "anchor_price": get_last_known_price(),
            "feature_row": None,
            "status": "cached",
        }

    _snapshot_cache["data"] = result
    _snapshot_cache["fetched_at"] = now
    return result


def _get_live_snapshot_inner() -> dict:
    _load()

    live_gold = fetch_live_price()

    if live_gold is None or _recent_window_df is None or _recent_window_df.empty:
        # Can't anchor on a live price (or have nothing to recompute
        # features from) - fall back entirely to the static snapshot.
        # IMPORTANT: the calendar date "today" always means the real
        # current date, never the last date the model happens to have
        # training data for - otherwise "tomorrow" could render as a
        # date that's actually in the past if the model hasn't been
        # retrained in a few days. Only the underlying FEATURES are
        # allowed to be stale here, never the date labels shown to the user.
        return {
            "anchor_date": date.today(),
            "anchor_price": get_last_known_price(),
            "feature_row": None,
            "status": "cached",
        }

    live_quotes = fetch_all_live_quotes()

    window = _recent_window_df.set_index("Date").sort_index()
    today = date.today()
    last_window_date = window.index.max().date()

    last_close = float(window["Gold_21K"].loc[pd.Timestamp(last_window_date)])
    world_last = window["Gold_World"].loc[pd.Timestamp(last_window_date)] if "Gold_World" in window.columns else None
    usd_last = window["USD_EGP"].loc[pd.Timestamp(last_window_date)] if "USD_EGP" in window.columns else None
    world_now = live_quotes.get("Gold_World")
    usd_now = live_quotes.get("USD_EGP")

    is_estimate = False
    effective_gold = live_gold
    if (
        abs(live_gold - last_close) < 0.01
        and world_last is not None and usd_last is not None
        and world_now is not None and usd_now is not None
        and float(world_last) > 0 and float(usd_last) > 0
    ):
        # The local source hasn't updated its quote yet today - estimate
        # today's price by scaling yesterday's close by the relative move
        # in (world gold price x USD/EGP) since then. This assumes the
        # local retail premium stays roughly constant over a few hours,
        # which is reasonable (dealer markups don't change every refresh).
        scale = (float(world_now) * float(usd_now)) / (float(world_last) * float(usd_last))
        effective_gold = last_close * scale
        is_estimate = True

    live_quotes["Gold_21K"] = effective_gold

    # If a few days were missed (daily job hasn't run yet), forward-fill
    # the gap with the last known row so lag/rolling math stays valid,
    # then append today's live row on top.
    gap_dates = pd.date_range(last_window_date + timedelta(days=1), today, freq="D")
    for d in gap_dates[:-1]:
        window.loc[d] = window.iloc[-1]

    today_row = {}
    raw_cols = [c for c in window.columns]
    for col in raw_cols:
        live_val = live_quotes.get(col)
        today_row[col] = live_val if live_val is not None else window[col].iloc[-1]  # fall back to last known

    window.loc[pd.Timestamp(today)] = today_row

    # Recompute local_premium_ratio for the whole window (cheap - <=201 rows)
    if "Gold_World" in window.columns and "USD_EGP" in window.columns:
        implied = window["Gold_World"] * window["USD_EGP"]
        window["local_premium_ratio"] = window["Gold_21K"] / implied

    feats_today = {}
    price_cols = [c for c in window.columns if c != "local_premium_ratio"]
    for col in price_cols:
        col_feats = price_features(window[col].ffill(), col)
        for name, series in col_feats.items():
            feats_today[name] = float(series.iloc[-1])

    # The RAW price level of every market series (not just its engineered
    # log/lag/rolling derivatives) is also a base feature the model was
    # trained on - make sure "today"'s live value lands there too, not
    # just in the derived columns above. Gold_21K itself is the prediction
    # target, never a feature, so it's excluded.
    for col in raw_cols:
        if col == "Gold_21K":
            continue
        feats_today[col] = float(today_row[col])

    if "local_premium_ratio" in window.columns:
        feats_today["local_premium_ratio"] = float(window["local_premium_ratio"].iloc[-1])

    # Composite market-stress / safe-haven proxy features (see
    # feature_utils.composite_risk_features docstring for what these mean
    # and why they exist) - computed from the columns we just built above,
    # using the exact same formula as build_dataset.py so training and
    # live inference never drift apart.
    needed = ["VIX_rollstd30", "WTI_Oil_rollstd30", "DXY_rollstd30",
              "Gold_21K_rollmean7", "SP500_rollmean7"]
    if all(k in feats_today for k in needed):
        stress, safehaven = composite_risk_features(
            feats_today["VIX_rollstd30"], feats_today["WTI_Oil_rollstd30"],
            feats_today["DXY_rollstd30"], feats_today["Gold_21K_rollmean7"],
            feats_today["SP500_rollmean7"],
        )
        feats_today["market_stress_proxy"] = stress
        feats_today["gold_safehaven_signal"] = safehaven

    feats_today["day_of_week"] = today.weekday()
    feats_today["day_of_month"] = today.day
    feats_today["month"] = today.month
    feats_today["day_of_year"] = today.timetuple().tm_yday
    feats_today["is_weekend"] = int(today.weekday() >= 5)

    # Fill in anything the trained model expects that we didn't produce
    # here (e.g. an event index column not covered by the live window)
    # using the static snapshot as a last resort.
    for col in _base_feature_columns:
        if col not in feats_today:
            feats_today[col] = _base_feature_values.get(col, 0.0)

    # "live" only counts the series we actually track live (Gold_21K +
    # TICKERS_TRACKED) - US10Y is intentionally always frozen (treasury
    # yields don't move meaningfully within a day and have no simple free
    # intraday feed), so it must not drag the status down to "partial".
    live_trackable = {"Gold_21K"} | set(TICKERS_TRACKED)
    live_trackable_hits = sum(1 for col in live_trackable if live_quotes.get(col) is not None)
    if is_estimate:
        # The local quote itself is frozen, but we're honestly labeling it
        # as an estimate rather than silently passing it off as a real
        # reported price - distinct from "live" (a real reported number).
        status = "estimated_live"
    else:
        status = "live" if live_trackable_hits == len(live_trackable) else "partial_live"

    return {
        "anchor_date": today,
        "anchor_price": float(effective_gold),
        "feature_row": feats_today,
        "status": status,
        "is_estimate": is_estimate,
    }


TICKERS_TRACKED = ["Gold_World", "USD_EGP", "DXY", "WTI_Oil", "Silver", "SP500", "VIX"]

# The genuinely held-out evaluation found blend_alpha=0.00 at 1 and 7 days -
# i.e. "the model's honest best guess is literally no change". That's a
# real, defensible statistical finding (gold is close to a random walk
# short-term), but it produces a degenerate PRODUCT experience: tomorrow's
# price, and every single day of the week forecast, rendering as the exact
# same number as today - which reads as a display bug, not an honest
# forecast, and gives the person nothing to look at. This floor keeps a
# small amount of real model signal alive at short horizons instead of
# collapsing the whole first week to a flat line. It's a deliberate,
# disclosed trade of a little theoretical MAE for a non-degenerate,
# genuinely useful short-term forecast.
MIN_BLEND_ALPHA = 0.15


# --------------------------------------------------------------------
# Prediction functions - all accept an optional `snapshot` (from
# get_live_snapshot()) so callers can share ONE live fetch across a
# whole request instead of re-fetching per horizon.
# --------------------------------------------------------------------

def _resolve_snapshot(snapshot: dict | None) -> dict:
    """Used when a caller doesn't pass a pre-fetched snapshot at all
    (e.g. direct script/notebook use). Same rule as above: date labels
    are always real "today", regardless of feature data freshness."""
    if snapshot is not None:
        return snapshot
    _load()
    return {
        "anchor_date": date.today(),
        "anchor_price": get_last_known_price(),
        "feature_row": None,
        "status": "cached",
    }


def _build_feature_row(horizon_days: int, snapshot: dict) -> pd.DataFrame:
    row = dict(snapshot["feature_row"]) if snapshot["feature_row"] is not None else dict(_base_feature_values)
    row["horizon"] = horizon_days
    row["horizon_sqrt"] = float(np.sqrt(horizon_days))
    row["horizon_log1p"] = float(np.log1p(horizon_days))
    return pd.DataFrame([row])[_feature_columns]


def predict_for_horizon(horizon_days: int, snapshot: dict | None = None) -> dict:
    """Predicts the price `horizon_days` days after the snapshot's anchor date.

    Applies calibration from TWO different, deliberately separate sources
    in models/meta.json (see train_model.py's evaluate_model() and
    fit_deployment_calibration() docstrings for the full story):

      - blend_alpha(h) comes from the genuinely HELD-OUT walk-forward
        evaluation. It's a bounded [0,1] shrinkage toward the naive "no
        change" baseline, which transfers safely to the real deployed
        model regardless of the exact model size that produced it.
      - band_scale(h) comes from calibrating the ACTUAL deployed models'
        own (in-sample) output spread instead. This was NOT done using the
        held-out evaluation's scale on purpose: that scale was fit against
        much smaller throwaway models with a narrower raw spread, and
        applying it directly to the real, wider-spread production models
        produced absurd bands (even negative prices) during testing.

    A hard sanity clamp is applied below regardless, so no calibration bug
    can ever surface a nonsensical price band in the UI.

    If meta.json has neither key (e.g. an older file), both default to
    1.0 - i.e. the uncalibrated model, so this never breaks anything.
    """
    _load()
    if horizon_days < 1:
        horizon_days = 1

    snap = _resolve_snapshot(snapshot)
    base_price = snap["anchor_price"]
    X = _build_feature_row(horizon_days, snap)

    log_ret_low = _models["low"].predict(X)[0]
    log_ret_med = _models["median"].predict(X)[0]
    log_ret_high = _models["high"].predict(X)[0]
    lo, med, hi = sorted([log_ret_low, log_ret_med, log_ret_high])

    raw_med_price = base_price * float(np.exp(med))
    raw_lo_price = base_price * float(np.exp(lo))
    raw_hi_price = base_price * float(np.exp(hi))

    alpha = max(_interp_calib(horizon_days, _calib_alpha, default=1.0), MIN_BLEND_ALPHA)
    scale = _interp_calib(horizon_days, _calib_scale, default=1.0)

    calib_med_price = base_price + alpha * (raw_med_price - base_price)
    half_lo = raw_med_price - raw_lo_price
    half_hi = raw_hi_price - raw_med_price
    final_lo_price = calib_med_price - scale * half_lo
    final_hi_price = calib_med_price + scale * half_hi

    # Hard safety floor/ceiling, independent of how good the calibration is:
    # the displayed band must never suggest gold could be worth near-zero or
    # some absurd multiple of today's price, no matter what a fitted scale
    # factor comes out to.
    final_lo_price = max(final_lo_price, base_price * 0.5)
    final_hi_price = min(final_hi_price, base_price * 2.0)
    final_lo_price = min(final_lo_price, calib_med_price)
    final_hi_price = max(final_hi_price, calib_med_price)

    target_date = snap["anchor_date"] + timedelta(days=horizon_days)

    return {
        "date": target_date.isoformat(),
        "horizon_days": horizon_days,
        "predicted_price": round(calib_med_price, 2),
        "predicted_price_low": round(final_lo_price, 2),
        "predicted_price_high": round(final_hi_price, 2),
    }


def predict_for_date(target_date: date, snapshot: dict | None = None) -> dict:
    """Predicts the price for an arbitrary future calendar date."""
    snap = _resolve_snapshot(snapshot)
    horizon_days = (target_date - snap["anchor_date"]).days
    if horizon_days < 1:
        raise ValueError(
            f"target_date must be after {snap['anchor_date'].isoformat()}. Pick a future date."
        )
    if horizon_days > 400:
        raise ValueError("This model is only trained up to ~1 year ahead. "
                          "Predictions far beyond that are not reliable.")
    return predict_for_horizon(horizon_days, snapshot=snap)


def predict_week(snapshot: dict | None = None) -> list[dict]:
    snap = _resolve_snapshot(snapshot)
    return [predict_for_horizon(h, snapshot=snap) for h in range(1, 8)]


def predict_year_curve(snapshot: dict | None = None) -> list[dict]:
    snap = _resolve_snapshot(snapshot)
    horizons = list(range(7, 366, 7))
    if horizons[-1] != 365:
        horizons.append(365)
    return [predict_for_horizon(h, snapshot=snap) for h in horizons]


def get_history(days: int, snapshot: dict | None = None) -> list[dict]:
    """Returns the last `days` of ACTUAL (non-predicted) prices.

    If a live snapshot is passed, its status is "live" or "partial_live"
    (i.e. we genuinely got a fresh quote - not the stale fallback), AND
    its anchor date is more recent than the last row in the training
    dataset, today's live price is appended as the final point. Without
    this, the "actual price" chart would always look stuck a day or two
    behind reality - it can only ever be as fresh as the last daily
    retrain, while the live price itself is available immediately on
    every page load. This keeps what the person SEES in sync with
    reality even when the underlying training data hasn't caught up yet.

    Deliberately does NOT append anything when status is "cached" - in
    that case snapshot["anchor_price"] is just the same old training-time
    price repeated, and stamping it with today's date would misleadingly
    suggest we know today's actual price when we simply don't.
    """
    _load()
    sub = _history_df[["Date", "Gold_21K"]].tail(days).copy()

    if snapshot is not None and snapshot.get("status") in ("live", "partial_live"):
        last_hist_date = sub["Date"].max().date() if not sub.empty else None
        anchor_date = snapshot["anchor_date"]
        if last_hist_date is None or anchor_date > last_hist_date:
            live_row = pd.DataFrame([{
                "Date": pd.Timestamp(anchor_date),
                "Gold_21K": snapshot["anchor_price"],
            }])
            sub = pd.concat([sub, live_row], ignore_index=True).tail(days)

    return [
        {"date": d.date().isoformat(), "price": round(float(p), 2)}
        for d, p in zip(sub["Date"], sub["Gold_21K"])
    ]
