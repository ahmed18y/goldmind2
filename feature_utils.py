"""
feature_utils.py
-----------------
The exact log-price / log-return / lag / rolling-stat formulas used to
turn a raw price series into model features.

This lives in its own tiny module and is imported by BOTH:
  - data_pipeline/build_dataset.py   (offline, builds the training set)
  - model/predictor.py               (online, recomputes "today"'s
                                       features live, at request time)

Keeping the formulas in exactly one place is what makes the live
request-time features and the training-time features consistent -
if you ever change how a lag/rolling feature is computed, you only
change it here and both pipelines stay in sync automatically.
"""

import numpy as np
import pandas as pd

LAG_DAYS = [1, 2, 3, 5, 7, 14, 21, 30, 60, 90]
ROLL_WINDOWS = [7, 14, 30, 60]


def price_features(price: pd.Series, col: str) -> dict:
    """Builds log price / log return / lags / rolling stats for one raw
    price series (indexed by date, ascending). Returns a dict of
    {feature_name: Series} - the caller decides how to combine them.

    `price` must already be forward-filled (no gaps) before calling this.
    """
    # Clip to a small positive floor: WTI crude famously went negative for
    # one day in April 2020, which would otherwise break log().
    log_price = np.log(price.clip(lower=0.01))
    log_ret = log_price.diff()

    feats = {
        f"{col}_log": log_price,
        f"{col}_ret1d": log_ret,
    }

    for lag in LAG_DAYS:
        feats[f"{col}_lag{lag}_ret"] = log_price - log_price.shift(lag)

    for win in ROLL_WINDOWS:
        feats[f"{col}_rollmean{win}"] = log_ret.rolling(win).mean()
        feats[f"{col}_rollstd{win}"] = log_ret.rolling(win).std()

    return feats


def composite_risk_features(vix_rollstd30, wti_rollstd30, dxy_rollstd30,
                             gold_rollmean7, sp500_rollmean7):
    """Two proxy features standing in for "political/economic risk" until
    a real news-based index (GPR/EPU - see update_gold_project.py) is
    available in the training data:

    - market_stress_proxy: sum of the 30-day realized volatility (already
      computed by price_features, just combined here) of VIX, oil (WTI),
      and the dollar (DXY). Political and economic shocks - wars,
      sanctions, surprise rate decisions, elections - show up almost
      immediately as a simultaneous spike in volatility ACROSS these
      unrelated markets, which is exactly why cross-asset volatility is
      the standard practitioner's substitute when a dedicated news-based
      risk index isn't at hand.

    - gold_safehaven_signal: the gap between gold's own recent trend
      (Gold_21K_rollmean7, a log-return average) and the S&P 500's
      (SP500_rollmean7). A positive value means gold has been
      outperforming stocks lately - the classic "flight to safety"
      pattern that shows up during geopolitical or economic turmoil.

    Works transparently whether the inputs are pandas Series (offline,
    build_dataset.py - the whole historical column at once) or plain
    Python floats (live, predictor.py - a single "today" value) since
    it's just arithmetic either way. This is why the formula lives here
    instead of being duplicated in both places.
    """
    market_stress_proxy = vix_rollstd30 + wti_rollstd30 + dxy_rollstd30
    gold_safehaven_signal = gold_rollmean7 - sp500_rollmean7
    return market_stress_proxy, gold_safehaven_signal
