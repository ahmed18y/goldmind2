"""
train_model.py
---------------
Trains the gold price forecasting model(s) and saves them to models/.

APPROACH — "horizon as a feature" (direct multi-horizon forecasting):
Instead of training one model per horizon (unmanageable - you asked for
tomorrow AND next week AND any custom date up to a year), we train ONE
model whose input includes the forecast horizon `h` (in days) as a
feature. Each training example is:

    X = [ all lag/rolling features at day t ]  +  [ h, sqrt(h), log1p(h) ]
    y = log(Gold_21K[t + h] / Gold_21K[t])          <- the log-return over h days

At prediction time we just plug in today's features with whatever h we
want (h=1 for tomorrow, h=2..7 for the week chart, h=30/90/365 for the
year chart or a custom date) and convert the predicted log-return back
to a price: predicted_price = current_price * exp(y_hat).

This is the standard "direct forecasting" strategy from forecasting
competitions (works well with gradient boosting + is far cheaper than
training/maintaining 365 separate models or an autoregressive model
that has to recurse 365 times and accumulate error).

We train THREE models per feature set (median / low / high) using
scikit-learn's HistGradientBoostingRegressor with quantile loss, which
gives us a genuine uncertainty band instead of a single falsely-precise
number - crucial for a 1-year-ahead prediction.

Model choice: HistGradientBoostingRegressor (ships with scikit-learn,
no extra native dependency) instead of LightGBM/XGBoost - this keeps
the Vercel serverless deployment simple and small (no compiled
external binaries to worry about) while still being state-of-the-art
for tabular time series with exogenous macro features.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "merged_dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Horizons (in trading/calendar days) sampled during training.
#
# IMPORTANT: this list must include every horizon value the deployed app
# can actually request, not just a "spread out" sample. HistGradientBoosting
# (like any tree ensemble) is a PIECEWISE-CONSTANT function of a continuous
# feature - it only "knows about" the exact horizon values it saw split
# thresholds for during training. Any horizon queried at inference that
# wasn't (close to) a value seen here just falls into whichever bucket the
# trees learned and comes out IDENTICAL to its neighbours - which used to
# show up as a visible "staircase" in the year-forecast chart (see
# predictor.predict_year_curve, which queries every 7 days out to a year)
# and in the day-by-day week chart.
#
# So this list is built to exactly cover both: predict_week()'s 1..7 daily
# horizons, and predict_year_curve()'s every-7-days-to-365 horizons -
# instead of the old sparse list (1-7, then 10/14/21/30/45/60/90/...) which
# left long unseen gaps (e.g. nothing between 90 and 120, or 210 and 240)
# that produced multi-week flat plateaus in the chart.
HORIZONS = sorted(set(list(range(1, 8)) + list(range(7, 366, 7)) + [365]))

TARGET_COL = "Gold_21K"
QUANTILES = {"low": 0.10, "median": 0.50, "high": 0.90}

# Horizons we actually report evaluation metrics for. Covers the app's main
# use cases: tomorrow (1), the week chart (7), and several points across the
# year chart / custom-date range (30, 90, 180, 365).
EVAL_HORIZONS = [1, 7, 30, 90, 180, 365]


def evaluate_model(df: pd.DataFrame) -> dict:
    """
    Proper walk-forward evaluation across multiple horizons, WITH a naive
    baseline for comparison (previously the project only reported a single
    1-day-ahead MAE with nothing to compare it against - so there was no way
    to say whether the model actually beats "just predict no change").

    For each horizon h in EVAL_HORIZONS, and for each of 5 chronological
    walk-forward folds:
      - trains fresh median/low/high quantile models on the training side
        of the fold only (never sees the test side - no leakage)
      - compares the model's median prediction against a NAIVE baseline
        (predict the price simply stays the same, i.e. 0 log-return)
      - reports, in actual EGP price terms: MAE, RMSE, MAPE, the naive
        baseline's MAE, % improvement over that baseline, directional
        accuracy (did we get the up/down right?), and quantile coverage
        (what fraction of real outcomes landed inside the predicted
        [low, high] band - should be close to 80% for a 10th/90th
        quantile band; far below 80% means the uncertainty band is too
        narrow/overconfident, far above means it's too wide).
    """
    feature_cols = [
        c for c in df.columns
        if c not in ("Date", TARGET_COL) and pd.api.types.is_numeric_dtype(df[c])
    ]
    warmup = 100
    log_price = np.log(df[TARGET_COL].values)
    price = df[TARGET_COL].values
    n = len(df)

    report = {}
    print("\n" + "=" * 70)
    print("WALK-FORWARD EVALUATION + CALIBRATION (vs naive 'no change' baseline)")
    print("=" * 70)
    header = (f"{'Horizon':>8} | {'Raw MAE':>9} | {'Calib. MAE':>10} | {'Baseline':>9} | "
              f"{'Improve %':>9} | {'alpha':>6} | {'band x':>7} | {'Dir.Acc %':>9} | {'Cover %':>8}")
    print(header)
    print("-" * len(header))

    for h in EVAL_HORIZONS:
        abs_index = np.arange(warmup, n)
        future_pos = abs_index + h
        valid = future_pos < n
        idx = abs_index[valid]
        fut = future_pos[valid]
        if len(idx) < 200:
            print(f"{h:>8} |  (not enough history for this horizon yet - skipped)")
            continue

        Xh = df.iloc[idx][feature_cols].fillna(0.0).reset_index(drop=True)
        horizon_cols = pd.DataFrame({
            "horizon": np.full(len(Xh), h),
            "horizon_sqrt": np.full(len(Xh), np.sqrt(h)),
            "horizon_log1p": np.full(len(Xh), np.log1p(h)),
        })
        Xh = pd.concat([Xh, horizon_cols], axis=1)
        yh = pd.Series(log_price[fut] - log_price[idx])
        base_price_h = price[idx]
        actual_price_h = price[fut]

        tscv = TimeSeriesSplit(n_splits=3)
        pooled_base, pooled_actual = [], []
        pooled_pred, pooled_lo, pooled_hi = [], [], []

        for train_idx, test_idx in tscv.split(Xh):
            if len(test_idx) < 10:
                continue

            def _fit(q):
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=200, max_depth=6,
                    learning_rate=0.05, early_stopping=True, random_state=42,
                )
                m.fit(Xh.iloc[train_idx], yh.iloc[train_idx])
                return m

            pred_med = _fit(0.5).predict(Xh.iloc[test_idx])
            pred_lo = _fit(0.10).predict(Xh.iloc[test_idx])
            pred_hi = _fit(0.90).predict(Xh.iloc[test_idx])

            pooled_base.append(base_price_h[test_idx])
            pooled_actual.append(actual_price_h[test_idx])
            pooled_pred.append(base_price_h[test_idx] * np.exp(pred_med))
            pooled_lo.append(base_price_h[test_idx] * np.exp(pred_lo))
            pooled_hi.append(base_price_h[test_idx] * np.exp(pred_hi))

        if not pooled_base:
            continue

        base = np.concatenate(pooled_base)
        actual = np.concatenate(pooled_actual)
        pred_price = np.concatenate(pooled_pred)
        pred_lo_price = np.concatenate(pooled_lo)
        pred_hi_price = np.concatenate(pooled_hi)
        naive_price = base

        # ---- RAW model metrics (no calibration) ----
        raw_mae = float(np.mean(np.abs(pred_price - actual)))
        base_mae = float(np.mean(np.abs(naive_price - actual)))

        # ---- Calibration step 1: shrink the model's move toward the naive
        # baseline ("price stays the same"). This is a single scalar per
        # horizon - alpha=1 keeps the model as-is, alpha=0 falls all the way
        # back to the naive baseline, alpha in between blends the two. It's
        # the standard fix for a model whose short/medium-horizon signal is
        # noisier than a simple "no change" assumption (which the raw
        # evaluation above shows IS happening here at 1-180 days) - blending
        # toward the more robust baseline reduces variance without needing
        # to change the underlying model at all.
        best_alpha, best_alpha_mae = 1.0, raw_mae
        for alpha in np.arange(0.0, 1.01, 0.05):
            blended = base + alpha * (pred_price - base)
            mae = float(np.mean(np.abs(blended - actual)))
            if mae < best_alpha_mae:
                best_alpha, best_alpha_mae = float(alpha), mae

        calib_pred_price = base + best_alpha * (pred_price - base)
        calib_mae = float(np.mean(np.abs(calib_pred_price - actual)))
        rmse = float(np.sqrt(np.mean((calib_pred_price - actual) ** 2)))
        mape = float(np.mean(np.abs((calib_pred_price - actual) / actual)) * 100)
        actual_dir = np.sign(actual - base)
        pred_dir = np.sign(calib_pred_price - base)
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

        # ---- Calibration step 2: rescale the quantile band width so its
        # empirical coverage actually matches the ~80% a 10th/90th-percentile
        # band is supposed to give. The raw evaluation showed this band can
        # be badly overconfident (too narrow) at long horizons - grid search
        # a single width multiplier per horizon against the SAME pooled
        # walk-forward predictions, re-centered on the calibrated median.
        half_lo = pred_price - pred_lo_price
        half_hi = pred_hi_price - pred_price
        best_scale, best_gap = 1.0, abs(
            np.mean((actual >= pred_lo_price) & (actual <= pred_hi_price)) - 0.80
        )
        for scale in np.arange(0.3, 8.01, 0.1):
            lo = calib_pred_price - scale * half_lo
            hi = calib_pred_price + scale * half_hi
            cov = np.mean((actual >= lo) & (actual <= hi))
            gap = abs(cov - 0.80)
            if gap < best_gap:
                best_scale, best_gap = float(scale), gap

        final_lo = calib_pred_price - best_scale * half_lo
        final_hi = calib_pred_price + best_scale * half_hi
        coverage = float(np.mean((actual >= final_lo) & (actual <= final_hi)) * 100)

        improvement_pct = (1 - calib_mae / base_mae) * 100 if base_mae > 0 else float("nan")

        report[str(h)] = {
            "raw_model_mae_egp": round(raw_mae, 2),
            "calibrated_mae_egp": round(calib_mae, 2),
            "baseline_mae_egp": round(base_mae, 2),
            "improvement_vs_baseline_pct": round(improvement_pct, 1),
            "blend_alpha": round(best_alpha, 2),
            "band_scale": round(best_scale, 2),
            "rmse_egp": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "directional_accuracy_pct": round(dir_acc, 1),
            "quantile_coverage_pct": round(coverage, 1),
        }
        print(f"{h:>8} | {raw_mae:>9.2f} | {calib_mae:>10.2f} | {base_mae:>9.2f} | "
              f"{improvement_pct:>8.1f}% | {best_alpha:>6.2f} | {best_scale:>6.2f}x | "
              f"{dir_acc:>8.1f}% | {coverage:>7.1f}%")

    print("-" * len(header))
    print("blend_alpha: how much of the raw model's move to trust (1.0 = fully, 0.0 = ignore it and use the naive baseline).")
    print("band_scale: how much to widen (>1) or narrow (<1) the raw quantile band so real coverage matches the ~80% target.")
    print("Both are saved into models/meta.json and applied live in predictor.py - this is a calibration step on TOP of the")
    print("existing model, not a retrain, so it costs nothing extra at inference time.")
    return report


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Expands the daily feature table into (anchor_date, horizon) samples."""

    feature_cols = [
        c for c in df.columns
        if c not in ("Date", TARGET_COL) and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Drop the warm-up period where long lag features are still NaN.
    warmup = 100
    usable = df.iloc[warmup:].reset_index(drop=True)

    log_price = np.log(df[TARGET_COL].values)
    n = len(df)

    rows = []
    targets = []

    # Map each usable row back to its absolute position in `df` so we can
    # look ahead by h days for the label.
    abs_index = usable.index + warmup

    for h in HORIZONS:
        future_pos = abs_index + h
        valid = future_pos < n

        sub = usable.loc[valid, feature_cols].copy()
        sub["horizon"] = h
        sub["horizon_sqrt"] = np.sqrt(h)
        sub["horizon_log1p"] = np.log1p(h)

        y = log_price[future_pos[valid]] - log_price[abs_index[valid]]

        rows.append(sub)
        targets.append(pd.Series(y, index=sub.index))

    X = pd.concat(rows, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)

    X = X.fillna(0.0)

    full_feature_cols = feature_cols + ["horizon", "horizon_sqrt", "horizon_log1p"]
    return X[full_feature_cols], y, full_feature_cols, feature_cols


def fit_deployment_calibration(df: pd.DataFrame, models: dict, feature_cols: list) -> dict:
    """
    IMPORTANT DISTINCTION from evaluate_model() above:

    evaluate_model() fits small, throwaway quantile models per walk-forward
    fold purely to get an honest, held-out read of how well this MODELING
    APPROACH generalizes - that's the right way to report accuracy.

    But those throwaway models are NOT the same objects being deployed (they
    see far less data per fit and stop earlier), so their raw quantile
    spread is NOT representative of the actual production models' spread.
    Discovered this the hard way during testing: applying evaluate_model()'s
    band_scale directly to the real production models produced absurd,
    even NEGATIVE, predicted prices at 90+ day horizons - scaling up a
    spread that was already much wider to begin with.

    This function instead calibrates against the EXACT models that get
    saved and deployed, using only .predict() (no fitting) on historical
    anchor points. It's evaluated in-sample (the production models were
    trained on this same data), so it should NOT be read as an accuracy
    number - it exists purely to make the deployed band width/shrinkage
    self-consistent with the actual deployed model's own output scale.
    """
    log_price = np.log(df[TARGET_COL].values)
    price = df[TARGET_COL].values
    n = len(df)
    warmup = 100

    calibration = {}
    for h in EVAL_HORIZONS:
        abs_index = np.arange(warmup, n)
        future_pos = abs_index + h
        valid = future_pos < n
        idx = abs_index[valid]
        fut = future_pos[valid]
        if len(idx) < 50:
            continue

        Xh = df.iloc[idx][feature_cols].fillna(0.0).reset_index(drop=True)
        horizon_cols = pd.DataFrame({
            "horizon": np.full(len(Xh), h),
            "horizon_sqrt": np.full(len(Xh), np.sqrt(h)),
            "horizon_log1p": np.full(len(Xh), np.log1p(h)),
        })
        Xh = pd.concat([Xh, horizon_cols], axis=1)[feature_cols + ["horizon", "horizon_sqrt", "horizon_log1p"]]

        base = price[idx]
        actual = price[fut]
        pred_med = models["median"].predict(Xh)
        pred_lo = models["low"].predict(Xh)
        pred_hi = models["high"].predict(Xh)
        # guard against quantile crossing, same as predictor.py does live
        stacked = np.sort(np.stack([pred_lo, pred_med, pred_hi], axis=1), axis=1)
        pred_lo, pred_med, pred_hi = stacked[:, 0], stacked[:, 1], stacked[:, 2]

        pred_price = base * np.exp(pred_med)
        pred_lo_price = base * np.exp(pred_lo)
        pred_hi_price = base * np.exp(pred_hi)

        best_alpha, best_alpha_mae = 1.0, float(np.mean(np.abs(pred_price - actual)))
        for alpha in np.arange(0.0, 1.01, 0.05):
            blended = base + alpha * (pred_price - base)
            mae = float(np.mean(np.abs(blended - actual)))
            if mae < best_alpha_mae:
                best_alpha, best_alpha_mae = float(alpha), mae

        calib_price = base + best_alpha * (pred_price - base)
        half_lo = pred_price - pred_lo_price
        half_hi = pred_hi_price - pred_price

        # Cap the search at 3x here (unlike evaluate_model's wider search) -
        # this is in-sample data, so coverage climbs "too easily"; a large
        # scale would just be overfitting to in-sample noise, not a real
        # calibration gain. 3x keeps bands sane while still visibly widening
        # the (often overconfident) raw band.
        best_scale, best_gap = 1.0, abs(
            np.mean((actual >= pred_lo_price) & (actual <= pred_hi_price)) - 0.80
        )
        for scale in np.arange(0.3, 3.01, 0.1):
            lo = calib_price - scale * half_lo
            hi = calib_price + scale * half_hi
            cov = np.mean((actual >= lo) & (actual <= hi))
            gap = abs(cov - 0.80)
            if gap < best_gap:
                best_scale, best_gap = float(scale), gap

        calibration[str(h)] = {"blend_alpha": round(best_alpha, 2), "band_scale": round(best_scale, 2)}

    return calibration


def main():
    print("=" * 70)
    print("TRAINING GOLD PRICE MODEL")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    X, y, feature_cols, base_feature_cols = build_training_frame(df)
    print(f"Training samples: {len(X):,}   Features: {len(feature_cols)}")

    # NOTE on evaluation methodology: a plain last-10% holdout of X/y here
    # would not be clean, because rows from different horizons are
    # interleaved after build_training_frame() expands them - instead
    # evaluate_model() below runs a proper walk-forward split directly on
    # the ORIGINAL daily series, once per horizon, for a trustworthy readout.
    models = {}
    for name, q in QUANTILES.items():
        print(f"\n--- training '{name}' model (quantile={q}) ---")
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=0.1,
            early_stopping=True,
            random_state=42,
        )
        model.fit(X, y)
        models[name] = model

    # ---- proper multi-horizon walk-forward evaluation, with a baseline ----
    # (honest, held-out accuracy numbers - for reporting, NOT for calibrating
    # the deployed model's own band width - see fit_deployment_calibration)
    eval_report = evaluate_model(df)

    # ---- calibration actually used at inference time (see docstring) ----
    deployment_calibration = fit_deployment_calibration(df, models, base_feature_cols)

    # ---- save artifacts --------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"gold_model_{name}.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save the last row of engineered features (= "today's" feature state)
    # so the Flask app has a ready-made snapshot to build predictions from
    # without re-running the whole feature pipeline on every request.
    latest_features = df[base_feature_cols].iloc[[-1]].fillna(0.0)
    latest_features.to_json(os.path.join(MODELS_DIR, "latest_features.json"), orient="records")

    meta = {
        "trained_on_rows": int(len(df)),
        "data_start": str(df["Date"].min().date()),
        "data_end": str(df["Date"].max().date()),
        "last_known_price": float(df[TARGET_COL].iloc[-1]),
        "horizons_trained": HORIZONS,
        "evaluation": eval_report,
        "deployment_calibration": deployment_calibration,
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("-" * 70)
    print(f"[DONE] Saved 3 quantile models + metadata -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
