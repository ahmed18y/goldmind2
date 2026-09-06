"""
app.py
------
Flask backend for the Gold Price Predictor.

Routes:
    GET  /                      -> the web app (templates/index.html)
    GET  /api/summary           -> current price + tomorrow + next-week predictions
    GET  /api/year-forecast     -> weekly-spaced predictions for the coming year
    GET  /api/history?range=1y|1m -> actual (non-predicted) historical prices
    GET  /api/predict?date=YYYY-MM-DD -> prediction for any custom future date
    GET  /api/health            -> simple liveness check
"""

from datetime import date, datetime

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from model import predictor

# static_folder=None: CSS/JS/vendor assets live in public/** and are served
# directly by Vercel's CDN (see public/), not through Flask. Vercel's own
# Flask docs are explicit that app.static_folder should not be relied on for
# static files in production - that mismatch (Flask serving /static/* itself
# while the deployed function's file layout doesn't guarantee those files are
# bundled the same way) is what caused the site to load with no CSS at all.
app = Flask(__name__, static_folder=None)


def _wants_force() -> bool:
    """True if the request explicitly asked to bypass the live-data cache
    (?force=1) - used by the manual refresh button in the UI."""
    return request.args.get("force") in ("1", "true", "True")


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    """Last-resort safety net: normal 404s / 400s etc. pass through as-is,
    but any genuinely unexpected crash (e.g. a corrupted model file) returns
    a clean JSON error instead of an HTML traceback page - appropriate for
    a JSON API, and avoids leaking internal details in production."""
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unhandled error on %s", request.path)
    return jsonify({"error": "internal server error, please try again"}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/summary")
def summary():
    snapshot = predictor.get_live_snapshot(force=_wants_force())

    tomorrow = predictor.predict_for_horizon(1, snapshot=snapshot)
    week = predictor.predict_week(snapshot=snapshot)
    meta = predictor.get_meta()

    return jsonify({
        "current_price": round(snapshot["anchor_price"], 2),
        "price_source": snapshot["status"],   # "live" | "partial_live" | "estimated_live" | "cached"
        "as_of": date.today().isoformat(),
        "tomorrow": tomorrow,
        "next_7_days": week,
        "model_data_through": meta["data_end"],
    })


@app.route("/api/year-forecast")
def year_forecast():
    snapshot = predictor.get_live_snapshot(force=_wants_force())
    curve = predictor.predict_year_curve(snapshot=snapshot)
    return jsonify({"forecast": curve})


@app.route("/api/history")
def history():
    range_param = request.args.get("range", "1m")
    days_map = {"1m": 30, "1y": 365}
    days = days_map.get(range_param)
    if days is None:
        return jsonify({"error": "range must be '1m' or '1y'"}), 400
    snapshot = predictor.get_live_snapshot(force=_wants_force())
    return jsonify({"history": predictor.get_history(days, snapshot=snapshot)})


@app.route("/api/predict")
def predict_custom_date():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "missing required query param 'date' (YYYY-MM-DD)"}), 400

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "date must be in YYYY-MM-DD format"}), 400

    snapshot = predictor.get_live_snapshot(force=_wants_force())

    try:
        result = predictor.predict_for_date(target_date, snapshot=snapshot)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result["price_source"] = snapshot["status"]
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
