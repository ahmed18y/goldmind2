import os
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# PROJECT SETTINGS
# ============================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

# Historical start for Yahoo market data.
# We intentionally start early because we want to preserve
# the full available historical range.
YAHOO_START_DATE = "2000-01-01"

# Yahoo's "end" is exclusive, so use tomorrow to include
# today's available observations.
YAHOO_END_DATE = (
    date.today() + timedelta(days=1)
).isoformat()


# ============================================================
# FOLDER STRUCTURE
# ============================================================

FOLDERS = {
    "TARGET": os.path.join(BASE_DIR, "01_TARGET"),
    "CORE": os.path.join(BASE_DIR, "02_CORE"),
    "STRONG": os.path.join(BASE_DIR, "03_STRONG"),
    "EXPERIMENTAL": os.path.join(BASE_DIR, "04_EXPERIMENTAL"),
    "EVENTS": os.path.join(BASE_DIR, "06_EVENTS"),
}

for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)


# ============================================================
# YAHOO FINANCE
# ============================================================

# Main source files.
#
# We deliberately keep only one source for variables where
# we currently have duplicates (e.g. WTI/FRED backup,
# S&P/FRED backup).
YAHOO_DATA = {
    "CORE": {
        "GC=F": "Gold_World.csv",
        "EGP=X": "USD_EGP.csv",
    },

    "STRONG": {
        "DX-Y.NYB": "US_Dollar_Index_DXY.csv",
        "CL=F": "WTI_Oil.csv",
    },

    "EXPERIMENTAL": {
        "^GSPC": "SP500.csv",
        "^VIX": "VIX.csv",
        "SI=F": "Silver.csv",
    },
}


def download_yahoo_data(ticker: str, filename: str, folder: str) -> bool:
    """
    Download one Yahoo Finance daily series and save it as CSV.

    Important:
    - No missing-value treatment.
    - No interpolation.
    - No resampling.
    - No merging.
    - No feature engineering.
    - No model-related processing.
    """

    print("\n" + "=" * 70)
    print(f"YAHOO: {ticker}")
    print(f"OUTPUT: {filename}")
    print(f"PERIOD: {YAHOO_START_DATE} -> {YAHOO_END_DATE}")
    print("=" * 70)

    try:
        df = yf.download(
            ticker,
            start=YAHOO_START_DATE,
            end=YAHOO_END_DATE,
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=False,
            progress=True,
            threads=False,
        )

        if df is None or df.empty:
            print("[ERROR] Empty response.")
            return False

        # yfinance can return MultiIndex columns even for one ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date

        output_path = os.path.join(folder, filename)

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"[OK] Saved: {output_path}")
        print(f"Rows      : {len(df):,}")

        if "Date" in df.columns:
            print(f"First date: {df['Date'].min()}")
            print(f"Last date : {df['Date'].max()}")

        return True

    except Exception as exc:
        print(f"[ERROR] Yahoo download failed for {ticker}")
        print(f"Reason: {exc}")
        return False


# ============================================================
# FRED
# ============================================================

# We keep the current main FRED series that we actually need.
#
# No FRED API key is required here because we are using FRED's
# public graph CSV endpoint rather than the authenticated API.
FRED_DATA = {
    "DGS10": (
        "US_10Y_Treasury_Yield.csv",
        "STRONG",
    ),

    # Economic Policy Uncertainty - captures how "unsettled" the economic
    # policy backdrop is (elections, trade wars, debt-ceiling fights, ...).
    # US daily series.
    "USEPUINDXD": (
        "US_Economic_Policy_Uncertainty_Daily.csv",
        "EVENTS",
    ),

    # Global Economic Policy Uncertainty - GDP-weighted average across 20
    # economies. Monthly only (FRED does not publish a daily global series).
    "GEPUCURRENT": (
        "Global_Economic_Policy_Uncertainty_Monthly.csv",
        "EVENTS",
    ),
}


def download_fred_csv(series_id: str, filename: str, category: str) -> bool:
    """
    Download a FRED series through the public CSV endpoint.
    """

    print("\n" + "=" * 70)
    print(f"FRED: {series_id}")
    print(f"OUTPUT: {filename}")
    print("=" * 70)

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
    )

    try:
        df = pd.read_csv(url)

        if df is None or df.empty:
            print("[ERROR] Empty response.")
            return False

        output_path = os.path.join(
            FOLDERS[category],
            filename
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"[OK] Saved: {output_path}")
        print(f"Rows: {len(df):,}")

        if "DATE" in df.columns:
            print(f"First date: {df['DATE'].min()}")
            print(f"Last date : {df['DATE'].max()}")

        return True

    except Exception as exc:
        print(f"[ERROR] FRED download failed for {series_id}")
        print(f"Reason: {exc}")
        return False


# ============================================================
# GEOPOLITICAL RISK (GPR) INDEX — Caldara & Iacoviello
# ============================================================

# The GPR index (Federal Reserve Board economists Caldara & Iacoviello)
# tallies newspaper coverage of geopolitical tensions (wars, terrorist
# threats, sanctions, ...) into a single daily/monthly index. This is
# the academically standard way to turn "political events" into a
# numeric feature a model can learn from, instead of hand-picking event
# dates ourselves.
#
# Published as a plain xlsx on the authors' own site - no API key,
# no auth. If the file layout on their site ever changes, this function
# fails gracefully (like every other source here) and the rest of the
# pipeline keeps working without it.
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def download_gpr_index() -> bool:
    """
    Download the daily Geopolitical Risk (GPR) index.
    """

    print("\n" + "=" * 70)
    print("GPR — GEOPOLITICAL RISK INDEX (Caldara & Iacoviello)")
    print("=" * 70)

    try:
        # This is a legacy .xls (not .xlsx) file, which needs the `xlrd`
        # engine specifically - openpyxl (already a dependency) only reads
        # .xlsx/.xlsm. Without xlrd installed, this raises:
        #   ImportError: Import xlrd failed. Install xlrd >= 2.0.1 for
        #   xls Excel support
        # which is exactly why this download was silently failing.
        df = pd.read_excel(GPR_DAILY_URL, engine="xlrd")

        if df is None or df.empty:
            print("[ERROR] Empty response.")
            return False

        # The published file's first column is the date, and "GPRD" is the
        # main daily geopolitical risk index column. We keep everything the
        # file provides (threats/acts sub-indices etc.) rather than
        # dropping columns ourselves.
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        output_path = os.path.join(
            FOLDERS["EVENTS"],
            "Geopolitical_Risk_Index_Daily.csv"
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[OK] Saved: {output_path}")
        print(f"Rows: {len(df):,}")
        print(f"First date: {df['Date'].min().date()}")
        print(f"Last date : {df['Date'].max().date()}")

        return True

    except Exception as exc:
        print("[ERROR] GPR download failed")
        print(f"Reason: {exc}")
        print("(Site layout may have changed - check "
              "https://www.matteoiacoviello.com/gpr.htm for the current file name)")
        return False


# ============================================================
# GOLD RATE 24 — EGYPT 21K
# ============================================================

# This is the source used by your current x.py.
# We keep the same endpoint/parameters because your existing
# collector already proved that it returns historical observations.
GOLDRATE24_URL = "https://www.elomla.com/api/historical/lastdays"

GOLDRATE24_PARAMS = {
    "base": "XAU",
    "to": "EGP",
    "days": 20000,
    "premium": "true",
    "format": "highcharts",
    "factor": 0.02813125,
}

GOLDRATE24_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def download_goldrate24() -> bool:
    """
    Download GoldRate24 21K historical data.

    IMPORTANT:
    The API response is converted to a clean tabular representation
    because timestamp -> Date is required to create the CSV.

    We intentionally do NOT:
    - drop NaNs
    - remove duplicates
    - interpolate
    - fill missing dates
    - resample
    - merge with other data
    """

    print("\n" + "=" * 70)
    print("GOLDRATE24 — EGYPT GOLD 21K")
    print("=" * 70)

    try:
        response = requests.get(
            GOLDRATE24_URL,
            params=GOLDRATE24_PARAMS,
            headers=GOLDRATE24_HEADERS,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("success"):
            raise RuntimeError(
                "GoldRate24 API returned success=False"
            )

        rates = data["data"]["rates"]

        print(f"API observations: {len(rates):,}")

        # API's historical values are pairs:
        # [timestamp_ms, Gold_21K]
        rows = []

        for item in rates:
            if len(item) < 2:
                continue

            timestamp_ms = item[0]
            value = item[1]

            rows.append({
                "timestamp_ms": timestamp_ms,
                "Gold_21K": value,
            })

        if not rows:
            raise RuntimeError(
                "No usable GoldRate24 observations returned."
            )

        df = pd.DataFrame(rows)

        # Convert timestamp to date.
        # This is a representation conversion, not modeling preprocessing.
        df["Date"] = pd.to_datetime(
            df["timestamp_ms"],
            unit="ms",
            utc=True,
        ).dt.tz_localize(None)

        df["Date"] = df["Date"].dt.normalize()

        # Remove only the technical timestamp column.
        # We keep the values as returned by the source.
        df = df[
            ["Date", "Gold_21K"]
        ]

        # IMPORTANT:
        # No dropna()
        # No drop_duplicates()
        # No sort_values()
        # No interpolation
        #
        # We preserve the source ordering/content as much as practical.

        output_path = os.path.join(
            FOLDERS["TARGET"],
            "GoldRate24_Egypt_21K.csv"
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )

        print(f"[OK] Saved: {output_path}")
        print(f"Rows: {len(df):,}")

        print(
            f"First source date: "
            f"{df['Date'].min().date()}"
        )

        print(
            f"Last source date : "
            f"{df['Date'].max().date()}"
        )

        return True

    except Exception as exc:
        print("[ERROR] GoldRate24 download failed")
        print(f"Reason: {exc}")
        return False


# ============================================================
# DOWNLOAD ALL
# ============================================================

results = []


def record_result(
    source: str,
    item: str,
    filename: str,
    category: str,
    success: bool,
):
    results.append({
        "Source": source,
        "Item": item,
        "File": filename,
        "Category": category,
        "Status": "SUCCESS" if success else "FAILED",
    })


def main():

    print("\n")
    print("#" * 70)
    print(" GOLD ML PROJECT — UNIFIED DATA DOWNLOADER")
    print("#" * 70)

    print(f"Yahoo start : {YAHOO_START_DATE}")
    print(f"Yahoo end   : {YAHOO_END_DATE}")
    print("#" * 70)

    # required_failures collects (source, item) pairs for anything that MUST
    # succeed for the dataset to be trustworthy. If this is non-empty at the
    # end, main() exits non-zero so the GitHub Actions job (and therefore the
    # "build dataset -> retrain -> commit" steps after it) actually STOPS
    # instead of silently retraining/committing on stale data while GitHub
    # still shows a green checkmark.
    required_failures = []

    # --------------------------------------------------------
    # 1. GoldRate24 Target (REQUIRED - this is the prediction target itself)
    # --------------------------------------------------------

    success = download_goldrate24()

    record_result(
        source="GoldRate24",
        item="Egypt Gold 21K",
        filename="GoldRate24_Egypt_21K.csv",
        category="TARGET",
        success=success,
    )
    if not success:
        required_failures.append("GoldRate24 (Egypt Gold 21K target)")

    # --------------------------------------------------------
    # 2. Yahoo (REQUIRED - core model features)
    # --------------------------------------------------------

    for category, ticker_map in YAHOO_DATA.items():

        folder = FOLDERS[category]

        for ticker, filename in ticker_map.items():

            success = download_yahoo_data(
                ticker=ticker,
                filename=filename,
                folder=folder,
            )

            record_result(
                source="Yahoo Finance",
                item=ticker,
                filename=filename,
                category=category,
                success=success,
            )
            if not success:
                required_failures.append(f"Yahoo Finance ({ticker})")

    # --------------------------------------------------------
    # 3. FRED (REQUIRED - includes US10Y + both EPU series)
    # --------------------------------------------------------

    for series_id, (filename, category) in FRED_DATA.items():

        success = download_fred_csv(
            series_id=series_id,
            filename=filename,
            category=category,
        )

        record_result(
            source="FRED",
            item=series_id,
            filename=filename,
            category=category,
            success=success,
        )
        if not success:
            required_failures.append(f"FRED ({series_id})")

    # --------------------------------------------------------
    # 3b. GPR (Geopolitical Risk Index) - OPTIONAL.
    #
    # This is not (yet) wired into build_dataset.py/feature_utils.py as a
    # real model feature (a market-stress proxy is used instead until a
    # real news-based index is available - see PROJECT_EXPLANATION.md), so
    # a failure here is logged as a warning but must NOT stop the pipeline.
    # --------------------------------------------------------

    success = download_gpr_index()

    record_result(
        source="Matteo Iacoviello (FRB)",
        item="GPR Daily Index",
        filename="Geopolitical_Risk_Index_Daily.csv",
        category="EVENTS",
        success=success,
    )
    if not success:
        print("[WARN] GPR download failed - continuing anyway (optional, not yet used as a model feature).")

    # --------------------------------------------------------
    # 4. Save download report
    # --------------------------------------------------------

    summary = pd.DataFrame(results)

    summary_path = os.path.join(
        BASE_DIR,
        "DOWNLOAD_SUMMARY.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("#" * 70)
    print(" DOWNLOAD FINISHED")
    print("#" * 70)

    print(summary.to_string(index=False))

    print("\n")
    print("Data directory:")
    print(os.path.abspath(BASE_DIR))

    print("\n")
    print("IMPORTANT:")
    print("- No ML preprocessing was performed.")
    print("- No NaN filling was performed.")
    print("- No interpolation was performed.")
    print("- No resampling was performed.")
    print("- No datasets were merged.")
    print("- No outlier removal was performed.")

    if required_failures:
        print("\n")
        print("#" * 70)
        print(" REQUIRED SOURCE(S) FAILED — STOPPING HERE")
        print("#" * 70)
        for item in required_failures:
            print(f"  - {item}")
        print(
            "\nNot building the dataset or retraining on this data: a "
            "required source failed, so today's data would be incomplete "
            "or stale. Exiting with a non-zero status so the CI job (and "
            "the build/train/commit steps after it) is marked as FAILED "
            "instead of silently succeeding on bad data."
        )
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()