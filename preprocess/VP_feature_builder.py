import pandas as pd
import sqlite3
import numpy as np
import os
import logging

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import utility.yaml_loader as loader
import utility.helpers as utility

def load_existing(PARQUET_FILE):
    # Load historical parquet (if exists)
    if os.path.exists(PARQUET_FILE):
        df_existing = pd.read_parquet(PARQUET_FILE)
        last_ts = df_existing["ts"].max()
        logging.info(f"Existing features parquet loaded up to {last_ts}")
    else:
        df_existing = pd.DataFrame()
        last_ts = None
        logging.info("No existing parquet found, full rebuild.")
    return df_existing, last_ts

def pull_new_data(DB_FILE, TABLE, last_ts):
    # Pull new data from DB
    conn = sqlite3.connect(DB_FILE)

    if last_ts is None:
        query = f"SELECT * FROM {TABLE} ORDER BY ts"
    else:
        query = f"SELECT * FROM {TABLE} WHERE ts > '{last_ts}' ORDER BY ts"

    df_new = pd.read_sql(query, conn, parse_dates=["ts"])
    conn.close()

    return df_new

def aggregate_df(df):
    return df.groupby("ts").apply(lambda x: pd.Series({
                                    "open": (x["open"] * x["volume"]).sum() / x["volume"].sum(),
                                    "high": x["high"].max(),
                                    "low": x["low"].min(),
                                    "close": (x["close"] * x["volume"]).sum() / x["volume"].sum(),
                                    "volume": x["volume"].sum()
                                })).reset_index()

def assign_session(ts):
    hour = ts.hour
    if 0 <= hour < 6:
        return "Asia"
    elif 6 <= hour < 12:
        return "London"
    elif 12 <= hour < 20:
        return "NewYork"
    else:
        return "Close"

def compute_session_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session-level features incrementally (no lookahead).
    Each row will only use data available up to that bar.
    """
    sess_open = group["open"].iloc[0]

    highs, lows, closes, vols = [], [], [], []
    results = []

    for i, row in group.iterrows():
        highs.append(row["high"])
        lows.append(row["low"])
        closes.append(row["close"])
        vols.append(row["volume"])

        sess_high = max(highs)
        sess_low = min(lows)
        sess_close = closes[-1]
        sess_volume = sum(vols)

        # Initial Balance (first 4 bars only if available)
        IB_high = max(highs[:min(4, len(highs))])
        IB_low = min(lows[:min(4, len(lows))])
        IB_range = IB_high - IB_low

        # Volume Profile up to now
        bins = np.linspace(sess_low, sess_high, 101)
        volume_per_bin = np.zeros(len(bins) - 1)

        for j in range(len(highs)):
            bar_low, bar_high, vol = lows[j], highs[j], vols[j]
            mask = (bins[:-1] < bar_high) & (bins[1:] > bar_low)
            if mask.sum() > 0:
                volume_per_bin[mask] += vol / mask.sum()
            else:
                idx = np.searchsorted(bins, closes[j]) - 1
                idx = np.clip(idx, 0, len(volume_per_bin)-1)  # safety
                volume_per_bin[idx] += vol

        # POC
        POC_idx = np.argmax(volume_per_bin)
        POC = (bins[POC_idx] + bins[POC_idx + 1]) / 2

        # Value Area (70% of volume so far)
        sorted_idx = np.argsort(volume_per_bin)[::-1]
        cum_vol = np.cumsum(volume_per_bin[sorted_idx])
        VA_total = cum_vol[-1] * 0.7
        VA_mask = cum_vol <= VA_total
        if VA_mask.any():
            VA_bins = sorted_idx[VA_mask]
            VAH = bins[VA_bins.max() + 1]
            VAL = bins[VA_bins.min()]
        else:
            VAH = sess_high
            VAL = sess_low

        results.append({
            "ts": row["ts"],
            "session_date": row["session_date"],
            "session": row["session"],
            "session_open": sess_open,
            "session_close": sess_close,
            "session_high": sess_high,
            "session_low": sess_low,
            "session_volume": sess_volume,
            "IB_high": IB_high,
            "IB_low": IB_low,
            "IB_range": IB_range,
            "POC_VP": POC,
            "VAH_VP": VAH,
            "VAL_VP": VAL,
        })

    return pd.DataFrame(results)

def parallel_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parallelize incremental session feature calculation.
    """
    groups = [g for _, g in df.groupby(["session_date", "session"])]
    n_workers = max(1, multiprocessing.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        results = list(exe.map(compute_session_features, groups))

    return pd.concat(results, ignore_index=True)

def compute_vwap_features(df):

    # Typical price and price*volume
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["pv"] = df["typical"] * df["volume"]

    bars_per_day = 24 * 4

    # ---- Rolling VWAPs ----
    for days in (7, 30, 90):
        window = days * bars_per_day
        pv_roll = df["pv"].rolling(window=window, min_periods=1).sum()
        vol_roll = df["volume"].rolling(window=window, min_periods=1).sum()
        df[f"VWAP_{days}d"] = pv_roll / vol_roll.replace({0: np.nan})

    # =========================================================
    # ====================== MONTHLY ==========================
    # =========================================================

    df["month"] = df["ts"].dt.to_period("M")

    # anchored VWAP inside month
    df["month_pv_cum"] = df.groupby("month")["pv"].cumsum()
    df["month_vol_cum"] = df.groupby("month")["volume"].cumsum()
    df["VWAP_month_anchored"] = df["month_pv_cum"] / df["month_vol_cum"].replace({0: np.nan})

    # final monthly VWAP
    monthly_totals = df.groupby("month").agg(
        pv_sum=("pv", "sum"),
        vol_sum=("volume", "sum")
    )
    monthly_totals["vwap"] = monthly_totals["pv_sum"] / monthly_totals["vol_sum"].replace({0: np.nan})

    # -------- volume-weighted std for month --------
    # merge vwap back to bar level
    df["month_vwap_final"] = df["month"].map(monthly_totals["vwap"])

    # deviation squared * volume
    df["month_var_term"] = (df["typical"] - df["month_vwap_final"]) ** 2 * df["volume"]

    # aggregate var terms
    month_var = df.groupby("month").agg(
        var_sum=("month_var_term", "sum"),
        vol_sum=("volume", "sum")
    )

    month_var["std1"] = np.sqrt(month_var["var_sum"] / month_var["vol_sum"].replace({0: np.nan}))

    # map std back
    df["VWAP_month_std1"] = df["month"].map(month_var["std1"])

    # 1σ bands
    df["VWAP_month_upper_1std"] = df["month_vwap_final"] + df["VWAP_month_std1"]
    df["VWAP_month_lower_1std"] = df["month_vwap_final"] - df["VWAP_month_std1"]

    # -------- previous month VWAP + std --------
    monthly_totals["vwap_prev"] = monthly_totals["vwap"].shift(1)
    month_var["std1_prev"] = month_var["std1"].shift(1)

    df["VWAP_prev_month_final"] = df["month"].map(monthly_totals["vwap_prev"])
    df["VWAP_prev_month_std1"] = df["month"].map(month_var["std1_prev"])

    df["VWAP_prev_month_upper_1std"] = df["VWAP_prev_month_final"] + df["VWAP_prev_month_std1"]
    df["VWAP_prev_month_lower_1std"] = df["VWAP_prev_month_final"] - df["VWAP_prev_month_std1"]

    # =========================================================
    # ====================== QUARTERLY ========================
    # =========================================================

    df["quarter"] = df["ts"].dt.to_period("Q")

    df["q_pv_cum"] = df.groupby("quarter")["pv"].cumsum()
    df["q_vol_cum"] = df.groupby("quarter")["volume"].cumsum()
    df["VWAP_quarter_anchored"] = df["q_pv_cum"] / df["q_vol_cum"].replace({0: np.nan})

    quarterly_totals = df.groupby("quarter").agg(
        pv_sum=("pv", "sum"),
        vol_sum=("volume", "sum")
    )
    quarterly_totals["vwap"] = quarterly_totals["pv_sum"] / quarterly_totals["vol_sum"].replace({0: np.nan})

    # compute var terms
    df["quarter_vwap_final"] = df["quarter"].map(quarterly_totals["vwap"])
    df["quarter_var_term"] = (df["typical"] - df["quarter_vwap_final"]) ** 2 * df["volume"]

    quarter_var = df.groupby("quarter").agg(
        var_sum=("quarter_var_term", "sum"),
        vol_sum=("volume", "sum")
    )
    quarter_var["std1"] = np.sqrt(quarter_var["var_sum"] / quarter_var["vol_sum"].replace({0: np.nan}))

    df["VWAP_quarter_std1"] = df["quarter"].map(quarter_var["std1"])

    df["VWAP_quarter_upper_1std"] = df["quarter_vwap_final"] + df["VWAP_quarter_std1"]
    df["VWAP_quarter_lower_1std"] = df["quarter_vwap_final"] - df["VWAP_quarter_std1"]

    # previous quarter
    quarterly_totals["vwap_prev"] = quarterly_totals["vwap"].shift(1)
    quarter_var["std1_prev"] = quarter_var["std1"].shift(1)

    df["VWAP_prev_quarter_final"] = df["quarter"].map(quarterly_totals["vwap_prev"])
    df["VWAP_prev_quarter_std1"] = df["quarter"].map(quarter_var["std1_prev"])

    df["VWAP_prev_quarter_upper_1std"] = df["VWAP_prev_quarter_final"] + df["VWAP_prev_quarter_std1"]
    df["VWAP_prev_quarter_lower_1std"] = df["VWAP_prev_quarter_final"] - df["VWAP_prev_quarter_std1"]

    return df

def add_volume_stats(df, window, z_col, mad_col):
    """
    Add rolling z-score and MAD-based score for volume across the full time series.
    More robust handling to avoid NaNs and divide-by-zero.
    """
    vol = df["volume"]

    # Use min_periods=1 so we compute starting immediately (you can set it higher if desired)
    roll_mean = vol.rolling(window, min_periods=1).mean()
    # use ddof=0 (population std) so std is 0 rather than NaN for single value windows
    roll_std = vol.rolling(window, min_periods=1).std(ddof=0)

    # Rolling median and MAD (median absolute deviation)
    roll_median = vol.rolling(window, min_periods=1).median()
    roll_mad = vol.rolling(window, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # Replace zeros (or very small) in denominators with epsilon to avoid division by zero / inf / NaN
    eps = 1e-8
    roll_std = roll_std.replace(0, eps).fillna(eps)
    roll_mad = roll_mad.replace(0, eps).fillna(eps)

    df[z_col] = (vol - roll_mean) / roll_std
    df[mad_col] = (vol - roll_median) / roll_mad

    return df

def compute_price_delta_zscore(df, windows=[16, 96]):
    """
    Compute price delta (returns) and z-scores over multiple windows
    
    Args:
        df: DataFrame with 'close' column
        windows: List of lookback periods for z-score calculation
    """
    # Calculate percentage returns
    df['price_delta_pct'] = df['close'].pct_change() * 100
    
    # Calculate absolute price change (alternative metric)
    df['price_delta_abs'] = df['close'].diff()
    
    # Calculate z-scores for different windows
    for window in windows:
        # Z-score of percentage returns
        rolling_mean = df['price_delta_pct'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['price_delta_pct'].rolling(window=window, min_periods=1).std(ddof=0)
        df[f'price_delta_z{window}'] = (df['price_delta_pct'] - rolling_mean) / rolling_std.replace(0, 1e-6)
    
    return df

def merge_and_save(df_existing, df_new, PARQUET_FILE):
    # If existing parquet has timestamps, match its timezone/dtype
    df_new["ts"] = pd.to_datetime(df_new["ts"], utc=True)

    # Merge new + existing data
    if not df_existing.empty:
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        df_out = df_out.drop_duplicates(subset=["ts"])
    else:
        df_out = df_new

    # Sort for safety
    df_out = df_out.sort_values("ts").reset_index(drop=True)

    # Save to parquet
    df_out.to_parquet(PARQUET_FILE, engine="pyarrow", index=False)

    # Safe logging (won’t raise tz mismatch)
    last_ts = df_out["ts"].max()
    logging.info(f"VP Parquet updated up to {last_ts}")

def main(asset: str = "btc"):
    asset = asset.lower()
    assert asset in {"btc", "eth", "sol"}, "asset must be btc, eth, or sol"

    # === CONFIG ===
    cfg = loader.load_config("config/rules_config.yaml")

    # shared DB file
    DB_FILE = cfg.files.db

    # --- choose asset parquet target & DB table ---
    if asset == "btc":
        PARQUET_FILE = cfg.btc_files.vp
        TABLE = "btc_15m_ohlcv_clean"
    elif asset == "eth":
        PARQUET_FILE = cfg.eth_files.vp
        TABLE = "eth_15m_ohlcv_clean"
    else:  # sol
        PARQUET_FILE = cfg.sol_files.vp
        TABLE = "sol_15m_ohlcv_clean"

    utility.setup_logging()
    logging.info(f"Running pipeline for asset={asset.upper()}")

    # Check for existing parquet data
    df_existing, last_ts = load_existing(PARQUET_FILE)

    # Pull data from DB
    df_new = pull_new_data(DB_FILE, TABLE, last_ts)
    if df_new.empty:
        logging.info(f"No new data needed for {asset.upper()}. Parquet already up to date.")
        return

    logging.info(f"Pulled {len(df_new)} new rows from DB for {asset.upper()}.")

    # Aggregate DB
    df_agg = aggregate_df(df_new)

    # Assign Session Time
    df_agg["session"] = df_agg["ts"].apply(assign_session)
    df_agg["session_date"] = df_agg["ts"].dt.date

    # ===== CONTINUITY FIX =====
    MAX_WINDOW = max(60, 96, 90 * 24 * 4)

    if not df_existing.empty:
        df_tail = df_existing.tail(MAX_WINDOW * 2).copy()

        last_session = df_existing["session"].iloc[-1]
        last_session_date = df_existing["session_date"].iloc[-1]

        mask_session = (
            (df_existing["session"] == last_session)
            & (df_existing["session_date"] == last_session_date)
        )
        df_last_session = df_existing[mask_session]

        df_context = pd.concat([df_tail, df_last_session]).drop_duplicates(subset=["ts"])

        df_combined = pd.concat([df_context, df_agg], ignore_index=True)
        logging.info(
            f"{asset.upper()}: Using {len(df_context)} historical rows for continuity."
        )
    else:
        df_combined = df_agg

    # Compute Session Feature
    df_session = parallel_session_features(df_combined)

    # VWAP (including std bands you just added)
    compute_vwap_features(df_combined)

    # Price Delta Z-Scores
    df_combined = compute_price_delta_zscore(df_combined, windows=[16, 32])

    # Volume Stats
    df_combined = add_volume_stats(df_combined, 20, "volume_zscore_20", "volume_mad_20")
    df_combined = add_volume_stats(df_combined, 60, "volume_zscore_60", "volume_mad_60")

    # Normalize timestamp
    for d in (df_combined, df_agg, df_session):
        d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)

    # select session columns
    session_cols = [
        "ts", "session_open", "session_close", "session_high", "session_low",
        "session_volume", "IB_high", "IB_low", "IB_range",
        "POC_VP", "VAH_VP", "VAL_VP"
    ]
    df_session_final = df_session[df_session["ts"].isin(df_agg["ts"])][session_cols].copy()

    # Merge results
    if df_existing.empty:
        df_final = df_combined.merge(df_session_final, on="ts", how="left")
    else:
        upd_cols = [
            "session_open", "session_close", "session_high", "session_low",
            "session_volume", "IB_high", "IB_low", "IB_range",
            "POC_VP", "VAH_VP", "VAL_VP"
        ]

        df_session_final["ts"] = pd.to_datetime(df_session_final["ts"]).dt.tz_localize(None)

        df_combined = df_combined.set_index("ts")
        df_session_final = df_session_final.set_index("ts")

        df_combined.update(df_session_final[upd_cols])

        df_combined = df_combined.reset_index()

        df_final = df_combined[df_combined["ts"].isin(df_agg["ts"])].copy()

    # Save parquet
    merge_and_save(df_existing, df_final, PARQUET_FILE)

    logging.info(f"Finished parquet update for {asset.upper()}")

if __name__ == "__main__":
    # Setup Logging
    utility.setup_logging() 
    for asset in ["btc", "eth", "sol"]:
        try:
            main(asset)
        except Exception as e:
            logging.exception(f"Error processing {asset.upper()}: {e}")
