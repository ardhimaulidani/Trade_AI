import os
import sqlite3
import logging
import pandas as pd
import numpy as np
import utility.yaml_loader as loader
import utility.helpers as utility


# === HELPERS ===
def load_existing(PARQUET_FILE):
    if os.path.exists(PARQUET_FILE):
        df_existing = pd.read_parquet(PARQUET_FILE)
        last_ts = df_existing["ts"].max()
        logging.info(f"Existing parquet loaded up to {last_ts}")
    else:
        df_existing = pd.DataFrame()
        last_ts = None
        logging.info("No existing parquet found, full rebuild.")
    return df_existing, last_ts


def pull_new_data(DB_FILE, TABLE, last_ts):
    conn = sqlite3.connect(DB_FILE)
    if last_ts is None:
        query = f"SELECT * FROM {TABLE} ORDER BY ts"
    else:
        query = f"SELECT * FROM {TABLE} WHERE ts > '{last_ts}' ORDER BY ts"
    df_new = pd.read_sql(query, conn, parse_dates=["ts"])
    conn.close()
    return df_new


def zscore(series, window):
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std(ddof=0)

    eps = 1e-8
    roll_std = roll_std.replace(0, eps).fillna(eps)

    return (series - roll_mean) / roll_std


def rolling_features(df):
    # Ensure numeric columns
    for col in ["oi", "oi_delta", "cvd", "delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Multi-horizon deltas
    df["oi_delta_1h"] = df["oi"].diff(4)
    df["oi_delta_4h"] = df["oi"].diff(16)
    df["cvd_delta_1h"] = df["cvd"].diff(4)
    df["cvd_delta_4h"] = df["cvd"].diff(16)

    # Z-scores
    for col in ["oi", "oi_delta", "cvd", "delta"]:
        for w in [16, 32, 96]:
            df[f"{col}_z{w}"] = zscore(df[col], w)

    return df


def merge_and_save(df_existing, df_new, PARQUET_FILE):
    if not df_existing.empty:
        df_out = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset=["ts"])
        )
    else:
        df_out = df_new

    df_out.to_parquet(PARQUET_FILE, engine="pyarrow", index=False)
    logging.info(f"Parquet updated. Total rows: {len(df_out)}")


# === PROCESS ONE ASSET ===
def process_asset(asset, cfg):
    logging.info(f"===== Processing {asset.upper()} derivatives =====")

    DB_FILE = cfg.files.db

    if asset == "btc":
        TABLE = "btc_deriv"
        PARQUET_FILE = cfg.btc_files.deriv
    elif asset == "eth":
        TABLE = "eth_deriv"
        PARQUET_FILE = cfg.eth_files.deriv
    else:
        TABLE = "sol_deriv"
        PARQUET_FILE = cfg.sol_files.deriv

    os.makedirs(os.path.dirname(PARQUET_FILE), exist_ok=True)

    # load existing parquet
    df_existing, last_ts = load_existing(PARQUET_FILE)

    # pull new data
    df_new = pull_new_data(DB_FILE, TABLE, last_ts)
    if df_new.empty:
        logging.info(f"{asset.upper()}: No new rows. Up to date.")
        return

    logging.info(f"{asset.upper()}: Pulled {len(df_new)} new rows.")

    MAX_WINDOW = 96

    if not df_existing.empty:
        df_context = df_existing.tail(MAX_WINDOW * 2).copy()
        df_combined = pd.concat([df_context, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = rolling_features(df_combined)

    df_new_final = df_combined.iloc[-len(df_new):].copy()

    merge_and_save(df_existing, df_new_final, PARQUET_FILE)


# === MAIN ===
def main():
    utility.setup_logging()
    cfg = loader.load_config("config/rules_config.yaml")

    for asset in ["btc", "eth", "sol"]:
        process_asset(asset, cfg)


if __name__ == "__main__":
    main()
