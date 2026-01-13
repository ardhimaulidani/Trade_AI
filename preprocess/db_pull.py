import pandas as pd
import sqlite3
import time
import logging

from pricehub import get_ohlc
import utility.yaml_loader as loader
import utility.helpers as utility

# === CONFIG ===
cfg = loader.load_config("config/rules_config.yaml")
DB_FILE = cfg.files.db

# Define assets and their configurations
ASSETS = {
    "BTC": {
        "raw_table": "btc_15m_ohlcv",
        "clean_table": "btc_15m_ohlcv_clean",
        "brokers": {
            "bybit_spot": "BTCUSDT",
            "binance_spot": "BTCUSDT",
            "coinbase_spot": "BTC-USD"
        }
    },
    "ETH": {
        "raw_table": "eth_15m_ohlcv",
        "clean_table": "eth_15m_ohlcv_clean",
        "brokers": {
            "bybit_spot": "ETHUSDT",
            "binance_spot": "ETHUSDT",
            "coinbase_spot": "ETH-USD"
        }
    },
    "SOL": {
        "raw_table": "sol_15m_ohlcv",
        "clean_table": "sol_15m_ohlcv_clean",
        "brokers": {
            "bybit_spot": "SOLUSDT",
            "binance_spot": "SOLUSDT",
            "coinbase_spot": "SOL-USD"
        }
    }
}

MAX_RETRIES = 5  # maximum attempts before raising
BASE_SLEEP = 2   # seconds, base for exponential backoff

utility.setup_logging()

# === CONNECT DB ===
conn = sqlite3.connect(DB_FILE)

# === PROCESS EACH ASSET ===
for asset_name, asset_config in ASSETS.items():
    logging.info(f"\n{'='*80}")
    logging.info(f"PROCESSING {asset_name}")
    logging.info(f"{'='*80}")
    
    RAW_TABLE = asset_config["raw_table"]
    CLEAN_TABLE = asset_config["clean_table"]
    BROKERS = asset_config["brokers"]
    
    # Ensure raw table exists
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
        ts DATETIME NOT NULL,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        PRIMARY KEY (ts, exchange, symbol)
    );
    """)
    conn.commit()
    
    # === FETCH LATEST DATA FOR EACH BROKER ===
    import requests
    import urllib3
    
    for broker, symbol in BROKERS.items():
        logging.info(f"Fetching {broker} ({symbol})...")
    
        # Determine last timestamp
        last_ts_row = pd.read_sql(f"""
            SELECT MAX(ts) as last_ts FROM {RAW_TABLE} WHERE exchange='{broker}' AND symbol='{symbol}'
        """, conn)
        last_ts = last_ts_row["last_ts"].values[0]
    
        if pd.isna(last_ts):
            start_ts = int(time.mktime(time.strptime("2023-01-01", "%Y-%m-%d")))
            logging.info(f"No existing data. Starting from 2023-01-01")
        else:
            start_ts = int(pd.Timestamp(last_ts).timestamp()) + 15*60
            logging.info(f"Last timestamp: {last_ts}. Fetching from {pd.Timestamp(start_ts, unit='s')}")
    
        candle_interval = 15 * 60
        now = time.time()
        end_ts = int(now // candle_interval * candle_interval)
        end_ts_use = end_ts - candle_interval if broker == "bybit_spot" else end_ts - candle_interval + 1
    
        if start_ts > end_ts_use:
            logging.info("No new data needed.")
            continue
    
        # --- RETRY LOOP ---
        attempt = 0
        while True:
            try:
                df = get_ohlc(
                    broker=broker,
                    symbol=symbol,
                    interval="15m",
                    start=start_ts,
                    end=end_ts_use*1000
                )
                if df.empty:
                    logging.info(f"No OHLCV data retrieved for {broker}. Retrying...")
                    raise ValueError("Empty dataframe")
                
                # Reset index if timestamp is in index
                df = df.reset_index()
    
                # Rename columns to standard
                col_map = {
                    "Open time": "ts",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
                # Convert timestamp to datetime
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
                # Keep required columns
                df = df[["ts", "open", "high", "low", "close", "volume"]]
    
                # Add exchange & symbol
                df["exchange"] = broker
                df["symbol"] = symbol
    
                # Drop duplicates
                df = df.drop_duplicates(subset=["ts", "exchange", "symbol"])
                
                # Append to raw table
                df.to_sql(RAW_TABLE, conn, if_exists="append", index=False)
                logging.info(f"Inserted {len(df)} rows for {broker}.")
                
                break  # success
            
            except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, ValueError) as e:
                attempt += 1
                sleep_time = BASE_SLEEP * (2 ** (attempt - 1))  # exponential backoff
                logging.warning(f"Attempt {attempt} failed for {broker}: {e}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                if attempt >= MAX_RETRIES:
                    logging.error(f"Max retries reached for {broker}. Raising error.")
                    raise e
    
    # === PATCH / CLEAN DATA ===
    logging.info(f"\nCleaning {asset_name} data...")
    
    exchanges = pd.read_sql(f"SELECT DISTINCT exchange, symbol FROM {RAW_TABLE};", conn)
    all_clean = []
    
    for _, row in exchanges.iterrows():
        exch, sym = row["exchange"], row["symbol"]
        logging.info(f"Processing {exch} ({sym})...")
    
        df = pd.read_sql(f"""
            SELECT ts, open, high, low, close, volume
            FROM {RAW_TABLE}
            WHERE exchange='{exch}' AND symbol='{sym}'
            ORDER BY ts
        """, conn, parse_dates=["ts"])
    
        if df.empty:
            continue
    
        # Build full 15m timeline
        full_range = pd.date_range(df["ts"].min(), df["ts"].max(), freq="15min")
        df = df.set_index("ts").reindex(full_range)
    
        # Forward fill missing closes
        df["close"] = df["close"].ffill()
    
        # Fill OHLC from last close if missing
        for col in ["open", "high", "low"]:
            df[col] = df[col].fillna(df["close"])
    
        # Missing volume = 0
        df["volume"] = df["volume"].fillna(0)
    
        # Reset index
        df = df.reset_index().rename(columns={"index": "ts"})
    
        # Add exchange + symbol
        df["exchange"] = exch
        df["symbol"] = sym
    
        all_clean.append(df)
    
    # Merge everything
    df_clean = pd.concat(all_clean, ignore_index=True)
    
    # Save into clean table
    df_clean.to_sql(CLEAN_TABLE, conn, if_exists="replace", index=False)
    
    logging.info(f"Updated and cleaned {asset_name} data saved to '{CLEAN_TABLE}'")
    logging.info(f"Total rows in {CLEAN_TABLE}: {len(df_clean):,}")

conn.close()

logging.info(f"\n{'='*80}")
logging.info("ALL ASSETS UPDATED SUCCESSFULLY")
logging.info(f"{'='*80}")