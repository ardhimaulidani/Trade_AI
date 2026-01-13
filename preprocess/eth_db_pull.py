import pandas as pd
import sqlite3
import time
import logging
import os
from datetime import datetime

from pricehub import get_ohlc
import utility.yaml_loader as loader
import utility.helpers as utility

# === CONFIG ===
cfg = loader.load_config("config/rules_config.yaml")
DB_FILE = cfg.files.db

RAW_TABLE = "sol_15m_ohlcv"
CLEAN_TABLE = "sol_15m_ohlcv_clean"
TEMP_CSV_DIR = "temp_sol_data"

BROKERS = {
    "bybit_spot": "SOLUSDT",
    "binance_spot": "SOLUSDT",
    "coinbase_spot": "SOL-USD"
}

# Start date: January 1, 2023
START_DATE = "2023-01-01"

utility.setup_logging()

# === HELPER FUNCTIONS ===
def ensure_temp_dir():
    """Create temporary directory for CSV files if needed"""
    if not os.path.exists(TEMP_CSV_DIR):
        os.makedirs(TEMP_CSV_DIR)
        logging.info(f"Created temporary directory: {TEMP_CSV_DIR}")

def cleanup_temp_files():
    """Remove temporary CSV files and directory"""
    if os.path.exists(TEMP_CSV_DIR):
        for file in os.listdir(TEMP_CSV_DIR):
            os.remove(os.path.join(TEMP_CSV_DIR, file))
        os.rmdir(TEMP_CSV_DIR)
        logging.info(f"Cleaned up temporary directory: {TEMP_CSV_DIR}")

def fetch_with_retry(broker, symbol, start_ts, end_ts, max_retries=5, base_sleep=2):
    """Fetch OHLC data with exponential backoff retry"""
    import requests
    import urllib3
    
    attempt = 0
    while True:
        try:
            df = get_ohlc(
                broker=broker,
                symbol=symbol,
                interval="15m",
                start=start_ts,
                end=end_ts
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
            
            return df
        
        except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, ValueError) as e:
            attempt += 1
            sleep_time = base_sleep * (2 ** (attempt - 1))
            logging.warning(f"Attempt {attempt} failed for {broker}: {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            if attempt >= max_retries:
                logging.error(f"Max retries reached for {broker}. Raising error.")
                raise e

# === CONNECT DB ===
conn = sqlite3.connect(DB_FILE)

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

# === FETCH DATA ===
start_ts = int(time.mktime(time.strptime(START_DATE, "%Y-%m-%d")))
candle_interval = 15 * 60
now = time.time()
end_ts = int(now // candle_interval * candle_interval)

# Try to fetch all data at once first
USE_YEARLY_BATCHES = False

try:
    logging.info("Attempting to fetch all data at once...")
    
    for broker, symbol in BROKERS.items():
        logging.info(f"Fetching {broker} ({symbol})...")
        
        end_ts_use = end_ts - candle_interval if broker == "bybit_spot" else end_ts - candle_interval + 1
        
        df = fetch_with_retry(broker, symbol, start_ts, end_ts_use * 1000)
        
        # Try to insert directly
        df.to_sql(RAW_TABLE, conn, if_exists="append", index=False)
        logging.info(f"Inserted {len(df)} rows for {broker}.")
    
    logging.info("Successfully fetched all data in one go!")

except MemoryError:
    logging.warning("Memory error encountered. Switching to yearly batch mode...")
    USE_YEARLY_BATCHES = True
    
    # Clear any partial data
    conn.execute(f"DELETE FROM {RAW_TABLE}")
    conn.commit()

# === YEARLY BATCH MODE ===
if USE_YEARLY_BATCHES:
    ensure_temp_dir()
    
    # Generate year ranges from 2023 to current year
    start_year = 2023
    current_year = datetime.now().year
    
    for broker, symbol in BROKERS.items():
        logging.info(f"Processing {broker} ({symbol}) in yearly batches...")
        
        for year in range(start_year, current_year + 1):
            # Define year boundaries
            year_start = f"{year}-01-01"
            if year == current_year:
                year_end_ts = end_ts
            else:
                year_end_ts = int(time.mktime(time.strptime(f"{year+1}-01-01", "%Y-%m-%d")))
            
            year_start_ts = int(time.mktime(time.strptime(year_start, "%Y-%m-%d")))
            year_end_ts_use = year_end_ts - candle_interval if broker == "bybit_spot" else year_end_ts - candle_interval + 1
            
            logging.info(f"  Fetching {year} data...")
            
            df_year = fetch_with_retry(broker, symbol, year_start_ts, year_end_ts_use * 1000)
            
            # Save to temporary CSV
            csv_file = os.path.join(TEMP_CSV_DIR, f"{broker}_{year}.csv")
            df_year.to_csv(csv_file, index=False)
            logging.info(f"  Saved {len(df_year)} rows to {csv_file}")
    
    # Now load all CSVs and append to database
    logging.info("Loading all CSV files into database...")
    for csv_file in sorted(os.listdir(TEMP_CSV_DIR)):
        csv_path = os.path.join(TEMP_CSV_DIR, csv_file)
        df_batch = pd.read_csv(csv_path, parse_dates=["ts"])
        df_batch.to_sql(RAW_TABLE, conn, if_exists="append", index=False)
        logging.info(f"Loaded {csv_file} into database")
    
    cleanup_temp_files()

# === PATCH / CLEAN DATA ===
logging.info("Starting data cleaning process...")

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
conn.close()

logging.info(f"Updated and cleaned ETH data saved to '{CLEAN_TABLE}'")
logging.info(f"Total rows in clean table: {len(df_clean)}")