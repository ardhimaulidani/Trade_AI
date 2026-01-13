import os
import json
import time
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone

import utility.yaml_loader as loader
import utility.helpers as utility

# === CONFIG ===
cfg = loader.load_config("config/rules_config.yaml")
DB_FILE = cfg.files.db
STATE_FILE = "config/deriv_state.json"  # Single shared state file

COINS_CONFIG = {
    "BTC": {"symbol": "BTCUSDT", "table": "btc_deriv"},
    "ETH": {"symbol": "ETHUSDT", "table": "eth_deriv"},
    "SOL": {"symbol": "SOLUSDT", "table": "sol_deriv"},
}

utility.setup_logging()


# === STATE HANDLING ===
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    # Default empty state
    return {
        "last_id_btc": None, "last_ts_btc": None,
        "last_id_eth": None, "last_ts_eth": None,
        "last_id_sol": None, "last_ts_sol": None,
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# === FETCH TRADES ===
def fetch_trades(symbol, start_ms, end_ms, last_id=None, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    all_trades = []
    params_base = {"symbol": symbol, "limit": limit}

    cur_start = start_ms
    while cur_start < end_ms:
        params = params_base.copy()
        params["startTime"] = cur_start
        params["endTime"] = end_ms
        if last_id is not None:
            params["fromId"] = last_id + 1

        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    print(f"Rate limited ({symbol}), sleeping 60s...")
                    time.sleep(60)
                    continue
                r.raise_for_status()
                trades = r.json()
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"Error fetching trades ({symbol}): {e}, retry {attempt + 1}/5...")
                time.sleep(wait)
        else:
            print(f"All retries failed for {symbol} trades")
            break

        if not trades:
            break

        all_trades.extend(trades)
        last_trade_time = trades[-1]["T"]
        if last_trade_time >= end_ms - 1000:
            break

        cur_start = last_trade_time + 1
        time.sleep(0.05)

    return all_trades


def process_trades(trades, last_cvd=0.0):
    if not trades:
        return pd.DataFrame(), last_cvd

    df = pd.DataFrame(trades)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["q"] = df["q"].astype(float)
    df["delta"] = df["q"] * (1 - 2 * df["m"].astype(int))

    df = df.set_index("T")[["delta"]]
    # CHANGE: closed='left', label='left' → bins labeled at start time (matches OI)
    delta_15m = df.resample("15min", closed='left', label='left').sum()
    # REMOVED: delta_15m = delta_15m[delta_15m.index >= df.index.min().ceil("15min")]  # This was causing the issue

    cvd_15m = delta_15m.cumsum() + last_cvd

    out = pd.DataFrame({
        "ts": delta_15m.index,
        "delta": delta_15m["delta"].values,
        "cvd": cvd_15m["delta"].values
    })
    return out, float(cvd_15m["delta"].iloc[-1])


# === FETCH OI ===
def fetch_open_interest(symbol, interval, start_ms, end_ms):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": interval, "limit": 500,
              "startTime": start_ms, "endTime": end_ms}
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                print(f"Rate limited (OI {symbol}), sleeping 60s...")
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"Error fetching OI ({symbol}): {e}, retry {attempt + 1}/5...")
            time.sleep(wait)
    print(f"All retries failed for {symbol} OI")
    return []


def process_oi(data, last_oi=None):
    if not data:
        return pd.DataFrame(), last_oi

    df = pd.DataFrame(data)
    # CRITICAL FIX: timestamp is milliseconds!
    df["ts"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
    
    # Optional safety: filter out impossibly early data (in case of API glitches)
    df = df[df["ts"] >= pd.Timestamp('2019-01-01', tz='UTC')]
    
    df["oi"] = pd.to_numeric(df["sumOpenInterest"], errors='coerce')
    df["oi_value"] = pd.to_numeric(df["sumOpenInterestValue"], errors='coerce')

    df = df.sort_values("ts").drop_duplicates("ts")
    df["oi_delta"] = df["oi"].diff()

    if last_oi is not None and not df.empty:
        df.iloc[0, df.columns.get_loc("oi_delta")] = df.iloc[0]["oi"] - last_oi

    df["oi_delta"] = df["oi_delta"].fillna(0.0)

    return df[["ts", "oi", "oi_value", "oi_delta"]], float(df.iloc[-1]["oi"])


# === UPDATE TABLE ===
def update_table(conn, table_name, df, last_cvd=0.0):
    if df.empty:
        return

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ts DATETIME PRIMARY KEY,
            delta REAL,
            cvd REAL,
            oi REAL,
            oi_value REAL,
            oi_delta REAL
        )
    """)

    for _, row in df.iterrows():
        ts = row['ts'].to_pydatetime()
        
        # Handle NaNs
        delta = None if pd.isna(row['delta']) else float(row['delta'])
        cvd = None if pd.isna(row['cvd']) else float(row['cvd'])
        oi = None if pd.isna(row['oi']) else float(row['oi'])
        oi_value = None if pd.isna(row['oi_value']) else float(row['oi_value'])
        oi_delta = None if pd.isna(row['oi_delta']) else float(row['oi_delta'])

        # Check if row exists
        cur = conn.cursor()
        cur.execute(f"SELECT delta, cvd, oi, oi_value, oi_delta FROM {table_name} WHERE ts = ?", (ts,))
        existing = cur.fetchone()

        if existing:
            exist_delta, exist_cvd, exist_oi, exist_oi_value, exist_oi_delta = existing
            
            # For delta: accumulate if new data
            new_delta = (exist_delta or 0.0) + (delta or 0.0) if delta is not None else exist_delta
            
            # For cvd: keep new if provided (already correctly computed), else existing
            new_cvd = cvd if cvd is not None else exist_cvd
            
            # For OI fields: update if new data provided, else keep existing
            new_oi = oi if oi is not None else exist_oi
            new_oi_value = oi_value if oi_value is not None else exist_oi_value
            new_oi_delta = oi_delta if oi_delta is not None else exist_oi_delta
        else:
            new_delta = delta or 0.0  # Default to 0 if None
            new_cvd = cvd or last_cvd  # Use last_cvd if None
            new_oi = oi
            new_oi_value = oi_value
            new_oi_delta = oi_delta

        # Insert or replace
        cur.execute(f"""
            INSERT OR REPLACE INTO {table_name}
            (ts, delta, cvd, oi, oi_value, oi_delta)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ts, new_delta, new_cvd, new_oi, new_oi_value, new_oi_delta))

    conn.commit()


# === MAIN ===
def main():
    state = load_state()
    conn = sqlite3.connect(DB_FILE)

    candle_interval_sec = 15 * 60
    now_sec = time.time()
    current_candle_start_sec = int(now_sec // candle_interval_sec) * candle_interval_sec
    end_ts_ms = current_candle_start_sec * 1000
    end_ts_oi_ms = (current_candle_start_sec - candle_interval_sec) * 1000

    for coin, config in COINS_CONFIG.items():
        symbol = config["symbol"]
        table = config["table"]
        coin_lower = coin.lower()

        print(f"\n{'='*50}")
        print(f"Updating {coin} ({symbol}) → {table}")
        print(f"{'='*50}")

        last_id_key = f"last_id_{coin_lower}"
        last_ts_key = f"last_ts_{coin_lower}"

        last_id = state.get(last_id_key)
        last_ts = state.get(last_ts_key)

        # --- Last CVD from DB ---
        cur = conn.execute(f"SELECT cvd FROM {table} ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        last_cvd = float(row[0]) if row else 0.0

        # --- Last OI and start time from DB ---
        cur = conn.execute(f"SELECT ts, oi FROM {table} ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            last_db_ts_str, last_oi_db = row
            last_db_dt = pd.to_datetime(last_db_ts_str, utc=True).to_pydatetime()

            # Align to end of current 15min candle
            minutes_offset = last_db_dt.minute % 15
            aligned = last_db_dt - timedelta(
                minutes=minutes_offset,
                seconds=last_db_dt.second,
                microseconds=last_db_dt.microsecond
            )
            start_dt = aligned + timedelta(minutes=15)
            start_ts_oi_ms = int(start_dt.timestamp() * 1000)
            last_oi = last_oi_db
        else:
            start_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
            start_ts_oi_ms = int(start_dt.timestamp() * 1000)
            last_oi = None

        # --- Trades start time ---
        start_ts_trades_ms = last_ts + 1 if last_ts else int(datetime(2021, 1, 1).timestamp() * 1000)

        if start_ts_oi_ms > end_ts_oi_ms:
            print(f"No new candles for {coin}")
            continue

        print(f"Fetching up to: {datetime.utcfromtimestamp(current_candle_start_sec)} UTC")

        # Fetch trades
        print("Fetching new trades...")
        trades = fetch_trades(symbol, start_ts_trades_ms, end_ts_ms, last_id=last_id)
        df_trades, new_cvd = process_trades(trades, last_cvd)

        # Fetch OI
        print("Fetching open interest...")
        oi_data = fetch_open_interest(symbol, "15m", start_ts_oi_ms, end_ts_oi_ms)
        df_oi, new_oi = process_oi(oi_data, last_oi)

        # Merge
        df_merged = pd.merge(df_trades, df_oi, on="ts", how="outer").sort_values("ts")

        # FIX: If no new trades but new OI, set delta=0 and cvd=last_cvd (no change)
        if df_trades.empty and not df_oi.empty:
            df_merged["delta"] = 0.0
            df_merged["cvd"] = last_cvd

        if not df_merged.empty:
            update_table(conn, table, df_merged, last_cvd=last_cvd)

            # Update state
            if trades:
                state[last_id_key] = trades[-1]["a"]
            state[last_ts_key] = int(df_merged["ts"].max().timestamp() * 1000)

            save_state(state)
            print(f"Updated {table} with {len(df_merged)} new rows → up to {df_merged['ts'].max()}")
        else:
            print("No new data")

    conn.close()
    print(f"\nAll coins updated successfully!")


if __name__ == "__main__":
    main()