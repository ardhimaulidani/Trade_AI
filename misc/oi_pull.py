import os
import time
import requests
import zipfile
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime, timedelta

# ===============================
# CONFIG
# ===============================
SYMBOL = "BTCUSDT"
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 9, 25)  # yesterday
OUT_DB = "database/btc_historical.db"
OUT_DIR = "data_metrics"
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

# ===============================
# CHECKSUM / DOWNLOAD
# ===============================
def get_remote_sha256(symbol: str, date):
    date_str = date.strftime("%Y-%m-%d")
    checksum_url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip.CHECKSUM"
    try:
        r = requests.get(checksum_url, timeout=30)
        if r.status_code != 200:
            return None
        return r.text.split()[0]
    except Exception as e:
        print("⚠️ Error fetching checksum:", e)
        return None

def verify_sha256(file_path, expected_sha256):
    if expected_sha256 is None:
        return True
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256

def download_zip_file(symbol: str, date, out_dir=OUT_DIR):
    date_str = date.strftime("%Y-%m-%d")
    fname = f"{symbol}-metrics-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{fname}"
    local_path = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)

    sha256_remote = get_remote_sha256(symbol, date)
    if os.path.exists(local_path) and sha256_remote and verify_sha256(local_path, sha256_remote):
        return local_path

    try:
        print(f"Downloading {url} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                print(f"⚠️ File not found: {url} (HTTP {r.status_code})")
                return None
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        if sha256_remote and not verify_sha256(local_path, sha256_remote):
            print(f"⚠️ SHA-256 mismatch, deleting {local_path}")
            os.remove(local_path)
            return None
        return local_path
    except Exception as e:
        print(f"⚠️ Download error for {date_str}: {e}")
        return None

def download_zip_file_retry(symbol: str, date, max_retries=3, delay=5):
    for attempt in range(1, max_retries + 1):
        p = download_zip_file(symbol, date)
        if p:
            return p
        print(f"⚠️ Attempt {attempt} failed for {date.strftime('%Y-%m-%d')}, retrying in {delay}s ...")
        time.sleep(delay)
    return None

# ===============================
# PARSE AND RESAMPLE
# ===============================
def parse_metrics_zip(path: str):
    """Parse a metrics zip and return a DataFrame with columns:
       ts (datetime UTC), sum_open_interest, sum_open_interest_value
    """
    try:
        with zipfile.ZipFile(path, "r") as z:
            name = z.namelist()[0]  # one CSV per zip
            with z.open(name) as f:
                df = pd.read_csv(f)
    except zipfile.BadZipFile:
        print(f"⚠️ Bad zip file: {path}")
        return pd.DataFrame(columns=["ts","sum_open_interest","sum_open_interest_value"])
    except Exception as e:
        print(f"⚠️ Error reading zip {path}: {e}")
        return pd.DataFrame(columns=["ts","sum_open_interest","sum_open_interest_value"])

    # Detect time column
    time_candidates = ["create_time", "timestamp", "ts", "time", "date", "datetime"]
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # fallback: assume first column is time-like
        time_col = df.columns[0]
        print(f"⚠️ Time column not standard; using '{time_col}' as time column")

    # Convert to datetime
    try:
        # If numeric unix ms -> use unit='ms'
        if pd.api.types.is_integer_dtype(df[time_col]) or pd.api.types.is_float_dtype(df[time_col]):
            df["ts"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
        else:
            # parse strings, enforce UTC
            df["ts"] = pd.to_datetime(df[time_col], utc=True)
    except Exception:
        # last-resort: generic parse
        df["ts"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    # Keep only necessary columns (be defensive if some aren't present)
    keep = {}
    if "sum_open_interest" in df.columns:
        keep["sum_open_interest"] = pd.to_numeric(df["sum_open_interest"], errors="coerce")
    else:
        keep["sum_open_interest"] = pd.Series(dtype="float64")

    if "sum_open_interest_value" in df.columns:
        keep["sum_open_interest_value"] = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
    else:
        keep["sum_open_interest_value"] = pd.Series(dtype="float64")

    out = pd.DataFrame(keep)
    out["ts"] = df["ts"].values
    out = out.dropna(subset=["ts"]).set_index("ts").sort_index()

    return out

def resample_to_15m(df):
    """Input df indexed by timestamp with columns sum_open_interest and sum_open_interest_value.
       Returns DataFrame with ts, oi, oi_value, oi_delta at 15-minute frequency.
    """
    if df.empty:
        return pd.DataFrame(columns=["ts","oi","oi_value","oi_delta"])

    # Make sure index has UTC tz
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    oi_15 = df["sum_open_interest"].resample("15T").last().ffill()
    oi_value_15 = df["sum_open_interest_value"].resample("15T").last().ffill()
    oi_delta = oi_15.diff().fillna(0)

    res = pd.DataFrame({
        "ts": oi_15.index,   # already UTC
        "oi": oi_15.values,
        "oi_value": oi_value_15.values,
        "oi_delta": oi_delta.values
    })
    return res


# ===============================
# SAVE TO SQLITE (no duplicates)
# ===============================
def save_to_db(df, db_path=OUT_DB, table="btc_oi"):
    if df.empty:
        return
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        # Create table if not exists with ts as primary key (ISO text)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                ts TEXT PRIMARY KEY,
                oi REAL,
                oi_value REAL,
                oi_delta REAL
            );
        """)
        rows = []
        for _, r in df.iterrows():
            ts_iso = r["ts"].isoformat()
            oi = None if pd.isna(r["oi"]) else float(r["oi"])
            oi_value = None if pd.isna(r["oi_value"]) else float(r["oi_value"])
            oi_delta = None if pd.isna(r["oi_delta"]) else float(r["oi_delta"])
            rows.append((ts_iso, oi, oi_value, oi_delta))
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} (ts, oi, oi_value, oi_delta) VALUES (?, ?, ?, ?)",
            rows
        )

# ===============================
# MAIN PIPELINE
# ===============================
def process_date(date):
    zip_path = download_zip_file_retry(SYMBOL, date)
    if not zip_path:
        print(f"⚠️ Skipped {date.date()} (file missing)")
        return False
    df_daily = parse_metrics_zip(zip_path)
    resampled = resample_to_15m(df_daily)
    save_to_db(resampled)
    print(f"✅ Processed {date.date()} -> {len(resampled)} rows (15m)")
    return True

def main():
    date = START_DATE
    while date <= END_DATE:
        try:
            process_date(date)
        except Exception as e:
            print(f"⚠️ Error processing {date.date()}: {e}")
        date += timedelta(days=1)
    print("🎯 Finished processing all days!")

if __name__ == "__main__":
    main()