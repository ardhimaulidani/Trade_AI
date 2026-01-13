import pandas as pd
import sqlite3
import logging
from datetime import datetime

import utility.yaml_loader as loader
import utility.helpers as utility

# === CONFIG ===
cfg = loader.load_config("config/rules_config.yaml")
DB_FILE = cfg.files.db

utility.setup_logging()

# === CONNECT DB ===
conn = sqlite3.connect(DB_FILE)

print("="*80)
print("DATABASE DATA QUALITY ANALYSIS")
print("="*80)

# === ANALYZE BOTH TABLES ===
for asset, table in [("BTC", "btc_15m_ohlcv_clean"), ("ETH", "eth_15m_ohlcv_clean")]:
    print(f"\n{'='*80}")
    print(f"ANALYZING {asset} DATA ({table})")
    print(f"{'='*80}")
    
    # Get basic stats
    total_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", conn)["cnt"].values[0]
    print(f"\nTotal rows: {total_rows:,}")
    
    # Get exchange breakdown
    exchange_stats = pd.read_sql(f"""
        SELECT exchange, COUNT(*) as row_count,
               MIN(ts) as first_date,
               MAX(ts) as last_date
        FROM {table}
        GROUP BY exchange
        ORDER BY exchange
    """, conn)
    
    print(f"\n{'-'*80}")
    print("EXCHANGE BREAKDOWN:")
    print(f"{'-'*80}")
    for _, row in exchange_stats.iterrows():
        print(f"\n{row['exchange']}:")
        print(f"  Rows: {row['row_count']:,}")
        print(f"  First date: {row['first_date']}")
        print(f"  Last date: {row['last_date']}")
    
    # Check for duplicates
    print(f"\n{'-'*80}")
    print("CHECKING FOR DUPLICATES:")
    print(f"{'-'*80}")
    
    duplicates = pd.read_sql(f"""
        SELECT ts, exchange, symbol, COUNT(*) as dup_count
        FROM {table}
        GROUP BY ts, exchange, symbol
        HAVING COUNT(*) > 1
    """, conn)
    
    if len(duplicates) > 0:
        print(f"⚠️  Found {len(duplicates)} duplicate timestamps!")
        print("\nFirst 10 duplicates:")
        print(duplicates.head(10).to_string(index=False))
    else:
        print("✓ No duplicates found")
    
    # Check for missing timestamps (gaps) in each exchange
    print(f"\n{'-'*80}")
    print("CHECKING FOR MISSING TIMESTAMPS (GAPS):")
    print(f"{'-'*80}")
    
    exchanges = exchange_stats['exchange'].tolist()
    
    for exchange in exchanges:
        df = pd.read_sql(f"""
            SELECT ts FROM {table}
            WHERE exchange='{exchange}'
            ORDER BY ts
        """, conn, parse_dates=['ts'])
        
        if df.empty:
            print(f"\n{exchange}: No data")
            continue
        
        # Expected number of 15-min intervals
        first_ts = df['ts'].min()
        last_ts = df['ts'].max()
        expected_range = pd.date_range(first_ts, last_ts, freq='15min')
        expected_count = len(expected_range)
        actual_count = len(df)
        missing_count = expected_count - actual_count
        
        print(f"\n{exchange}:")
        print(f"  Expected timestamps: {expected_count:,}")
        print(f"  Actual timestamps: {actual_count:,}")
        print(f"  Missing: {missing_count:,} ({missing_count/expected_count*100:.2f}%)")
        
        if missing_count > 0:
            # Find actual gaps
            df_set = set(df['ts'])
            expected_set = set(expected_range)
            missing_ts = sorted(expected_set - df_set)
            
            # Group consecutive missing timestamps
            gaps = []
            if missing_ts:
                gap_start = missing_ts[0]
                gap_end = missing_ts[0]
                
                for i in range(1, len(missing_ts)):
                    if (missing_ts[i] - missing_ts[i-1]).total_seconds() == 900:  # 15 min
                        gap_end = missing_ts[i]
                    else:
                        gaps.append((gap_start, gap_end, 
                                   int((gap_end - gap_start).total_seconds() / 900) + 1))
                        gap_start = missing_ts[i]
                        gap_end = missing_ts[i]
                
                gaps.append((gap_start, gap_end,
                           int((gap_end - gap_start).total_seconds() / 900) + 1))
            
            print(f"  Number of gap periods: {len(gaps)}")
            if len(gaps) > 0:
                print(f"\n  Largest gaps:")
                gaps_sorted = sorted(gaps, key=lambda x: x[2], reverse=True)
                for gap_start, gap_end, gap_size in gaps_sorted[:5]:
                    print(f"    {gap_start} to {gap_end}: {gap_size} missing candles")

# === CROSS-EXCHANGE COMPARISON ===
print(f"\n{'='*80}")
print("CROSS-EXCHANGE TIMESTAMP ALIGNMENT")
print(f"{'='*80}")

for asset, table in [("BTC", "btc_15m_ohlcv_clean"), ("ETH", "eth_15m_ohlcv_clean")]:
    print(f"\n{asset} Data:")
    print(f"{'-'*80}")
    
    exchanges = pd.read_sql(f"SELECT DISTINCT exchange FROM {table}", conn)['exchange'].tolist()
    
    # Get all timestamps for each exchange
    exchange_timestamps = {}
    for exchange in exchanges:
        ts_df = pd.read_sql(f"""
            SELECT DISTINCT ts FROM {table}
            WHERE exchange='{exchange}'
        """, conn, parse_dates=['ts'])
        exchange_timestamps[exchange] = set(ts_df['ts'])
    
    # Find common and unique timestamps
    if len(exchanges) >= 2:
        # Find intersection (timestamps present in ALL exchanges)
        common_ts = set.intersection(*exchange_timestamps.values())
        print(f"\nTimestamps present in ALL exchanges: {len(common_ts):,}")
        
        # Find timestamps unique to each exchange
        for exchange in exchanges:
            unique_ts = exchange_timestamps[exchange] - set.union(
                *[exchange_timestamps[e] for e in exchanges if e != exchange]
            )
            if len(unique_ts) > 0:
                print(f"\nTimestamps unique to {exchange}: {len(unique_ts):,}")
                unique_list = sorted(list(unique_ts))[:10]
                print(f"  First 10: {[str(ts) for ts in unique_list]}")

# === SUMMARY COMPARISON ===
print(f"\n{'='*80}")
print("BTC vs ETH COMPARISON")
print(f"{'='*80}")

btc_count = pd.read_sql("SELECT COUNT(*) as cnt FROM btc_15m_ohlcv_clean", conn)["cnt"].values[0]
eth_count = pd.read_sql("SELECT COUNT(*) as cnt FROM eth_15m_ohlcv_clean", conn)["cnt"].values[0]

print(f"\nBTC total rows: {btc_count:,}")
print(f"ETH total rows: {eth_count:,}")
print(f"Difference: {abs(btc_count - eth_count):,} rows ({abs(btc_count - eth_count)/btc_count*100:.2f}%)")

# Get date ranges
btc_dates = pd.read_sql("SELECT MIN(ts) as min_date, MAX(ts) as max_date FROM btc_15m_ohlcv_clean", conn).iloc[0]
eth_dates = pd.read_sql("SELECT MIN(ts) as min_date, MAX(ts) as max_date FROM eth_15m_ohlcv_clean", conn).iloc[0]

print(f"\nBTC date range: {btc_dates['min_date']} to {btc_dates['max_date']}")
print(f"ETH date range: {eth_dates['min_date']} to {eth_dates['max_date']}")

conn.close()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")