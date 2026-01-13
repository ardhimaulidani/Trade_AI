import pandas as pd
import sqlite3
import logging

import utility.yaml_loader as loader
import utility.helpers as utility

# === CONFIG ===
cfg = loader.load_config("config/rules_config.yaml")
DB_FILE = cfg.files.db

CLEAN_TABLE = "btc_15m_ohlcv_clean"
BACKUP_TABLE = "btc_15m_ohlcv_clean_backup"

utility.setup_logging()

# === CONNECT DB ===
conn = sqlite3.connect(DB_FILE)

print("="*80)
print("FIXING BTC DATABASE - CONSOLIDATING DUPLICATE EXCHANGES")
print("="*80)

# === BACKUP ORIGINAL TABLE ===
print("\n1. Creating backup...")
try:
    conn.execute(f"DROP TABLE IF EXISTS {BACKUP_TABLE}")
    conn.execute(f"CREATE TABLE {BACKUP_TABLE} AS SELECT * FROM {CLEAN_TABLE}")
    conn.commit()
    print(f"✓ Backup created: {BACKUP_TABLE}")
except Exception as e:
    print(f"✗ Backup failed: {e}")
    conn.close()
    exit(1)

# === ANALYZE CURRENT STATE ===
print("\n2. Analyzing current data...")
exchange_info = pd.read_sql(f"""
    SELECT exchange, 
           COUNT(*) as row_count,
           MIN(ts) as first_date,
           MAX(ts) as last_date
    FROM {CLEAN_TABLE}
    GROUP BY exchange
    ORDER BY exchange
""", conn)

print("\nCurrent exchanges:")
print(exchange_info.to_string(index=False))

# === STRATEGY ===
print("\n3. Consolidation strategy:")
print("   - Keep all data from exchanges WITHOUT '_spot' suffix (2023-01-01 to 2025-09-05)")
print("   - Keep all data from exchanges WITH '_spot' suffix (2024-09-30 to 2026-01-04)")
print("   - Result: Complete timeline from 2023-01-01 to 2026-01-04 for each exchange")

# Mapping old exchange names to new ones
EXCHANGE_MAP = {
    'binance': 'binance_spot',
    'bybit': 'bybit_spot',
    'coinbase': 'coinbase_spot'
}

# === CONSOLIDATE DATA ===
print("\n4. Consolidating data...")

consolidated_data = []

for old_name, new_name in EXCHANGE_MAP.items():
    print(f"\n   Processing {old_name} -> {new_name}...")
    
    # Get old exchange data (should be 2023-01-01 to 2025-09-05)
    df_old = pd.read_sql(f"""
        SELECT ts, open, high, low, close, volume, symbol
        FROM {CLEAN_TABLE}
        WHERE exchange = '{old_name}'
        ORDER BY ts
    """, conn, parse_dates=['ts'])
    
    # Get new exchange data (should be 2024-09-30 to 2026-01-04)
    df_new = pd.read_sql(f"""
        SELECT ts, open, high, low, close, volume, symbol
        FROM {CLEAN_TABLE}
        WHERE exchange = '{new_name}'
        ORDER BY ts
    """, conn, parse_dates=['ts'])
    
    if df_old.empty and df_new.empty:
        print(f"   ⚠️  No data found for {old_name} or {new_name}")
        continue
    
    if df_old.empty:
        print(f"   Using only {new_name} data: {len(df_new):,} rows")
        df_consolidated = df_new.copy()
    elif df_new.empty:
        print(f"   Using only {old_name} data: {len(df_old):,} rows")
        df_consolidated = df_old.copy()
    else:
        # Find the cutoff point - use the max date from old data
        cutoff_date = df_old['ts'].max()
        print(f"   Old data: {len(df_old):,} rows (up to {cutoff_date})")
        print(f"   New data: {len(df_new):,} rows (from {df_new['ts'].min()})")
        
        # Keep all old data
        # Keep only new data AFTER the cutoff
        df_new_filtered = df_new[df_new['ts'] > cutoff_date]
        
        # Combine
        df_consolidated = pd.concat([df_old, df_new_filtered], ignore_index=True)
        df_consolidated = df_consolidated.sort_values('ts').reset_index(drop=True)
        
        print(f"   ✓ Consolidated: {len(df_consolidated):,} rows ({len(df_old):,} old + {len(df_new_filtered):,} new)")
    
    # Set exchange to the new name
    df_consolidated['exchange'] = new_name
    
    # Extract symbol if it exists, otherwise use default
    if 'symbol' in df_consolidated.columns:
        symbol = df_consolidated['symbol'].iloc[0] if not df_consolidated.empty else 'BTCUSDT'
    else:
        symbol = 'BTCUSDT' if 'binance' in new_name or 'bybit' in new_name else 'BTC-USD'
        df_consolidated['symbol'] = symbol
    
    consolidated_data.append(df_consolidated)

# === CREATE NEW CLEAN TABLE ===
print("\n5. Creating new consolidated table...")

if consolidated_data:
    df_final = pd.concat(consolidated_data, ignore_index=True)
    
    # Drop the old table and create new one
    conn.execute(f"DROP TABLE IF EXISTS {CLEAN_TABLE}")
    df_final.to_sql(CLEAN_TABLE, conn, if_exists='replace', index=False)
    conn.commit()
    
    print(f"✓ New table created with {len(df_final):,} total rows")
    
    # Verify the result
    print("\n6. Verification:")
    new_exchange_info = pd.read_sql(f"""
        SELECT exchange, 
               COUNT(*) as row_count,
               MIN(ts) as first_date,
               MAX(ts) as last_date
        FROM {CLEAN_TABLE}
        GROUP BY exchange
        ORDER BY exchange
    """, conn)
    
    print("\nNew exchange breakdown:")
    print(new_exchange_info.to_string(index=False))
    
    # Check for gaps
    print("\n7. Checking for gaps...")
    for _, row in new_exchange_info.iterrows():
        exchange = row['exchange']
        df_check = pd.read_sql(f"""
            SELECT ts FROM {CLEAN_TABLE}
            WHERE exchange = '{exchange}'
            ORDER BY ts
        """, conn, parse_dates=['ts'])
        
        first_ts = df_check['ts'].min()
        last_ts = df_check['ts'].max()
        expected_range = pd.date_range(first_ts, last_ts, freq='15min')
        expected_count = len(expected_range)
        actual_count = len(df_check)
        missing = expected_count - actual_count
        
        print(f"\n   {exchange}:")
        print(f"   Expected: {expected_count:,} | Actual: {actual_count:,} | Missing: {missing:,}")
        
        if missing == 0:
            print(f"   ✓ No gaps!")
        else:
            print(f"   ⚠️  {missing:,} timestamps missing ({missing/expected_count*100:.2f}%)")
    
    # Final summary
    total_final = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {CLEAN_TABLE}", conn)['cnt'].values[0]
    total_backup = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {BACKUP_TABLE}", conn)['cnt'].values[0]
    
    print("\n" + "="*80)
    print("CONSOLIDATION COMPLETE")
    print("="*80)
    print(f"Original rows: {total_backup:,}")
    print(f"New rows: {total_final:,}")
    print(f"Difference: {total_backup - total_final:,} rows removed (duplicates)")
    print(f"\nBackup table '{BACKUP_TABLE}' preserved in case you need to revert.")
    print("="*80)

else:
    print("✗ No data to consolidate!")

conn.close()