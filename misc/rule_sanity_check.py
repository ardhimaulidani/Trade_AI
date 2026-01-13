import pandas as pd

# load trades
event_trades = pd.read_parquet("database/parquet/event_trades.parquet")
event_trades["entry_ts"] = pd.to_datetime(event_trades["entry_ts"], utc=True)

# define range ( – July 2025)
start   = pd.Timestamp("2025-11-1", tz="UTC")

# filter only 2025 trades before cutoff + A/S tiers
filtered = event_trades[
    (event_trades["entry_ts"] >= start) &
    (event_trades["tier"].isin(["S+", "S", "A"]))
]

# take random sample
if len(filtered) > 25:
    sample = filtered.sample(n=25, random_state=42)
else: 
    sample = filtered

# select useful columns
cols = [
    "entry_ts","exit_ts","entry_price","target","direction",
    "main_trigger","tier","raw_score","realized_rr","outcome", "flag_countertrade"
]

# only include existing columns
cols = [c for c in cols if c in sample.columns]

# show the sample with reasons
print(sample[cols])
