import pandas as pd

# Load your parquet
df = pd.read_parquet("database/parquet/event_trades.parquet")

# Ensure datetime
df["entry_ts"] = pd.to_datetime(df["entry_ts"])
# define range ( – July 2025)
start   = pd.Timestamp("2023-01-01", tz="UTC")

# filter only 2025 trades before cutoff + A/S tiers
df = df[
    (df["entry_ts"] >= start)
].sort_values("entry_ts").reset_index(drop=True)

# Define win condition (assuming "target_hit" means win)
df["is_win"] = (df["realized_rr"] > 0).astype(int)

# Define the main triggers list
main_triggers = [
    "cVAH_breakout", "cVAH_reject",
    "cVAL_breakout", "cVAL_reject",
    "ML", "MH","nML", "nMH",
    "nPDH", "nPDL",
    "PWH", "PWL", "nPWH", "nPWL",
    "PMH", "PML", "nPMH", "nPML"
]

# Expand multiple triggers into separate rows
df_expanded = (
    df.assign(main_trigger=df["main_trigger"].str.split(",\s*"))
      .explode("main_trigger")
)

# Keep only the triggers we care about
df_expanded = df_expanded[df_expanded["main_trigger"].isin(main_triggers)]

# --- Calls per main_trigger per tier ---
calls_per_trigger_tier = (
    df_expanded.groupby(["main_trigger", "tier"])
    .size()
    .unstack(fill_value=0)
    .reindex(main_triggers)
    .reindex(columns=["S","A","B","C"], fill_value=0)   # fixed order
)

# --- Winrate per main_trigger per tier ---
winrate_per_trigger_tier = (
    df_expanded.groupby(["main_trigger", "tier"])["is_win"]
    .mean()
    .unstack(fill_value=0)
    .reindex(columns=["S","A","B","C"], fill_value=0) * 100
)

# --- Winrate per main_trigger per tier ---
calls1_per_trigger_tier = (
    df_expanded.groupby(["direction", "tier"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["S","A","B","C"], fill_value=0)   # fixed order
)

winrate_per_direction_tier = (
    df_expanded.groupby(["direction", "tier"])["is_win"]
    .mean()
    .unstack(fill_value=0)
    .reindex(columns=["S","A","B","C"], fill_value=0) * 100
)

# --- Combined summary table ---
summary_trigger_tier = pd.concat(
    {"Calls": calls_per_trigger_tier, "Winrate %": winrate_per_trigger_tier},
    axis=1
)

summary1_trigger_tier = pd.concat(
    {"Calls": calls1_per_trigger_tier, "Winrate %": winrate_per_direction_tier},
    axis=1
)

print(summary_trigger_tier)
print(summary1_trigger_tier)