import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import utility.yaml_loader as loader
import utility.tpo as tpo

# =========================
# CONFIG
# =========================
cfg = loader.load_config("config/rules_config.yaml")

ASSETS = ["btc", "eth", "sol"]


# =========================
# HELPER: expand edges safely
# =========================
def expand_edges_counts(old_edges, old_centers, old_counts, new_low, new_high, tick=0.25):
    new_edges, new_centers = tpo.make_bins(new_low - 100, new_high + 100, tick=tick)
    new_counts = np.zeros(len(new_centers), dtype=int)

    idx_map = np.searchsorted(new_centers, old_centers)
    for i, idx in enumerate(idx_map):
        if 0 <= idx < len(new_centers):
            if idx > 0 and abs(new_centers[idx - 1] - old_centers[i]) < abs(new_centers[idx] - old_centers[i]):
                idx -= 1
            new_counts[idx] += old_counts[i]
    return new_edges, new_centers, new_counts


# =========================
# MAIN PER-ASSET FUNCTION
# =========================
def process_asset(asset):

    # --------------------------
    # resolve parquet paths
    # --------------------------
    if asset == "btc":
        PARQUET_IN = cfg.btc_files.vp
        PARQUET_OUT = cfg.btc_files.rollingTPO
    elif asset == "eth":
        PARQUET_IN = cfg.eth_files.vp
        PARQUET_OUT = cfg.eth_files.rollingTPO
    else:  # sol
        PARQUET_IN = cfg.sol_files.vp
        PARQUET_OUT = cfg.sol_files.rollingTPO

    CACHE_FILE = f"cache/tpo_daily_cache_{asset}.pkl"

    print(f"\n==== Processing {asset.upper()} ====")
    print(f"Input  : {PARQUET_IN}")
    print(f"Output : {PARQUET_OUT}")
    print(f"Cache  : {CACHE_FILE}")

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

    # =========================
    # LOAD INPUT BARS
    # =========================
    if not os.path.exists(PARQUET_IN):
        print(f"⚠ No parquet for {asset}, skipping.")
        return

    bars = pd.read_parquet(PARQUET_IN)
    bars["ts"] = pd.to_datetime(bars["ts"]).dt.tz_localize(None)
    bars = bars.sort_values("ts").reset_index(drop=True)
    bars["date"] = bars["ts"].dt.date

    # =========================
    # LOAD / INIT CACHE
    # =========================
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            daily_tpo_map = pickle.load(f)
    else:
        daily_tpo_map = {}

    # =========================
    # LOAD EXISTING PARQUET
    # =========================
    if os.path.exists(PARQUET_OUT):
        existing_out_df = pd.read_parquet(PARQUET_OUT)
        existing_out_df["ts"] = pd.to_datetime(existing_out_df["ts"]).dt.tz_localize(None)
        last_ts = existing_out_df["ts"].max()
        rebuild_all = False
    else:
        existing_out_df = pd.DataFrame(columns=["ts", "date",
                                                "POC_TPO_intraday",
                                                "VAL_TPO_intraday",
                                                "VAH_TPO_intraday"])
        last_ts = pd.Timestamp.min
        rebuild_all = True

    # =========================
    # DETERMINE NEW BARS
    # =========================
    def get_day_last_ts(day):
        if day in daily_tpo_map:
            return daily_tpo_map[day].get("last_ts", pd.Timestamp.min)
        return pd.Timestamp.min

    if rebuild_all:
        new_bars = bars.copy()
    else:
        new_parts = []
        for day in bars["date"].unique():
            last_day_ts = get_day_last_ts(day)
            part = bars[(bars["date"] == day) & (bars["ts"] > last_day_ts)]
            if not part.empty:
                new_parts.append(part)
        new_bars = pd.concat(new_parts) if new_parts else pd.DataFrame(columns=bars.columns)

    if new_bars.empty:
        print("No new bars to process.")
        return

    # =========================
    # MAIN PROCESSING LOOP
    # =========================
    new_snapshots = []

    for day in tqdm(sorted(new_bars["date"].unique()), desc=f"Rolling TPO {asset.upper()}"):
        day_bars_all = bars[bars["date"] == day].sort_values("ts").reset_index(drop=True)
        day_new = new_bars[new_bars["date"] == day].sort_values("ts").reset_index(drop=True)
        if day_new.empty:
            continue

        # Restore or initialize cache
        if day in daily_tpo_map and not rebuild_all:
            edges   = np.array(daily_tpo_map[day]["edges"])
            centers = np.array(daily_tpo_map[day]["centers"])
            counts  = np.array(daily_tpo_map[day]["counts"])
        else:
            day_low, day_high = day_bars_all["low"].min(), day_bars_all["high"].max()
            edges, centers = tpo.make_bins(day_low - 100, day_high + 100, tick=0.25)
            counts = np.zeros(len(centers), dtype=int)

        # Process only new bars
        for _, row in day_new.iterrows():
            row_low, row_high = float(row["low"]), float(row["high"])

            if row_low < edges[0] or row_high > edges[-1]:
                edges, centers, counts = expand_edges_counts(
                    edges, centers, counts, row_low, row_high, tick=0.25
                )

            counts = tpo.update_counts(counts, row, edges)
            poc, val, vah = tpo.poc_val_vah(counts, centers)

            new_snapshots.append({
                "ts": row["ts"],
                "date": day,
                "asset": asset.upper(),
                "POC_TPO_intraday": poc,
                "VAL_TPO_intraday": val,
                "VAH_TPO_intraday": vah
            })

            daily_tpo_map[day] = {
                "counts": counts,
                "edges": edges,
                "centers": centers,
                "last_ts": row["ts"]
            }

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(daily_tpo_map, f)

    # =========================
    # SAVE CACHE + PARQUET
    # =========================
    if new_snapshots:
        new_df = pd.DataFrame(new_snapshots)
        new_df["ts"] = pd.to_datetime(new_df["ts"])
    else:
        new_df = pd.DataFrame(columns=["ts", "date", "asset",
                                       "POC_TPO_intraday",
                                       "VAL_TPO_intraday",
                                       "VAH_TPO_intraday"])

    if rebuild_all:
        combined_df = new_df.copy()
    else:
        combined_df = pd.concat([existing_out_df, new_df], ignore_index=True)

    combined_df = combined_df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    combined_df.to_parquet(PARQUET_OUT, index=False)

    print(f"Saved {len(combined_df)} total snapshots for {asset.upper()}.")


# =========================
# RUN FOR ALL ASSETS
# =========================
for asset in ASSETS:
    process_asset(asset)
