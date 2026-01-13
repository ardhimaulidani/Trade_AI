import os
import pickle
import numpy as np
import pandas as pd
import utility.yaml_loader as loader
import utility.tpo as tpo

# ----------------- Config -----------------
cfg = loader.load_config("config/rules_config.yaml")

ASSETS = ["btc", "eth", "sol"]  # assets supported


def process_asset(asset):

    # -------- resolve parquet paths --------
    if asset == "btc":
        PARQUET_IN = cfg.btc_files.vp        
        DAILY_PARQUET_OUT = cfg.btc_files.daily
    elif asset == "eth":
        PARQUET_IN = cfg.eth_files.vp
        DAILY_PARQUET_OUT = cfg.eth_files.daily
    else:  # sol
        PARQUET_IN = cfg.sol_files.vp
        DAILY_PARQUET_OUT = cfg.sol_files.daily

    CACHE_FILE = f"cache/tpo_daily_cache_{asset}.pkl"

    print(f"\n==== Computing DAILY TPO for {asset.upper()} ====")
    print(f"Input VP parquet : {PARQUET_IN}")
    print(f"Output DAILY TPO  : {DAILY_PARQUET_OUT}")
    print(f"Cache             : {CACHE_FILE}")

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

    # ----------------- Load bars -----------------
    if not os.path.exists(PARQUET_IN):
        print(f"⚠ No parquet for {asset}, skipping")
        return

    bars = pd.read_parquet(PARQUET_IN)
    bars["ts"] = pd.to_datetime(bars["ts"]).dt.tz_localize(None)
    bars = bars.sort_values("ts").reset_index(drop=True)
    bars["date"] = bars["ts"].dt.date

    if len(bars) == 0:
        print("No bars found, skipping.")
        return

    # ----------------- Load cache -----------------
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            daily_tpo_map = pickle.load(f)
    else:
        daily_tpo_map = {}

    # ----------------- Completed days -----------------
    print("Computing daily TPO for completed days...")

    days = np.sort(bars["date"].unique())
    latest_day = days[-1]        # incomplete
    completed_days = days[:-1]   # only finished days

    daily_records = []

    for day in completed_days:

        entry = daily_tpo_map.get(day)
        if entry is None:
            # if missing we simply skip — your upstream builder fills cache
            continue

        counts = np.array(entry["counts"])
        centers = np.array(entry["centers"])

        total = counts.sum()

        if total == 0:
            poc = vah = val = np.nan
            va_width = np.nan
        else:
            poc, val, vah = tpo.poc_val_vah(counts, centers)
            va_width = vah - val if pd.notna(vah) and pd.notna(val) else np.nan

        # -------- single prints --------
        sp_idx = np.where(counts == 1)[0]
        sp_idx = sp_idx[sp_idx < len(centers)]
        sp_levels = np.floor(centers[sp_idx])
        sp_levels = np.unique(sp_levels).tolist() if sp_idx.size > 0 else []

        tick = 1.0
        group_avgs = []

        if sp_levels:
            group = [sp_levels[0]]
            for lvl in sp_levels[1:]:
                if abs(lvl - group[-1] - tick) < 1e-6:
                    group.append(lvl)
                else:
                    group_avgs.append(np.mean(group))
                    group = [lvl]
            group_avgs.append(np.mean(group))

        sp_str = ",".join([f"{p:.2f}" for p in group_avgs]) if group_avgs else ""

        daily_records.append({
            "asset": asset.upper(),
            "date": day,
            "tpo_total": int(total),
            "POC_TPO": poc,
            "VAH_TPO": vah,
            "VAL_TPO": val,
            "VA_width": va_width,
            "single_prints": sp_str
        })

    # -------- save parquet --------
    daily_tpo_df = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(DAILY_PARQUET_OUT), exist_ok=True)
    daily_tpo_df.to_parquet(DAILY_PARQUET_OUT, index=False)

    print(f"Saved {len(daily_tpo_df)} completed days to {DAILY_PARQUET_OUT}")


# ----------------- Run all assets -----------------
for asset in ASSETS:
    process_asset(asset)
