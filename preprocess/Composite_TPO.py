import os, pickle, json
import pandas as pd
import numpy as np
from copy import deepcopy
import utility.yaml_loader as loader
import utility.tpo as tpo

# ----------------- Parameters -----------------
COMPOSITE_OVERLAP_THRESH = 0.7

# ----------------- Select Asset -----------------
# ----------------- Config -----------------
cfg = loader.load_config("config/rules_config.yaml")

ASSETS = ["btc", "eth", "sol"]  # assets supported

def process_asset(asset):
    if asset == "btc":
        PARQUET_IN    = cfg.btc_files.daily
        COMPOSITE_OUT = cfg.btc_files.composite
        CACHE_FILE    = "cache/tpo_daily_cache_btc.pkl"
        SNAPSHOT_FILE = "snapshot_map_btc.pkl"
        MAP_FILE      = "composite_map_btc.pkl"
        STATE_DIR     = "cache"

    elif asset == "eth":
        PARQUET_IN    = cfg.eth_files.daily
        COMPOSITE_OUT = cfg.eth_files.composite
        CACHE_FILE    = "cache/tpo_daily_cache_eth.pkl"
        SNAPSHOT_FILE = "snapshot_map_eth.pkl"
        MAP_FILE      = "composite_map_eth.pkl"
        STATE_DIR     = "cache"

    else:  # sol
        PARQUET_IN    = cfg.sol_files.daily
        COMPOSITE_OUT = cfg.sol_files.composite
        CACHE_FILE    = "cache/tpo_daily_cache_sol.pkl"
        SNAPSHOT_FILE = "snapshot_map_sol.pkl"
        MAP_FILE      = "composite_map_sol.pkl"
        STATE_DIR     = "cache"

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # ----------------- Load data -----------------
    daily_tpo_df = pd.read_parquet(PARQUET_IN)

    # load TPO cache (daily profiles)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            daily_tpo_map = pickle.load(f)
    else:
        raise FileNotFoundError(f"Missing daily TPO cache file for {asset}: {CACHE_FILE}")

    # load previous state (if exists)
    snapshot_map_path = os.path.join(STATE_DIR, SNAPSHOT_FILE)
    composite_map_path = os.path.join(STATE_DIR, MAP_FILE)

    snapshot_map = {}
    composite_map = {}

    if os.path.exists(snapshot_map_path):
        with open(snapshot_map_path, "rb") as f:
            snapshot_map = pickle.load(f)

    if os.path.exists(composite_map_path):
        with open(composite_map_path, "rb") as f:
            composite_map = pickle.load(f)

    # ----------------- Compute composites -----------------
    print(f"Computing composites incrementally for {asset}...")

    completed_days = sorted(daily_tpo_df["date"].unique().tolist())
    snapshot_map_new = {}
    composite_id = 0
    composite_tpo = {}
    composite_edges = {}
    composite_meta = {}

    cur_comp = None
    cur_counts = None
    cur_edges = None

    for day in completed_days:
        entry = daily_tpo_map.get(day)
        if entry is None:
            composite_id += 1
            composite_map[day] = composite_id
            composite_tpo[composite_id] = np.zeros(1, dtype=np.int32)
            composite_edges[composite_id] = np.array([0.0, 1.0])
            composite_meta[composite_id] = {"start_date": day, "end_date": day}
            cur_comp = composite_id
            cur_counts = composite_tpo[cur_comp]
            cur_edges = composite_edges[cur_comp]
            continue

        day_counts = entry["counts"]
        day_edges = entry["edges"]

        row = daily_tpo_df.loc[daily_tpo_df["date"] == day]
        day_val = float(row["VAL_TPO"].values[0]) if "VAL_TPO" in row else np.nan
        day_vah = float(row["VAH_TPO"].values[0]) if "VAH_TPO" in row else np.nan

        # Merge or start new composite
        if cur_comp is None or pd.isna(day_val) or pd.isna(day_vah):
            composite_id += 1
            composite_map[day] = composite_id
            composite_tpo[composite_id] = day_counts.copy()
            composite_edges[composite_id] = day_edges.copy()
            composite_meta[composite_id] = {"start_date": day, "end_date": day}
            cur_comp = composite_id
            cur_counts = composite_tpo[cur_comp]
            cur_edges = composite_edges[cur_comp]
        else:
            comp_min, comp_max = cur_edges[0], cur_edges[-1]
            day_min, day_max = day_edges[0], day_edges[-1]
            new_min, new_max = min(comp_min, day_min), max(comp_max, day_max)
            new_edges = np.arange(new_min, new_max + 1.0, 1.0)
            new_centers = (new_edges[:-1] + new_edges[1:]) / 2.0

            re_cur = tpo.rebin_counts(cur_counts, cur_edges, new_edges)
            re_day = tpo.rebin_counts(day_counts, day_edges, new_edges)

            _, comp_val, comp_vah = tpo.poc_val_vah(re_cur, new_centers)
            _, day_val2, day_vah2 = tpo.poc_val_vah(re_day, new_centers)

            overlap_ratio = 0.0
            if not any(pd.isna([comp_val, comp_vah, day_val2, day_vah2])):
                inter_low = max(comp_val, day_val2)
                inter_high = min(comp_vah, day_vah2)
                inter = max(0.0, inter_high - inter_low)
                min_width = min(comp_vah - comp_val, day_vah2 - day_val2)
                overlap_ratio = (inter / min_width) if min_width > 0 else 0.0

            if overlap_ratio >= COMPOSITE_OVERLAP_THRESH:
                merged = re_cur + re_day
                cur_counts = merged
                cur_edges = new_edges
                composite_tpo[cur_comp] = cur_counts.copy()
                composite_edges[cur_comp] = cur_edges.copy()
                composite_meta[cur_comp]["end_date"] = day
                composite_map[day] = cur_comp
            else:
                composite_id += 1
                composite_map[day] = composite_id
                composite_tpo[composite_id] = day_counts.copy()
                composite_edges[composite_id] = day_edges.copy()
                composite_meta[composite_id] = {"start_date": day, "end_date": day}
                cur_comp = composite_id
                cur_counts = composite_tpo[cur_comp]
                cur_edges = composite_edges[cur_comp]

        # Generate snapshot
        composite_records = []
        for cid, counts in composite_tpo.items():
            edges = composite_edges[cid]
            centers = (edges[:-1] + edges[1:]) / 2.0
            if counts.sum() == 0:
                comp_poc = comp_vah = comp_val = np.nan
                comp_width = np.nan
            else:
                comp_poc, comp_val, comp_vah = tpo.poc_val_vah(counts, centers)
                comp_width = comp_vah - comp_val if pd.notna(comp_vah) and pd.notna(comp_val) else np.nan

            meta = composite_meta.get(cid, {})
            composite_records.append({
                "composite_id": cid,
                "start_date": meta.get("start_date"),
                "end_date": meta.get("end_date"),
                "comp_POC": comp_poc,
                "comp_VAH": comp_vah,
                "comp_VAL": comp_val,
                "comp_width": comp_width
            })

        current_composite_df = pd.DataFrame(composite_records).sort_values("composite_id").reset_index(drop=True)
        snapshot_map_new[day] = {
            "composite_df": current_composite_df.copy(deep=True),
            "composite_map": deepcopy(composite_map)
        }

    # Save composite snapshots
    composite_df = snapshot_map_new[completed_days[-1]]["composite_df"]
    composite_df.to_parquet(COMPOSITE_OUT, index=False)

    with open(snapshot_map_path, "wb") as f:
        pickle.dump(snapshot_map_new, f)

    with open(composite_map_path, "wb") as f:
        pickle.dump(composite_map, f)

    print(f"Composite updated for {asset} up to {completed_days[-1]}")

# ----------------- Run all assets -----------------
for asset in ASSETS:
    process_asset(asset)