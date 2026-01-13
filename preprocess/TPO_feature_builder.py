import numpy as np
import pandas as pd
import os
import pickle
import utility.yaml_loader as loader

# -------------------------
# Load config
# -------------------------
cfg = loader.load_config("config/rules_config.yaml")

ASSETS = ["btc", "eth", "sol"]

def run_for_asset(asset):

    print("\n" + "="*60)
    print(f"Processing asset: {asset}")
    print("="*60)

    # -----------------------------------------------
    # resolve per-asset paths from config
    # -----------------------------------------------
    if asset == "btc":
        PARQUET_IN = cfg.btc_files.vp
        ROLLING_TPO = cfg.btc_files.rollingTPO
        DAILY_TPO = cfg.btc_files.daily
        COMPOSITE_TPO = cfg.btc_files.composite
        PARQUET_OUT = cfg.btc_files.bars

        DAILY_CACHE_FILE = "cache/tpo_daily_cache_btc.pkl"
        COMPOSITE_MAP_FILE = "cache/composite_map_btc.pkl"
        COMPOSITE_SNAPSHOT_FILE = "cache/snapshot_map_btc.pkl"

    elif asset == "eth":
        PARQUET_IN = cfg.eth_files.vp
        ROLLING_TPO = cfg.eth_files.rollingTPO
        DAILY_TPO = cfg.eth_files.daily
        COMPOSITE_TPO = cfg.eth_files.composite
        PARQUET_OUT = cfg.eth_files.bars

        DAILY_CACHE_FILE = "cache/tpo_daily_cache_eth.pkl"
        COMPOSITE_MAP_FILE = "cache/composite_map_eth.pkl"
        COMPOSITE_SNAPSHOT_FILE = "cache/snapshot_map_eth.pkl"

    else:  # sol
        PARQUET_IN = cfg.sol_files.vp
        ROLLING_TPO = cfg.sol_files.rollingTPO
        DAILY_TPO = cfg.sol_files.daily
        COMPOSITE_TPO = cfg.sol_files.composite
        PARQUET_OUT = cfg.sol_files.bars

        DAILY_CACHE_FILE = "cache/tpo_daily_cache_sol.pkl"
        COMPOSITE_MAP_FILE = "cache/composite_map_sol.pkl"
        COMPOSITE_SNAPSHOT_FILE = "cache/snapshot_map_sol.pkl"

    # -------------------------
    # Load All Data
    # -------------------------
    print("Loading parquet...")
    if not os.path.exists(PARQUET_IN):
        raise FileNotFoundError(f"{PARQUET_IN} not found for {asset}")

    bars = pd.read_parquet(PARQUET_IN)
    daily_tpo_df = pd.read_parquet(DAILY_TPO)
    rolling_tpo_df = pd.read_parquet(ROLLING_TPO)
    composite_df = pd.read_parquet(COMPOSITE_TPO)

    # -------------------------
    # Load cache
    # -------------------------
    if os.path.exists(DAILY_CACHE_FILE):
        with open(DAILY_CACHE_FILE, "rb") as f:
            daily_tpo_map = pickle.load(f)

    if os.path.exists(COMPOSITE_MAP_FILE):
        with open(COMPOSITE_MAP_FILE, "rb") as f:
            composite_map = pickle.load(f)
    else:
        composite_map = {}

    if os.path.exists(COMPOSITE_SNAPSHOT_FILE):
        with open(COMPOSITE_SNAPSHOT_FILE, "rb") as f:
            snapshot_map = pickle.load(f)
    else:
        snapshot_map = {}

    # -------------------------
    # Preprocess bar structure
    # -------------------------
    needed = {"ts","open","high","low","close","volume","session","session_date","date"}

    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars["date"] = bars["ts"].dt.date
    bars = bars.sort_values("ts").reset_index(drop=True)

    rolling_tpo_df = rolling_tpo_df.sort_values("ts").reset_index(drop=True)
    rolling_tpo_df["ts"] = pd.to_datetime(rolling_tpo_df["ts"], utc=True)

    daily_tpo_df["date"] = pd.to_datetime(daily_tpo_df["date"]).dt.date
    composite_df["start_date"] = pd.to_datetime(composite_df["start_date"]).dt.date
    composite_df["end_date"] = pd.to_datetime(composite_df["end_date"]).dt.date

    # -------------------------
    # Merge intraday rolling TPO + daily features
    # -------------------------
    bars_with_tpo = bars.merge(
        rolling_tpo_df, on=["ts", "date"], how="left", validate="one_to_one"
    )
    bars_with_tpo = bars_with_tpo.merge(
        daily_tpo_df, on="date", how="left", validate="many_to_one"
    )

    # -------------------------
    # composite columns
    # -------------------------
    bars_with_tpo["comp_POC"] = np.nan
    bars_with_tpo["comp_VAH"] = np.nan
    bars_with_tpo["comp_VAL"] = np.nan
    bars_with_tpo["comp_width"] = np.nan
    bars_with_tpo["comp_id_applied"] = pd.NA
    bars_with_tpo["prev_comp_id"] = pd.NA
    bars_with_tpo["prev_comp_VAH"] = np.nan
    bars_with_tpo["prev_comp_VAL"] = np.nan
    bars_with_tpo["prev_comp_POC"] = np.nan

    # -------------------------
    # Fill composite snapshot data
    # -------------------------
    print("Applying composite snapshot values...")
    bar_dates = pd.to_datetime(bars_with_tpo["date"]).dt.date

    for idx, bar_date in enumerate(bar_dates):
        prev_day = bar_date - pd.Timedelta(days=1)
        prev_day = prev_day.date() if isinstance(prev_day, pd.Timestamp) else prev_day
        snap = snapshot_map.get(prev_day)
        if snap is None:
            continue
        comp_df = snap["composite_df"]
        if comp_df.empty:
            continue

        last_row = comp_df.sort_values("composite_id").iloc[-1]
        bars_with_tpo.at[idx, "comp_POC"] = last_row["comp_POC"]
        bars_with_tpo.at[idx, "comp_VAH"] = last_row["comp_VAH"]
        bars_with_tpo.at[idx, "comp_VAL"] = last_row["comp_VAL"]
        bars_with_tpo.at[idx, "comp_width"] = last_row["comp_width"]
        bars_with_tpo.at[idx, "comp_id_applied"] = int(last_row["composite_id"])

        prev_comp_df = comp_df.iloc[:-1]
        if not prev_comp_df.empty:
            prev_last = prev_comp_df.iloc[-1]
            bars_with_tpo.at[idx, "prev_comp_id"] = int(prev_last["composite_id"])
            bars_with_tpo.at[idx, "prev_comp_VAH"] = prev_last["comp_VAH"]
            bars_with_tpo.at[idx, "prev_comp_VAL"] = prev_last["comp_VAL"]
            bars_with_tpo.at[idx, "prev_comp_POC"] = prev_last["comp_POC"]

    # -------------------------
    # previous VA/POC carry-forward
    # -------------------------
    print("Aligning previous day TPO values...")

    daily_prev = (
        daily_tpo_df[["date", "VAH_TPO", "VAL_TPO", "POC_TPO"]]
        .rename(
            columns={
                "date": "prev_date",
                "VAH_TPO": "prev_VAH",
                "VAL_TPO": "prev_VAL",
                "POC_TPO": "prev_POC",
            }
        )
    )

    bars_with_tpo["prev_date"] = pd.to_datetime(bars_with_tpo["date"]) - pd.Timedelta(days=1)
    bars_with_tpo["prev_date"] = bars_with_tpo["prev_date"].dt.date

    bars_with_tpo = bars_with_tpo.merge(
        daily_prev, on="prev_date", how="left", validate="many_to_one"
    )

    # -------------------------
    # Opening type & Value migration
    # -------------------------
    print("Flagging opening type and migration value...")

    bars_dt = bars.copy()
    bars_dt["date"] = pd.to_datetime(bars_dt["date"]).dt.date

    day_first_open = (
        bars_dt.sort_values("ts").groupby("date", observed=True)["open"].first().rename("day_open")
    )

    bars_with_tpo["date"] = pd.to_datetime(bars_with_tpo["date"]).dt.date
    bars_with_tpo = bars_with_tpo.merge(
        day_first_open.reset_index(), on="date", how="left", validate="many_to_one"
    )

    bars_with_tpo["day_open"] = pd.to_numeric(bars_with_tpo["day_open"], errors="coerce")
    bars_with_tpo["prev_VAL"] = pd.to_numeric(bars_with_tpo.get("prev_VAL"), errors="coerce")
    bars_with_tpo["prev_VAH"] = pd.to_numeric(bars_with_tpo.get("prev_VAH"), errors="coerce")

    bars_with_tpo["opening_type"] = "unknown"
    mask_has_prev = (
        bars_with_tpo["day_open"].notna()
        & bars_with_tpo["prev_VAL"].notna()
        & bars_with_tpo["prev_VAH"].notna()
    )
    mask_below = mask_has_prev & (bars_with_tpo["day_open"] < bars_with_tpo["prev_VAL"])
    mask_above = mask_has_prev & (bars_with_tpo["day_open"] > bars_with_tpo["prev_VAH"])
    mask_invalue = mask_has_prev & ~(mask_below | mask_above)

    bars_with_tpo.loc[mask_below, "opening_type"] = "below_value_open"
    bars_with_tpo.loc[mask_above, "opening_type"] = "above_value_open"
    bars_with_tpo.loc[mask_invalue, "opening_type"] = "in_value_open"

    # Value Migration
    v_prev_vah = pd.to_numeric(bars_with_tpo.get("prev_VAH"), errors="coerce")
    v_prev_val = pd.to_numeric(bars_with_tpo.get("prev_VAL"), errors="coerce")
    v_intraday_vah = pd.to_numeric(bars_with_tpo.get("VAH_TPO_intraday"), errors="coerce")
    v_intraday_val = pd.to_numeric(bars_with_tpo.get("VAL_TPO_intraday"), errors="coerce")

    bars_with_tpo["value_migration"] = "unknown"
    mask_has_mig = (
        v_prev_vah.notna() & v_prev_val.notna() & v_intraday_vah.notna() & v_intraday_val.notna()
    )
    mask_higher = mask_has_mig & (v_intraday_vah > v_prev_vah) & (v_intraday_val > v_prev_val)
    mask_lower = mask_has_mig & (v_intraday_vah < v_prev_vah) & (v_intraday_val < v_prev_val)
    mask_overlap = mask_has_mig & ~(mask_higher | mask_lower)

    bars_with_tpo.loc[mask_higher, "value_migration"] = "higher_value"
    bars_with_tpo.loc[mask_lower, "value_migration"] = "lower_value"
    bars_with_tpo.loc[mask_overlap, "value_migration"] = "overlapping_value"

    # -------------------------
    # 7-Day Rolling Virgin SP
    # -------------------------
    print("Computing 7-day rolling virgin single prints...")

    sp_map = {}
    for _, row in daily_tpo_df.iterrows():
        date = pd.to_datetime(row["date"]).date()
        sp_str = str(row.get("single_prints", "")).strip()
        if not sp_str:
            continue
        try:
            prices = [float(x) for x in sp_str.split(",") if x]
        except Exception:
            prices = []
        if prices:
            sp_map[date] = prices

    bars_by_date = {
        d: df[["low", "high"]].to_numpy()
        for d, df in bars_with_tpo.groupby("date")
    }

    def is_swept(sp_price, from_date, to_date):
        if to_date <= from_date + pd.Timedelta(days=1):
            return False
        start = from_date + pd.Timedelta(days=1)
        end = to_date - pd.Timedelta(days=1)
        for d in pd.date_range(start, end, freq="D"):
            day = d.date()
            if day not in bars_by_date:
                continue
            arr = bars_by_date[day]
            if np.any((arr[:, 0] <= sp_price) & (arr[:, 1] >= sp_price)):
                return True
        return False

    rolling_window = 7
    virgin_sp_per_day = {}

    all_dates = np.sort(bars_with_tpo["date"].unique())
    for bar_date in all_dates:
        window_start = bar_date - pd.Timedelta(days=rolling_window)
        window_end = bar_date - pd.Timedelta(days=1)

        candidates = []
        for d in pd.date_range(window_start, window_end, freq="D"):
            day = d.date()
            if day in sp_map:
                for sp in sp_map[day]:
                    if not is_swept(sp, from_date=day, to_date=bar_date):
                        candidates.append(sp)

        virgin_sp_per_day[bar_date] = ",".join(f"{x:.2f}" for x in sorted(set(candidates))) if candidates else ""

    bars_with_tpo["7d_rSP"] = bars_with_tpo["date"].map(virgin_sp_per_day)

    # -------------------------
    # Save final parquet
    # -------------------------
    bars_with_tpo.to_parquet(PARQUET_OUT, index=False)
    print(f"Saved for {asset}: {PARQUET_OUT}")


# ===============================
# RUN ALL ASSETS
# ===============================
for a in ASSETS:
    run_for_asset(a)

print("\nAll assets completed ✓")
