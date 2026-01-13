import os
import json
import pandas as pd
import numpy as np

class BarProcessor:
    def __init__(self, cfg):
        """
        cfg = load_config("config.yaml")
        """
        self.cfg = cfg
        self.state = self.load_last_state()

    def load_data(self, asset):
        print("Loading data...")
        # --- choose asset parquet target & DB table ---
        if asset == "btc":
            self.files = self.cfg.btc_files
        elif asset == "eth":
            self.files = self.cfg.eth_files
        else:  # sol
            self.files = self.cfg.sol_files

        self.deriv = pd.read_parquet(self.files.deriv)
        self.bars = pd.read_parquet(self.files.bars)
        self.daily = pd.read_parquet(self.files.daily)
        self.comps = pd.read_parquet(self.files.composite)

    def preprocess_bars(self):
        bars, daily, comps, deriv = self.bars, self.daily, self.comps, self.deriv

        # normalize timestamps
        bars["ts"] = pd.to_datetime(bars["ts"])
        bars = bars.sort_values("ts").reset_index(drop=True)
        bars["date"] = bars["ts"].dt.date

        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        comps["start_date"] = pd.to_datetime(comps["start_date"]).dt.date
        comps["end_date"] = pd.to_datetime(comps["end_date"]).dt.date
        comps["comp_length"] = (pd.to_datetime(comps["end_date"]) - pd.to_datetime(comps["start_date"])).dt.days + 1

        # map composite lengths
        comp_len_map = comps.set_index("composite_id")["comp_length"].to_dict()
        bars["comp_length"] = bars["comp_id_applied"].map(comp_len_map)

        # === merge z-scores from deriv ===
        bars = bars.merge(
            deriv[["ts","oi_delta","oi_delta_1h","oi_delta_4h","oi_z16","oi_z96",
                   "oi_delta_z16","cvd","delta","cvd_z16","cvd_z96","delta_z16"]],
            on="ts", how="left"
        )

        # weekly info
        bars["week"] = bars["ts"].dt.to_period("W").apply(lambda r: r.start_time.date())
        bars["weekday"] = bars["ts"].dt.weekday

        # --- monday highs/lows + timestamp ---
        monday_frames = bars[bars["weekday"] == 0].groupby("week").agg(
            monday_high=("high", "max"),
            monday_high_idx=("high", lambda x: x.idxmax()),
            monday_low=("low", "min"),
            monday_low_idx=("low", lambda x: x.idxmin())
        ).reset_index()

        # Shift Monday highs/lows one week forward so that:
        # - On Monday itself, we still use previous week's Monday values
        # - From Tuesday onward, we use the most recent completed Monday
        weeks = sorted(monday_frames["week"].unique())
        monday_high_map, monday_low_map = {}, {}
        monday_high_idx_map, monday_low_idx_map = {}, {}

        for i, w in enumerate(weeks):
            if i == 0:
                monday_high_map[w] = np.nan
                monday_low_map[w] = np.nan
                monday_high_idx_map[w] = np.nan
                monday_low_idx_map[w] = np.nan
            else:
                prev_w = weeks[i - 1]
                prev_row = monday_frames[monday_frames["week"] == prev_w].iloc[0]
                monday_high_map[w] = prev_row["monday_high"]
                monday_low_map[w] = prev_row["monday_low"]
                monday_high_idx_map[w] = prev_row["monday_high_idx"]
                monday_low_idx_map[w] = prev_row["monday_low_idx"]

        # Assign the mapped values
        bars["monday_high"] = bars["week"].map(monday_high_map)
        bars["monday_low"] = bars["week"].map(monday_low_map)
        bars["monday_high_idx"] = bars["week"].map(monday_high_idx_map)
        bars["monday_low_idx"] = bars["week"].map(monday_low_idx_map)


        # --- previous week highs/lows + index ---
        bars = bars.reset_index().rename(columns={"index":"orig_index"})
        week_stats = bars.groupby("week").apply(
            lambda g: pd.Series({
                "week_high": g["high"].max(),
                "week_low": g["low"].min(),
                "week_high_idx": int(g.loc[g["high"].idxmax(), "orig_index"]),
                "week_low_idx": int(g.loc[g["low"].idxmin(), "orig_index"]),
            })
        ).reset_index()

        prev_high_map, prev_low_map = {}, {}
        prev_high_idx_map, prev_low_idx_map = {}, {}
        weeks = sorted(week_stats["week"].unique())

        for i, w in enumerate(weeks):
            if i == 0:
                prev_high_map[w] = prev_low_map[w] = np.nan
                prev_high_idx_map[w] = prev_low_idx_map[w] = np.nan
            else:
                prev = weeks[i-1]
                row = week_stats[week_stats["week"] == prev].iloc[0]
                prev_high_map[w] = row["week_high"]
                prev_low_map[w] = row["week_low"]
                prev_high_idx_map[w] = row["week_high_idx"]
                prev_low_idx_map[w] = row["week_low_idx"]

        bars["prev_week_high"] = bars["week"].map(prev_high_map)
        bars["prev_week_low"] = bars["week"].map(prev_low_map)
        bars["prev_week_high_idx"] = bars["week"].map(prev_high_idx_map)
        bars["prev_week_low_idx"] = bars["week"].map(prev_low_idx_map)

        # --- previous month highs/lows + index ---
        month_highlow = bars.groupby("month").agg(
            month_high=("high", "max"),
            month_low=("low", "min")
        ).reset_index()

        prev_month_high_map, prev_month_low_map = {}, {}
        prev_month_high_idx_map, prev_month_low_idx_map = {}, {}
        months = sorted(bars["month"].unique())

        for i, m in enumerate(months):
            if i == 0:
                prev_month_high_map[m] = prev_month_low_map[m] = np.nan
                prev_month_high_idx_map[m] = prev_month_low_idx_map[m] = np.nan
            else:
                prev = months[i-1]
                # previous month high/low value
                row = month_highlow[month_highlow["month"] == prev]
                prev_month_high_map[m] = row["month_high"].values[0]
                prev_month_low_map[m] = row["month_low"].values[0]

                # index should be first bar of current month
                first_idx_current_month = bars.index[bars["month"] == m][0]
                prev_month_high_idx_map[m] = first_idx_current_month
                prev_month_low_idx_map[m] = first_idx_current_month

        bars["prev_month_high"] = bars["month"].map(prev_month_high_map)
        bars["prev_month_low"] = bars["month"].map(prev_month_low_map)
        bars["prev_month_high_idx"] = bars["month"].map(prev_month_high_idx_map)
        bars["prev_month_low_idx"] = bars["month"].map(prev_month_low_idx_map)

        # daily highs/lows
        daily_highlow = bars.groupby("date").agg(
            daily_high=("high", "max"),
            daily_low=("low", "min")
        ).reset_index().set_index("date").to_dict(orient="index")

        # --- previous day highs/lows + index ---
        # Compute daily highs/lows and get first index of each date
        day_stats = bars.reset_index().groupby("date").apply(
            lambda g: pd.Series({
                "day_high": g["high"].max(),
                "day_low": g["low"].min(),
                "day_high_idx": int(g.loc[g["high"].idxmax(), "index"]),
                "day_low_idx": int(g.loc[g["low"].idxmin(), "index"]),
            })
        ).reset_index()

        # Build previous day maps
        prev_day_high_map, prev_day_low_map = {}, {}
        prev_day_high_idx_map, prev_day_low_idx_map = {}, {}
        dates = sorted(day_stats["date"].unique())

        for i, d in enumerate(dates):
            if i == 0:
                prev_day_high_map[d] = np.nan
                prev_day_low_map[d] = np.nan
                prev_day_high_idx_map[d] = np.nan
                prev_day_low_idx_map[d] = np.nan
            else:
                prev = dates[i - 1]
                row = day_stats[day_stats["date"] == prev].iloc[0]
                prev_day_high_map[d] = row["day_high"]
                prev_day_low_map[d] = row["day_low"]
                prev_day_high_idx_map[d] = row["day_high_idx"]
                prev_day_low_idx_map[d] = row["day_low_idx"]

        # Assign to bars
        bars["prev_day_high"] = bars["date"].map(prev_day_high_map)
        bars["prev_day_low"] = bars["date"].map(prev_day_low_map)
        bars["prev_day_high_idx"] = bars["date"].map(prev_day_high_idx_map)
        bars["prev_day_low_idx"] = bars["date"].map(prev_day_low_idx_map)

        # --- today_idx: index of first bar at/after UTC 00:00 of current day ---
        # Make a UTC-aware series from bars["ts"] (handles tz-naive and tz-aware)
        if bars["ts"].dt.tz is None:
            utc_ts = bars["ts"].dt.tz_localize("UTC")
        else:
            utc_ts = bars["ts"].dt.tz_convert("UTC")

        # Midnight (00:00) UTC of today
        today_utc_midnight = pd.Timestamp.now(tz="UTC").normalize()

        # Find first index where utc_ts >= midnight UTC
        mask_today = utc_ts >= today_utc_midnight
        if mask_today.any():
            # idxmax returns the first True index (integer index in current bars)
            today_idx = int(mask_today.idxmax())
            self.today_idx = today_idx
            # Integrate into DataFrame: repeated value and boolean marker
            bars["today_idx"] = today_idx
            # mark the exact row that starts today (True only at that row)
            bars["is_today_start"] = False
            bars.at[today_idx, "is_today_start"] = True
        else:
            # no bars for today available: store NA consistently
            self.today_idx = None
            bars["today_idx"] = pd.NA
            bars["is_today_start"] = False

        # save
        self.bars = bars
        self.daily_highlow = daily_highlow
        self.comp_len_map = comp_len_map

    def get_new_indices(self):
        last_ts = self.state["last_ts"]
        if last_ts is None:
            return list(self.bars.index)  # first run: process everything
        return self.bars.index[self.bars["ts"] > pd.to_datetime(last_ts)].tolist()

    def save_state(self):
        # --- Update state with latest ts ---
        new_last_ts = self.bars["ts"].iloc[-1].isoformat()
        with open(self.cfg.files.state, "w") as f:
            json.dump({"last_ts": new_last_ts}, f)

    def prepare_globals(self):
        bars = self.bars
        ts_list = bars["ts"].tolist()
        open_list = bars["open"].astype(float).tolist()
        high_list = bars["high"].astype(float).tolist()
        low_list = bars["low"].astype(float).tolist()
        close_list = bars["close"].astype(float).tolist()

        meta_cols = [
            "session","7d_rSP","POC_TPO_intraday","VAH_TPO_intraday","VAL_TPO_intraday",
            "POC_TPO","VAH_TPO","VAL_TPO","comp_POC","comp_VAH","comp_VAL","comp_length",
            "VWAP_7d","VWAP_30d","VWAP_90d","VWAP_month_anchored","VWAP_prev_month_final",
            "VWAP_month_upper_1std", "VWAP_month_lower_1std", 
            "VWAP_prev_month_upper_1std", "VWAP_prev_month_lower_1std",
            "VWAP_quarter_anchored","VWAP_prev_quarter_final",
            "VWAP_quarter_upper_1std", "VWAP_quarter_lower_1std", 
            "VWAP_prev_quarter_upper_1std", "VWAP_prev_quarter_lower_1std",
            "monday_high","monday_low", "prev_day_high", "prev_day_low",
            "prev_week_high","prev_week_low","prev_month_high","prev_month_low",
            "monday_high_idx","monday_low_idx","today_idx","prev_day_high_idx","prev_day_low_idx",
            "prev_week_high_idx","prev_week_low_idx","prev_month_high_idx", "prev_month_low_idx",
            "volume_zscore_20","volume_mad_20","volume_zscore_60","volume_mad_60",
            "session_high","session_low","IB_high","IB_low","IB_range",
            "opening_type","value_migration","high","low",
            "oi_delta","oi_delta_1h","oi_delta_4h","oi_z16","oi_z96","oi_delta_z16",
            "cvd","delta","cvd_z16","cvd_z96","delta_z16",
            "price_delta_z16","price_delta_z32","price_delta_pct",
            "prev_comp_id","prev_comp_VAH","prev_comp_VAL","prev_comp_POC"
        ]
        rows_meta = bars[meta_cols].to_dict(orient="records")

        return {
            'ts_list': ts_list,
            'open_list':open_list,
            'high_list': high_list,
            'low_list': low_list,
            'close_list': close_list,
            'rows_meta': rows_meta,
            'index_map': {ts: i for i, ts in enumerate(ts_list)},
            'daily_highlow': dict(self.daily_highlow),
            'comp_len_map': self.comp_len_map,
            'bars_len': len(ts_list)
        }

    def load_last_state(self):
        if os.path.exists(self.cfg.files.state):
            with open(self.cfg.files.state, "r") as f:
                return json.load(f)
        return {"last_ts": None}
