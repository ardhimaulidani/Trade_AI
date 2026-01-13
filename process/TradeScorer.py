# ------------------------
# tradeScorer.py
# ------------------------
import pandas as pd
import numpy as np
import utility.helpers as utility

class TradeScorer:
    def __init__(self, G, index, cfg):
        """
        cfg = load_config("config.yaml")
        """
        self.G = G
        self.index = index
        self.prev_index = max(0, self.index - 1)

        self.ts = G['ts_list']
        self.open = float(G['open_list'][index])
        self.close = float(G['close_list'][index])
        self.high = float(G['high_list'][index])
        self.low = float(G['low_list'][index])

        self.prev_open = float(G['open_list'][self.prev_index])
        self.prev_close = float(G['close_list'][self.prev_index])
        self.prev_high = float(G['high_list'][self.prev_index])
        self.prev_low = float(G['low_list'][self.prev_index])

        self.row = G['rows_meta'][index]  # dict of other columns

        # Config
        self.cfg = cfg
        self.pts = cfg.params.pts
        self.sessionWeightCfg = cfg.params.session_weight

        if self.index > 102200:
            import pickle, os
            debug = {
                'index': self.index,
                'ts': str(self.G['ts_list'][self.index]),
                'session': self.row.get("session", ""),
                'session_vah': self.row.get("session_high", np.nan),
                'session_val': self.row.get("session_low", np.nan),
                'ib_high': self.row.get("IB_high", np.nan),
                'ib_low': self.row.get("IB_low", np.nan),
                'opening_type': self.row.get("opening_type", ""),
                'value_migration': self.row.get("value_migration", ""),
                'v7': self.row.get("VWAP_7d", np.nan),
                'v30': self.row.get("VWAP_30d", np.nan),
                'v90': self.row.get("VWAP_90d", np.nan),
                'vol_z20': self.row.get("volume_zscore_20", 0),
                'vol_z60': self.row.get("volume_zscore_60", 0),
                'vol_mad20': self.row.get("volume_mad_20", 0),
                'vol_mad60': self.row.get("volume_mad_60", 0),
                'cvd': self.row.get("cvd", 0),
                'cvd_delta': self.row.get("delta", 0),
                'oi_delta': self.row.get("oi_delta", 0),
                'cvd_z16': self.row.get("cvd_z16", 0),
                'oi_z16': self.row.get("oi_z16", 0),
                'oi_delta_1h': self.row.get("oi_delta_1h", 0),
                'mvwap': self.row.get("VWAP_month_anchored", np.nan),
                'pmvwap': self.row.get("VWAP_prev_month_final", np.nan),
                'qvwap': self.row.get("VWAP_quarter_anchored", np.nan),
                'pqvwap': self.row.get("VWAP_prev_quarter_final", np.nan),
                'prev_cvah': self.row.get("prev_comp_VAH", np.nan),
                'prev_cval': self.row.get("prev_comp_VAL", np.nan),
                'comp_vah': self.row.get("comp_VAH", np.nan),
                'comp_val': self.row.get("comp_VAL", np.nan),
                'comp_len': self.row.get("comp_length", 0),
                'sp': utility.parse_single_prints(self.row.get("7d_rSP", "")),
                'monday_high': self.row.get("monday_high", np.nan),
                'monday_low': self.row.get("monday_low", np.nan),
                'prev_day_high': self.row.get("prev_day_high", np.nan),
                'prev_day_low': self.row.get("prev_day_low", np.nan),
                'prev_week_high': self.row.get("prev_week_high", np.nan),
                'prev_week_low': self.row.get("prev_week_low", np.nan),
                'prev_month_high': self.row.get("prev_month_high", np.nan),
                'prev_month_low': self.row.get("prev_month_low", np.nan)
            }
            os.makedirs("debug/tradescorer", exist_ok=True)
            with open(f"debug/tradescorer/idx_{self.index}.pkl", "wb") as f:
                pickle.dump(debug, f)

    def get_keylevel(self):
        # Session
        self.current_ts = self.row.get("ts", "")
        self.session = self.row.get("session", "")
        self.session_vah = self.row.get("session_high", np.nan)
        self.session_val = self.row.get("session_low", np.nan)
        self.ib_high = self.row.get("IB_high", np.nan)
        self.ib_low = self.row.get("IB_low", np.nan)
        self.opening_type = self.row.get("opening_type", "")
        self.value_migration = self.row.get("value_migration", "")
        
        # Time Price Opportunity
        self.sp_list = utility.parse_single_prints(self.row.get("7d_rSP", ""))
        self.poc = self.row.get("POC_TPO_intraday", np.nan)
        self.vah = self.row.get("VAH_TPO_intraday", np.nan)
        self.val = self.row.get("VAL_TPO_intraday", np.nan)

        # Composite TPO
        self.prev_cvah = self.row.get("prev_comp_VAH", np.nan)
        self.prev_cval = self.row.get("prev_comp_VAL", np.nan)
        self.comp_poc = self.row.get("comp_POC", np.nan)
        self.comp_vah = self.row.get("comp_VAH", np.nan)
        self.comp_val = self.row.get("comp_VAL", np.nan)
        self.comp_len = self.row.get("comp_length", 0)

        # Rolling VWAP
        self.v7 = self.row.get("VWAP_7d", np.nan)
        self.v30 = self.row.get("VWAP_30d", np.nan)
        self.v90 = self.row.get("VWAP_90d", np.nan)

        # Anchored VWAP
        self.mvwap = self.row.get("VWAP_month_anchored", np.nan)
        self.stdu_mvwap = self.row.get("VWAP_month_upper_1std", np.nan)
        self.stdl_mvwap = self.row.get("VWAP_month_lower_1std", np.nan)

        self.pmvwap = self.row.get("VWAP_prev_month_final", np.nan)
        self.stdu_pmvwap = self.row.get("VWAP_prev_month_upper_1std", np.nan)
        self.stdl_pmvwap = self.row.get("VWAP_prev_month_lower_1std", np.nan)

        self.qvwap = self.row.get("VWAP_quarter_anchored", np.nan)
        self.stdu_qvwap = self.row.get("VWAP_quarter_upper_1std", np.nan)
        self.stdl_qvwap = self.row.get("VWAP_quarter_lower_1std", np.nan)

        self.pqvwap = self.row.get("VWAP_prev_quarter_final", np.nan)
        self.stdu_pqvwap = self.row.get("VWAP_prev_quarter_upper_1std", np.nan)
        self.stdl_pqvwap = self.row.get("VWAP_prev_quarter_lower_1std", np.nan)

        # Daily Key Level
        self.monday_high = self.row.get("monday_high", np.nan)
        self.monday_low = self.row.get("monday_low", np.nan)
        self.prev_day_high = self.row.get("prev_day_high", np.nan)
        self.prev_day_low = self.row.get("prev_day_low", np.nan)
        self.prev_week_high = self.row.get("prev_week_high", np.nan)
        self.prev_week_low = self.row.get("prev_week_low", np.nan)
        self.prev_month_high = self.row.get("prev_month_high", np.nan)
        self.prev_month_low = self.row.get("prev_month_low", np.nan)

        self.monday_high_idx = self.row.get("monday_high_idx", np.nan)
        self.monday_low_idx = self.row.get("monday_low_idx", np.nan)
        self.prev_day_high_idx = self.row.get("prev_day_high_idx", np.nan)
        self.prev_day_low_idx = self.row.get("prev_day_low_idx", np.nan)
        self.prev_week_high_idx = self.row.get("prev_week_high_idx", np.nan)
        self.prev_week_low_idx = self.row.get("prev_week_low_idx", np.nan)
        self.prev_month_high_idx = self.row.get("prev_month_high_idx", np.nan)
        self.prev_month_low_idx = self.row.get("prev_month_low_idx", np.nan)
        self.today_idx = self.row.get("today_idx", np.nan)

        # Volume Profile
        self.vol_z20 = self.row.get("volume_zscore_20", 0)
        self.vol_mad20 = self.row.get("volume_mad_20", 0)
        self.vol_z60 = self.row.get("volume_zscore_60", 0)
        self.vol_mad60 = self.row.get("volume_mad_60", 0)

        # Volume Helper
        self.prev_vol_z20 = self.G['rows_meta'][self.prev_index].get("volume_zscore_20", 0)
        self.prev_oi_z16 = self.G['rows_meta'][self.prev_index].get("oi_z16", 0)

        self.prev2_vol_z20 = self.G['rows_meta'][self.prev_index-1].get("volume_zscore_20", 0)
        self.prev2_oi_z16 = self.G['rows_meta'][self.prev_index-1].get("oi_z16", 0)

        self.prev3_vol_z20 = self.G['rows_meta'][self.prev_index-2].get("volume_zscore_20", 0)
        self.prev3_oi_z16 = self.G['rows_meta'][self.prev_index-2].get("oi_z16", 0)

        self.prev4_vol_z20 = self.G['rows_meta'][self.prev_index-3].get("volume_zscore_20", 0)
        self.prev4_oi_z16 = self.G['rows_meta'][self.prev_index-3].get("oi_z16", 0)

        self.prev5_vol_z20 = self.G['rows_meta'][self.prev_index-4].get("volume_zscore_20", 0)
        self.prev5_oi_z16 = self.G['rows_meta'][self.prev_index-4].get("oi_z16", 0)

        # Futures Derivation
        self.cvd = self.row.get("cvd", 0)
        self.cvd_delta = self.row.get("delta", 0)
        self.cvd_z16 = self.row.get("cvd_z16", 0)
        self.cvd_z96 = self.row.get("cvd_z96", 0)
        self.cvd_delta_z16 = self.row.get("delta_z16", 0)

        self.oi_delta = self.row.get("oi_delta", 0)
        self.oi_delta_1h = self.row.get("oi_delta_1h", 0)
        self.oi_delta_4h = self.row.get("oi_delta_4h", 0)
        self.oi_z16 = self.row.get("oi_z16", 0)
        self.oi_z96 = self.row.get("oi_z96", 0)
        self.oi_delta_z16 = self.row.get("oi_delta_z16", 0)

        self.price_delta_pct = self.row.get("price_delta_pct", 0)
        self.price_delta_z16 = self.row.get("price_delta_z16", 0)
        self.price_delta_z32 = self.row.get("price_delta_z32", 0)

        # Append Target Candidate
        candidates = [*self.sp_list, self.session_vah, self.session_val,
                      self.poc, self.vah, self.val,
                      self.comp_poc, self.comp_vah, self.comp_val,
                      self.prev_cvah, self.prev_cval,
                      self.v7, self.v30, self.v90, 
                      self.mvwap, self.qvwap, self.pmvwap, self.pqvwap,
                      self.monday_high, self.monday_low, 
                      self.prev_day_high, self.prev_day_low,
                      self.prev_week_high, self.prev_week_low,
                      self.prev_month_high, self.prev_month_low]
        candidates = sorted(list(dict.fromkeys([float(c) for c in candidates if not pd.isna(c)])))

        return candidates
    
    def is_virgin_level(self, level, level_idx):
        if pd.isna(level) or pd.isna(level_idx):
            return False

        high_list = self.G['high_list']
        low_list = self.G['low_list']

        # scan from start_idx to the last bar
        for h, l in zip(high_list[int(level_idx)+4:self.index], low_list[int(level_idx)+4:self.index]):
            if float(l) <= level <= float(h):
                return False  # touched

        return True  # virgin

    def search_trigger(self):
        """
        Determine trade triggers using AMT levels, wick extremes, session ranges, and flow confirmation.
        Returns: dictionary keyed by direction with list of trigger reasons, e.g.,
        {
            "long": ["PW_L_bounce", "ML_sweep_touch"],
            "short": ["cVAH_reject"]
        }
        Returns None if no triggers.
        """

        triggers = {"long": [], "short": []}
        prox = self.cfg.params.prox_pct
        confirm_n = 12 * 4

        # --- 2. Composite VAH/VAL breakout with confirmation ---
        # Only take composite VA (comp_len > 1)
        # VA with comp_len == 1 is not composite but previous daily TPO

        # --- Composite VAH ---
        if not pd.isna(self.comp_vah) and self.comp_len >=3:
            # candidate breakout
            past_bars = range(max(0, self.index - confirm_n + 1), self.index)
            all_outside = all(
                float(self.G['close_list'][j]) > self.comp_vah for j in past_bars
            )
            if all_outside and self.close > self.comp_vah:
                triggers["long"].append("cVAH_breakout")
            elif self.high > self.comp_vah and self.close < self.comp_vah:
                triggers["short"].append("cVAH_reject")

        # --- Composite VAL ---
        if not pd.isna(self.comp_val) and self.comp_len >=3:
            # candidate breakout
            past_bars = range(max(0, self.index - confirm_n + 1), self.index)
            all_outside = all(
                float(self.G['close_list'][j]) < self.comp_val for j in past_bars
            )
            if all_outside and self.close < self.comp_val:
                triggers["short"].append("cVAL_breakout")
                # pass
            elif self.low < self.comp_val and self.close > self.comp_val:
                # same-bar rejection
                triggers["long"].append("cVAL_reject")

        # --- 3. Monday High/Low sweep ---
        # Take virgin monday sweep, no sweep at monday is allowed as the price is still developing
        if not pd.isna(self.monday_low) and (pd.Timestamp(self.current_ts).weekday() != 0) and utility.proximity(self.low, self.monday_low, prox):
            if self.is_virgin_level(self.monday_low, self.monday_low_idx):
                triggers["long"].append("nML")
            else:
                triggers["long"].append("ML")

        if not pd.isna(self.monday_high) and (pd.Timestamp(self.current_ts).weekday() != 0) and utility.proximity(self.high, self.monday_high, prox):
            if self.is_virgin_level(self.monday_high, self.monday_high_idx):
                triggers["short"].append("nMH")
            else:
                triggers["short"].append("MH")

        # --- 4. Previous Week High/Low rejection/bounce ---
        if not pd.isna(self.prev_week_high) and utility.proximity(self.high, self.prev_week_high, prox):
            if self.is_virgin_level(self.prev_week_high, self.prev_week_high_idx):
                triggers["short"].append("nPWH")
            else:
                triggers["short"].append("PWH")

        if not pd.isna(self.prev_week_low) and utility.proximity(self.low, self.prev_week_low, prox):
            if self.is_virgin_level(self.prev_week_low, self.prev_week_low_idx):
                triggers["long"].append("nPWL")
            else:
                triggers["long"].append("PWL")

        # --- 5. Previous Month High/Low rejection/bounce ---
        if not pd.isna(self.prev_month_high) and utility.proximity(self.high, self.prev_month_high, prox):
            if self.is_virgin_level(self.prev_month_high, self.prev_month_high_idx):
                triggers["short"].append("nPMH")
            else:
                triggers["short"].append("PMH")

        if not pd.isna(self.prev_month_low) and utility.proximity(self.low, self.prev_month_low, prox):
            if self.is_virgin_level(self.prev_month_low, self.prev_month_low_idx):
                triggers["long"].append("nPML")
            else:
                triggers["long"].append("PML")

        # --- 6. Previous Day High/Low ---
        if not pd.isna(self.prev_day_high) and utility.proximity(self.high, self.prev_day_high, prox) and self.is_virgin_level(self.prev_day_high, self.prev_day_high_idx):
            triggers["short"].append("nPDH")

        if not pd.isna(self.prev_day_low) and utility.proximity(self.low, self.prev_day_low, prox) and self.is_virgin_level(self.prev_day_low, self.prev_day_low_idx):
            triggers["long"].append("nPDL")
                
        # --- 7. Deduplicate + clean ---
        triggers = {k: list(dict.fromkeys(v)) for k, v in triggers.items() if v}
        return triggers or None

    def score_trade(self, MLModel_counter, MLModel_pred, MLModel_dd, MLModel_features, triggers):
        """
        Score trade based on triggers, candle shape, flow, volume, VWAP, session, and IB/value migration.
        Input: triggers = output of search_trigger()
        Returns: {direction: (tier, score, flags)}
        """
        results = {}
        
        for direction, trigger_list in triggers.items():
            score = 0.0
            flags = {
                "flag_volume_spike_c1": False,
                "flag_volume_spike_c2": False,
                "flag_volume_spike_c3": False,
                "flag_volume_spike_c4": False,
                "flag_volume_spike_c5": False,
                "flag_volume_spike_c6": False,
                "flag_vwap_touch": False,
                "flag_session_bias": False,
                "flag_ib_touch": False,
                "flag_value_migration": False,
                "flag_sweep_monday": False,
                "flag_sweep_pd": False,
                "flag_sweep_pw": False,
                "flag_sweep_pm": False,
                "flag_sweep_sp": False,
                "flag_breakout_va": False,
                "flag_rotation_va": False,
                "flag_countertrade": False
            }

            MLFeatures = {
                # Value Area Distance
                "ml_dist_cvah": 0,
                "ml_dist_cval": 0,
                "ml_dist_cpoc": 0,
                "ml_dist_pcvah": 0,
                "ml_dist_pcval": 0,
                # VWAP Distance Percentage
                "ml_dist_v7": 0,
                "ml_dist_v30": 0,
                "ml_dist_v90": 0,
                "ml_dist_mvwap": 0,
                "ml_dist_qvwap": 0,
                "ml_dist_pmvwap": 0,
                "ml_dist_pqvwap": 0,
                # STD VWAP Distance Percentage
                "ml_dist_stdu_mvwap": 0,
                "ml_dist_stdl_mvwap": 0,
                "ml_dist_stdu_pmvwap": 0,
                "ml_dist_stdl_pmvwap": 0,
                "ml_dist_stdu_qvwap": 0,
                "ml_dist_stdl_qvwap": 0,
                "ml_dist_stdu_pqvwap": 0,
                "ml_dist_stdl_pqvwap": 0,
                # Volume Spike 
                "ml_vol_spike_c1": 0,
                "ml_vol_spike_c2": 0,
                "ml_vol_spike_c3": 0,
                "ml_vol_spike_c4": 0,
                "ml_vol_spike_c5": 0,
                "ml_vol_spike_c6": 0,
                # Sweep
                "ml_sweep_monday": False,
                "ml_sweep_pd": False,
                "ml_sweep_pw": False,
                "ml_sweep_pm": False,
                "ml_sweep_sp": False,
                "ml_value_area_br": False,
                "ml_value_area_rotate": False,
                # Session
                "ml_session_bias": False,
                "ml_ib_touch": False,
                "ml_value_migration": False,
                # Price
                "price_delta_z16": 0.0,
                "price_delta_z32": 0.0,
            }

            for sp in self.sp_list:
                if utility.proximity(sp, self.low, self.cfg.params.prox_pct):
                    score += self.pts.get("sweep_trigger", 1.0)
                    flags["flag_sweep_sp"] = True
                    MLFeatures["ml_sweep_sp"] = True
                elif utility.proximity(sp, self.high, self.cfg.params.prox_pct):
                    score += self.pts.get("sweep_trigger", 1.0)
                    flags["flag_sweep_sp"] = True
                    MLFeatures["ml_sweep_sp"] = True

            # --- 1. Base points per trigger reason ---
            for trig in trigger_list:
                if trig in ("ML", "MH", "nML", "nMH"):
                    score += self.pts.get("sweep_trigger", 1.0)
                    flags["flag_sweep_monday"] = True
                    MLFeatures["ml_sweep_monday"] = True
                elif trig in ("nPDL", "nPDH"):
                    score += self.pts.get("rotation_trigger", 1.0)
                    flags["flag_sweep_pd"] = True
                    MLFeatures["ml_sweep_pd"] = True
                elif trig in ("PWL", "PWH", "nPWL", "nPWH"):
                    score += self.pts.get("rotation_trigger", 1.0)
                    flags["flag_sweep_pw"] = True
                    MLFeatures["ml_sweep_pw"] = True
                elif trig in ("PML", "PMH", "nPML", "nPMH"):
                    score += self.pts.get("rotation_trigger", 1.0)
                    flags["flag_sweep_pm"] = True
                    MLFeatures["ml_sweep_pm"] = True
                elif trig in ("cVAH_breakout", "cVAL_breakout"):
                    score += self.pts.get("breakout_trigger", 1.0)
                    flags["flag_breakout_va"] = True
                    MLFeatures["ml_value_area_br"] = True
                elif trig in ("cVAH_reject", "cVAL_reject"):
                    score += self.pts.get("rotation_va_reject", 1.0)
                    flags["flag_rotation_va"] = True
                    MLFeatures["ml_value_area_rotate"] = True
                    if self.comp_len >=4:
                        score += self.pts.get("major_reject", 1.0)


            # --- 2. Static Derivative Tape scoring ---
            score, deriv_flags, deriv_mlfeatures = self.static_derivative_score(score, direction)
            flags.update(deriv_flags)
            MLFeatures.update(deriv_mlfeatures)

            score, tape_flags, tape_mlfeatures = self.event_tape_detection(score, direction)
            flags.update(tape_flags)
            MLFeatures.update(tape_mlfeatures)

            # --- 3. Candle pattern scoring ---
            score, candle_flags = self.candle_pattern_score(score, direction)
            flags.update(candle_flags)
            MLFeatures.update(candle_flags)

            # --- 3. Volume Spike Detector ---
            vol_count = 0
            threshold_vol1 = 1.0
            threshold_vol2 = 2.0

            for vol_br in [self.vol_z20, self.prev_vol_z20, self.prev2_vol_z20, self.prev3_vol_z20, self.prev4_vol_z20, self.prev5_vol_z20]:
                if vol_br >= threshold_vol1:
                    vol_count += 1
                    score += (self.pts.get("volume_spike", 1.0) * 2) if vol_br >= threshold_vol2 else self.pts.get("volume_spike", 1.0)
                    flags[f"flag_volume_spike_c{vol_count}"] = True
                    MLFeatures[f"ml_vol_spike_c{vol_count}"] = vol_br

            # --- 4. VWAP Detector ---
            touch_vwaps = [v for v in [self.v7, self.v30, self.v90, self.mvwap, self.pmvwap, self.qvwap, self.pqvwap] if not pd.isna(v)]
            for v in touch_vwaps:
                if utility.proximity(v, self.close, self.cfg.params.prox_pct):
                    score += self.pts.get("vwap_touch", 0.5)
                    flags["flag_vwap_touch"] = True

            MLFeatures["ml_dist_v7"] = (self.close - self.v7) / self.close
            MLFeatures["ml_dist_v30"] = (self.close - self.v30) / self.close
            MLFeatures["ml_dist_v90"] = (self.close - self.v90) / self.close

            MLFeatures["ml_dist_mvwap"] = (self.close - self.mvwap) / self.close
            MLFeatures["ml_dist_qvwap"] = (self.close - self.qvwap) / self.close
            MLFeatures["ml_dist_pmvwap"] = (self.close - self.pmvwap) / self.close
            MLFeatures["ml_dist_pqvwap"] = (self.close - self.pqvwap) / self.close

            MLFeatures["ml_dist_stdu_mvwap"] = (self.close - self.stdu_mvwap) / self.close
            MLFeatures["ml_dist_stdl_mvwap"] = (self.close - self.stdl_mvwap) / self.close
            MLFeatures["ml_dist_stdu_pmvwap"] = (self.close - self.stdu_pmvwap) / self.close
            MLFeatures["ml_dist_stdl_pmvwap"] = (self.close - self.stdl_pmvwap) / self.close

            MLFeatures["ml_dist_stdu_qvwap"] = (self.close - self.stdu_qvwap) / self.close
            MLFeatures["ml_dist_stdl_qvwap"] = (self.close - self.stdl_qvwap) / self.close
            MLFeatures["ml_dist_stdu_pqvwap"] = (self.close - self.stdu_pqvwap) / self.close
            MLFeatures["ml_dist_stdl_pqvwap"] = (self.close - self.stdl_pqvwap) / self.close

            # --- 5. Session Bias ---
            if direction == "long":
                if utility.proximity(self.ib_low, self.close, self.cfg.params.prox_pct):
                    score += self.pts.get("ib_touch", 0.5)
                    flags["flag_ib_touch"] = True
                    MLFeatures["ml_ib_touch"] = True
                if utility.proximity(self.session_val, self.close, self.cfg.params.prox_pct):
                    score += self.pts.get("session_bias", 0.5)
                    flags["flag_session_bias"] = True
                    MLFeatures["ml_session_bias"] = True
                if self.opening_type in ("in_value_open", "below_value_open"):
                    score += self.pts.get("opening_type", 0.5)
                if self.value_migration == "higher_value":
                    score += self.pts.get("value_migration", 0.5)
                    flags["flag_value_migration"] = True
                    MLFeatures["ml_value_migration"] = True
            else:
                if utility.proximity(self.ib_high, self.close, self.cfg.params.prox_pct):
                    score += self.pts.get("ib_touch", 0.5)
                    flags["flag_ib_touch"] = True
                    MLFeatures["ml_ib_touch"] = True
                if utility.proximity(self.session_vah, self.close, self.cfg.params.prox_pct):
                    score += self.pts.get("session_bias", 0.5)
                    flags["flag_session_bias"] = True
                    MLFeatures["ml_session_bias"] = True
                if self.opening_type == "above_value_open":
                    score += self.pts.get("opening_type", 0.5)
                if self.value_migration == "lower_value":
                    score += self.pts.get("value_migration", 0.5)
                    flags["flag_value_migration"] = True
                    MLFeatures["ml_value_migration"] = True

            # --- 6. Value Area Distance ---
            MLFeatures["ml_dist_cvah"] = (self.close - self.comp_vah) / self.close
            MLFeatures["ml_dist_cval"] = (self.close - self.comp_val) / self.close
            MLFeatures["ml_dist_cpoc"] = (self.close - self.comp_poc) / self.close
            MLFeatures["ml_dist_pcvah"] = (self.close - self.prev_cvah) / self.close
            MLFeatures["ml_dist_pcval"] = (self.close - self.prev_cval) / self.close

            # --- 7. Pricing ---
            MLFeatures["price_delta_z16"] = self.price_delta_z16
            MLFeatures["price_delta_z32"] = self.price_delta_z32

            # --- 7. BETA: MLModel Scoring (Regression) ---
            X_live = self.make_live_row(MLFeatures, MLModel_features)
            MLFeatures["counter_score"] = float(MLModel_counter.predict(X_live)[0])
            X_live["counter_score"] = MLFeatures["counter_score"]

            if MLFeatures["counter_score"] > 0.50:
                flags["flag_countertrade"] = True
                X_live["is_long"] = ~X_live["is_long"]

            predicted_reward = float(MLModel_pred.predict(X_live)[0])
            predicted_dd = float(MLModel_dd.predict(X_live)[0])
                
            # --- 7. Scoring System ---
            predicted_move_pcg = (predicted_reward - abs(predicted_dd)) * abs(self.price_delta_z32) * 100
            final_tier = self.tier_func(predicted_move_pcg)

            results[direction] = (final_tier, predicted_move_pcg, flags, MLFeatures)

        return results

    def static_derivative_score(self, score, direction):
        conf_z = 0.5
        long_bias_z = 1.25

        flags = {
            "flag_flow_confirm_long": False,
            "flag_flow_confirm_short": False,
            "flag_oi_building": False,
            "flag_oi_unwinding": False,
            "flag_bias_bull": False,
            "flag_bias_bear": False,
        }
        MLFeatures = {
            "ml_oi_building": False,
            "ml_oi_unwinding": False,
        }

        # --- 1. short-term flow confirmation (cvd_delta_z16 & oi_delta_z16)
        if direction == "long" and self.cvd_delta_z16 >= conf_z and self.oi_delta_z16 >= conf_z:
            score += self.pts.get("flow_confirm", 0.5) * min(self.cvd_delta_z16, self.oi_delta_z16)
            flags["flag_flow_confirm_long"] = True
        elif direction == "short" and self.cvd_delta_z16 <= -conf_z and self.oi_delta_z16 <= -conf_z:
            score += self.pts.get("flow_confirm", 0.5) * min(abs(self.cvd_delta_z16), abs(self.oi_delta_z16))
            flags["flag_flow_confirm_short"] = True

        # --- 2. multi-horizon OI agreement (persistent positioning)
        oi_s15 = utility.sign(self.oi_delta)
        oi_s1h = utility.sign(self.oi_delta_1h)
        oi_s4h = utility.sign(self.oi_delta_4h)

        if all(x == 1 for x in [oi_s15, oi_s1h, oi_s4h] if x != 0):
            if direction == "long": score += self.pts.get("flow_confirm", 0.5)
            else: score -= self.pts.get("flow_confirm", 0.5)
            flags["flag_oi_building"] = True
        elif all(x == -1 for x in [oi_s15, oi_s1h, oi_s4h] if x != 0):
            if direction == "short": score += self.pts.get("flow_confirm", 0.5)
            else: score -= self.pts.get("flow_confirm", 0.5)
            flags["flag_oi_unwinding"] = True

        # --- 3. long-term structural bias (cvd_z96 / oi_z96)
        if (self.cvd_z96 >= long_bias_z or self.oi_z96 >= long_bias_z):
            if direction == "long": score += 0.5
            else: score -= 0.5
            flags["flag_bias_bull"] = True
        elif (self.cvd_z96 <= -long_bias_z or self.oi_z96 <= -long_bias_z):
            if direction == "short": score += 0.5
            else: score -= 0.5
            flags["flag_bias_bear"] = True

        return score, flags, MLFeatures
    
    def event_tape_detection(self, score, direction):
        climax_z = 2.0
        w_climax = 0.7
        w_climax_multitf = 0.3

        ev_flags = {
            "flag_climax_multitf": False,
            "flag_climax_top": False,
            "flag_climax_bottom": False,
            "flag_oi_spike_c1": False,
            "flag_oi_spike_c2": False,
            "flag_oi_spike_c3": False,
            "flag_oi_spike_c4": False
        }

        MLFeatures = {
            "ml_oi_spike_c1": 0,
            "ml_oi_spike_c2": 0,
            "ml_oi_spike_c3": 0,
            "ml_oi_spike_c4": 0
        }

        # --- 3. Volume Spike Detector ---
        oi_count = 0
        threshold_oi1 = 1.5
        threshold_oi2 = 2.5
        for oi_br in [self.oi_z16, self.prev_oi_z16, self.prev2_oi_z16, self.prev3_oi_z16]:
            oi_count += 1
            if oi_br >= threshold_oi1 and direction == "long":
                score += (self.pts.get("oi_spike", 0.25) * 2) if oi_br >= threshold_oi2 else self.pts.get("oi_spike", 0.25)
                ev_flags[f"flag_oi_spike_c{oi_count}"] = True
            elif oi_br <= -threshold_oi1 and direction == "short":
                score += (self.pts.get("oi_spike", 0.25) * 2) if oi_br <= -threshold_oi2 else self.pts.get("oi_spike", 0.25)
                ev_flags[f"flag_oi_spike_c{oi_count}"] = True
            MLFeatures[f"ml_oi_spike_c{oi_count}"] = oi_br if abs(oi_br) >= threshold_oi1 else 0
                
        # --- CLIMAX event detection ---
        if direction == "short" and (self.cvd_delta_z16 >= climax_z or self.oi_delta_z16 >= climax_z):
            base = w_climax
            oi_signs = [utility.sign(self.oi_delta_1h), utility.sign(self.oi_delta_4h)]
            if all(x > 0 for x in oi_signs):
                base *= (1 + w_climax_multitf)
                ev_flags["flag_climax_multitf"] = True
            else:
                ev_flags["flag_climax_top"] = True
            score += base

        elif direction == "long" and (self.cvd_delta_z16 <= -climax_z or self.oi_delta_z16 <= -climax_z):
            base = w_climax
            oi_signs = [utility.sign(self.oi_delta_1h), utility.sign(self.oi_delta_4h)]
            if all(x < 0 for x in oi_signs):
                base *= (1 + w_climax_multitf)
                ev_flags["flag_climax_multitf"] = True
            else:
                ev_flags["flag_climax_bottom"] = True
            score += base

        return score, ev_flags, MLFeatures

    def candle_pattern_score(self, score, direction):
        flags = {
                "ml_candle_hammer_shooting": False,
                "ml_candle_engulfing": False
            }
                
        body = abs(self.close - self.open)
        delta_body = np.sign(self.close - self.open)
        upper_wick = self.high - max(self.close, self.open)
        lower_wick = min(self.close, self.open) - self.low

        # --- Engulfing Pattern Detection ---
        prev_body_size = abs(self.prev_close - self.prev_open)

        # Bullish Engulfing Pattern
        is_bullish_engulfing = (
            (self.prev_close < self.prev_open) and
            (self.close > self.open) and
            self.open <= self.prev_close and
            self.close >= self.prev_open and
            body > prev_body_size
        )

        # Bearish Engulfing Pattern
        is_bearish_engulfing = (
            (self.prev_close > self.prev_open) and
            (self.close < self.open) and
            self.open >= self.prev_close and
            self.close <= self.prev_open and
            body > prev_body_size  
        )

        # Hammer / shooting star
        if direction == "long" and lower_wick > 2.5 * body and delta_body > 0:
            score += self.pts.get("candle_hammer", 1.0)
            flags["ml_candle_hammer_shooting"] = True
        elif direction == "short" and upper_wick > 2.5 * body and delta_body < 0:
            score += self.pts.get("candle_shooting_star", 1.0)
            flags["ml_candle_hammer_shooting"] = True
        # Engulfing
        if direction == "long" and is_bullish_engulfing:
            score += self.pts.get("candle_engulfing", 1.5)
            flags["ml_candle_engulfing"] = True
        elif direction == "short" and is_bearish_engulfing:
            score += self.pts.get("candle_engulfing", 1.5)
            flags["ml_candle_engulfing"] = True
        
        return score, flags

    def score_to_tier(self, score):
        if score >= self.cfg.params.tier_score["S"]: return "S"
        elif score >= self.cfg.params.tier_score["A"] and score < self.cfg.params.tier_score["S"]: return "A"
        elif score >= self.cfg.params.tier_score["B"] and score < self.cfg.params.tier_score["A"]: return "B"
        else: return "C"

    def make_live_row(self, MLFeatures, feature_cols):
        # Start with empty row
        row = {col: 0 for col in feature_cols}

        # Fill with provided values
        for k, v in MLFeatures.items():
            if k in row:
                row[k] = v

        # Convert boolean → int
        for k, v in row.items():
            if isinstance(v, bool):
                row[k] = int(v)

        # Return DataFrame
        return pd.DataFrame([row])

    def tier_func(self, p):
        if p >= 6: return "S+"
        if p >= 3: return "S"
        elif p >= 2: return "A"
        elif p > 1: return "B"
        else: return "C"

