# ------------------------
# tradeSimulator.py
# ------------------------
import pandas as pd
import numpy as np
import utility.helpers as utility

class TradeSimulator:
    def __init__(self, G, cfg):
        """
        cfg = load_config("config.yaml")
        """
        self.G = G
        self.lookahead_days = cfg.params.lookahead
        self.bars_per_day = cfg.params.bars["per_day"]
        self.bars_per_hour = cfg.params.bars["per_hour"]

    def simulate(self, i, entry_price, stop, target, direction, tier_raw):
        start_idx = i + 1
        max_bars = self._get_lookahead_bars(tier_raw)
        end_idx = min(self.G['bars_len'], start_idx + max_bars)

        stop_active = stop
        stop_moved = False
        bars_above_1R = 0
        outcome = "timeout"
        exit_price = np.nan
        exit_ts = pd.NaT
        highest_rr = 0.0

        for j in range(start_idx, end_idx):
            high = float(self.G['high_list'][j])
            low = float(self.G['low_list'][j])
            close = float(self.G['close_list'][j])

            # Stop movement logic
            if not stop_moved:
                bars_above_1R = self._check_stop_move(direction, entry_price, stop, bars_above_1R, close)
                if bars_above_1R >= 4:
                    stop_active = entry_price + (entry_price - stop) * 0.25 if direction=="long" else entry_price - (stop - entry_price) * 0.25
                    stop_moved = True

            # --- Compute unrealized RR and update highest RR ---
            if direction == "long":
                unrealized_rr = utility.compute_rr(entry_price, self.G['close_list'][j], stop, direction)
            else:
                unrealized_rr = utility.compute_rr(entry_price, self.G['close_list'][j], stop, direction)
            highest_rr = max(highest_rr, unrealized_rr)
            
            # Normal exit
            if direction=="long":
                if high >= target: outcome, exit_price, exit_ts = "target_hit", target, self.G['ts_list'][j]; break
                if low <= stop_active: outcome, exit_price, exit_ts = "stop_hit" if not stop_moved else "BEP", stop_active, self.G['ts_list'][j]; break
            else:
                if low <= target: outcome, exit_price, exit_ts = "target_hit", target, self.G['ts_list'][j]; break
                if high >= stop_active: outcome, exit_price, exit_ts = "stop_hit" if not stop_moved else "BEP", stop_active, self.G['ts_list'][j]; break

        if np.isnan(exit_price):  # no exit hit
            last_idx = end_idx - 1
            exit_price, exit_ts = float(self.G['close_list'][last_idx]), self.G['ts_list'][last_idx]

        realized_rr = utility.compute_rr(entry_price, exit_price, stop, direction)
        return exit_price, exit_ts, outcome, realized_rr, highest_rr

    def simulate_no_stop(self, i, entry_price, stop, target, direction, tier_raw):
        start_idx = i + 1
        max_bars = self._get_lookahead_bars("S")
        end_idx = min(self.G['bars_len'], start_idx + max_bars)

        highest_rr = 0.0
        lowest_rr = 0.0
        outcome = "timeout"
        exit_price = np.nan
        exit_ts = pd.NaT

        for j in range(start_idx, end_idx):
            high = float(self.G['high_list'][j])
            low = float(self.G['low_list'][j])
            close = float(self.G['close_list'][j])
            unrealized_rr = utility.compute_rr(entry_price, close, stop, direction)
            highest_rr = max(highest_rr, unrealized_rr)
            lowest_rr = min(lowest_rr, unrealized_rr)

            # only exit on target or time
            if direction=="long" and high >= target:
                outcome, exit_price, exit_ts = "target_hit", self.G['close_list'][j], self.G['ts_list'][j]; break
            elif direction=="short" and low <= target:
                outcome, exit_price, exit_ts = "target_hit", self.G['close_list'][j], self.G['ts_list'][j]; break

        if np.isnan(exit_price):
            last_idx = end_idx - 1
            exit_price, exit_ts = float(self.G['close_list'][last_idx]), self.G['ts_list'][last_idx]

        realized_rr = utility.compute_rr(entry_price, exit_price, stop, direction)
        return exit_price, exit_ts, outcome, realized_rr, highest_rr, lowest_rr

    def simulate_all(self, i, entry_price):
        start_idx = i + 1
        max_bars = self._get_lookahead_bars("B")
        end_idx = min(self.G['bars_len'], start_idx + max_bars)

        highest_idx = 0
        lowest_idx = 0
        highest_move = 0.0
        lowest_move = 0.0

        exit_price = np.nan
        exit_ts = pd.NaT

        for j in range(start_idx, end_idx):
            # Get Bar
            close = float(self.G['close_list'][j])

            # Compute Percentage
            unrealized_move = (close - entry_price) / entry_price
            if max(highest_move, unrealized_move) == unrealized_move:
                highest_move = unrealized_move
                highest_idx = j - start_idx
            if min(lowest_move, unrealized_move) == unrealized_move:
                lowest_move = unrealized_move
                lowest_idx = j - start_idx

        if np.isnan(exit_price):
            last_idx = end_idx - 1
            exit_price, exit_ts = float(self.G['close_list'][last_idx]), self.G['ts_list'][last_idx]

        return exit_price, exit_ts, highest_move, highest_idx, lowest_move, lowest_idx

    def _get_lookahead_bars(self, tier_raw):
        tier_map = {
            "S+": "s_days",
            "S": "s_days",
            "A": "a_days",
            "B": "b_hours",
            "C": "b_hours"
        }
        key = tier_map.get(tier_raw)
        if key is None:
            raise ValueError(f"Unknown tier: {tier_raw}")
        
        if tier_raw in ["S+", "S", "A"]:
            return self.lookahead_days[key] * self.bars_per_day
        else:
            return self.lookahead_days[key] * self.bars_per_hour

    def _check_stop_move(self, direction, entry_price, stop, bars_above_1R, close):
        if direction=="long":
            if close >= entry_price + (entry_price - stop):
                return bars_above_1R + 1
            else:
                return 0
        else:
            if close <= entry_price - (stop - entry_price):
                return bars_above_1R + 1
            else:
                return 0
            

# # ------------------------
# # tradeSimulator.py
# # ------------------------
# import pandas as pd
# import numpy as np
# import utility.helpers as utility

# class TradeSimulator:
#     def __init__(self, G, cfg):
#         """
#         cfg = load_config("config.yaml")
#         """
#         self.G = G
#         self.lookahead_days = cfg.params.lookahead
#         self.bars_per_day = cfg.params.bars["per_day"]
#         self.bars_per_hour = cfg.params.bars["per_hour"]

#     def simulate(self, i, entry_price, stop, target, direction, tier_raw, trail_offset=0.75):
#         """
#         trail_offset: distance (in R units) behind the current high/low once trailing is active.
#         Example: 0.5 = trailing stop 0.5R behind current high/low
#         """
#         start_idx = i + 1
#         max_bars = self._get_lookahead_bars(tier_raw)
#         end_idx = min(self.G['bars_len'], start_idx + max_bars)

#         stop_active = stop
#         trail_active = False
#         outcome = "timeout"
#         exit_price = np.nan
#         exit_ts = pd.NaT
#         highest_rr = -np.inf
#         bars_above_1R = 0
#         R_value = abs(entry_price - stop)

#         for j in range(start_idx, end_idx):
#             high = float(self.G['high_list'][j])
#             low = float(self.G['low_list'][j])
#             close = float(self.G['close_list'][j])

#             # --- Compute unrealized RR and update highest RR ---
#             if direction == "long":
#                 unrealized_rr = (close - entry_price) / R_value
#             else:
#                 unrealized_rr = (entry_price - close) / R_value
#             highest_rr = max(highest_rr, unrealized_rr)

#             # --- Activate or update trailing stop ---
#             if not trail_active:
#                 # Activate trailing once trade reaches +1R
#                 bars_above_1R = self._check_stop_move(direction, entry_price, stop, bars_above_1R, close)
#                 if bars_above_1R >= 1:
#                     trail_active = True
#             else:
#                 # Update trailing stop dynamically
#                 if direction == "long":
#                     new_stop = close - trail_offset * R_value
#                     stop_active = max(stop_active, new_stop)  # never lower stop
#                 else:
#                     new_stop = close + trail_offset * R_value
#                     stop_active = min(stop_active, new_stop)  # never raise stop

#             # --- Exit conditions ---
#             if direction == "long":
#                 if high >= target:
#                     outcome, exit_price, exit_ts = "target_hit", target, self.G['ts_list'][j]
#                     break
#                 if low <= stop_active:
#                     outcome = "trail_hit" if trail_active else "stop_hit"
#                     exit_price, exit_ts = stop_active, self.G['ts_list'][j]
#                     break
#             else:
#                 if low <= target:
#                     outcome, exit_price, exit_ts = "target_hit", target, self.G['ts_list'][j]
#                     break
#                 if high >= stop_active:
#                     outcome = "trail_hit" if trail_active else "stop_hit"
#                     exit_price, exit_ts = stop_active, self.G['ts_list'][j]
#                     break

#         # --- Timeout (no exit hit) ---
#         if np.isnan(exit_price):
#             last_idx = end_idx - 1
#             exit_price, exit_ts = float(self.G['close_list'][last_idx]), self.G['ts_list'][last_idx]

#         realized_rr = utility.compute_rr(entry_price, exit_price, stop, direction)
#         return exit_price, exit_ts, outcome, realized_rr, highest_rr

#     def _get_lookahead_bars(self, tier_raw):
#         tier_map = {
#             "S": "s_days",
#             "A": "a_days",
#             "B": "b_hours",
#             "C": "b_hours"
#         }
#         key = tier_map.get(tier_raw)
#         if key is None:
#             raise ValueError(f"Unknown tier: {tier_raw}")

#         if tier_raw in ["S", "A"]:
#             return self.lookahead_days[key] * self.bars_per_day
#         else:
#             return self.lookahead_days[key] * self.bars_per_hour

#     def _check_stop_move(self, direction, entry_price, stop, bars_above_1R, close):
#         if direction == "long":
#             if close >= entry_price + (entry_price - stop):
#                 return bars_above_1R + 1
#             else:
#                 return 0
#         else:
#             if close <= entry_price - (stop - entry_price):
#                 return bars_above_1R + 1
#             else:
#                 return 0