# ------------------------
# TradePicker.py
# ------------------------
import pandas as pd
import utility.helpers as utility

class TradePicker:
    def __init__(self, cfg):
        """
        cfg = load_config("config.yaml")
        """
        # Config
        self.cfg = cfg
        self.max_rr_allowed = cfg.params.max_rr_allowed
        self.tier_filters = cfg.params.tier_filters


    def _pick_from_score(self, entry_price, direction, score, final_tier, candidates):
        # Pick Multiplier
        multiplier = 0.0
        if final_tier == "S+" or final_tier =="S":
            multiplier = 0.75
        elif final_tier == "A":
            multiplier = 0.65
        else:
            multiplier = 0.5

        min_rr = self.tier_filters[final_tier]["min_rr"]
        min_dist_pct = self.tier_filters[final_tier]["min_dist_pct"]
        max_dist_pct = self.tier_filters[final_tier]["max_dist_pct"]
        max_tgt_dist_pct = self.tier_filters[final_tier]["max_tgt_dist_pct"]

        # target candidates
        score_tgt = 0.0
        if direction == "long":
            score_tgt = entry_price * (1.0 + (abs(score) * multiplier) * 0.01)
        else:
            score_tgt = entry_price * (1.0 - (abs(score) * multiplier) * 0.01)

        # target candidates
        if direction == "long":
            tgt_candidates = [c * (1.0 - 0.003) for c in candidates if c > entry_price]
        else:
            tgt_candidates = [c * (1.0 + 0.003) for c in candidates if c < entry_price]

        tgt_candidates.append(score_tgt)
        tgt_candidates = utility.dedup_levels(tgt_candidates, min_dist=1e-6)
        stop_candidates = self.candidate_stops_for_entry(entry_price, direction, candidates)

        pairs = []
        for tgt in tgt_candidates:
            reward_dist_pct = abs((tgt - entry_price) / entry_price)
            if reward_dist_pct > max_tgt_dist_pct:
                continue

            for s in stop_candidates:
                rr = utility.compute_rr(entry_price, tgt, s, direction)
                if rr is None or rr < min_rr or rr > self.max_rr_allowed:
                    continue
                stop_dist_pct = abs((entry_price - s) / entry_price)
                if stop_dist_pct < min_dist_pct or stop_dist_pct > max_dist_pct:
                    continue

                # penalize overly far targets
                score = rr - 0.4 * reward_dist_pct * 100  # weight penalty
                pairs.append((score, rr, tgt, s))

        if not pairs:
            return None, None, None

        _, best_rr, best_tgt, best_stop = max(pairs, key=lambda x: x[0])
        return best_tgt, best_stop, best_rr

    def _pick_non_rotation(self, entry_price, direction, final_tier, candidates):
        min_rr = self.tier_filters[final_tier]["min_rr"]
        min_dist_pct = self.tier_filters[final_tier]["min_dist_pct"]
        max_dist_pct = self.tier_filters[final_tier]["max_dist_pct"]
        max_tgt_dist_pct = self.tier_filters[final_tier]["max_tgt_dist_pct"]

        # target candidates
        if direction == "long":
            tgt_candidates = [c * (1.0 - 0.003) for c in candidates if c > entry_price]
        else:
            tgt_candidates = [c * (1.0 + 0.003) for c in candidates if c < entry_price]

        tgt_candidates = utility.dedup_levels(tgt_candidates, min_dist=1e-6)
        stop_candidates = self.candidate_stops_for_entry(entry_price, direction, candidates)

        pairs = []
        for tgt in tgt_candidates:
            reward_dist_pct = abs((tgt - entry_price) / entry_price)
            if reward_dist_pct > max_tgt_dist_pct:
                continue

            for s in stop_candidates:
                rr = utility.compute_rr(entry_price, tgt, s, direction)
                if rr is None or rr < min_rr or rr > self.max_rr_allowed:
                    continue
                stop_dist_pct = abs((entry_price - s) / entry_price)
                if stop_dist_pct < min_dist_pct or stop_dist_pct > max_dist_pct:
                    continue

                # penalize overly far targets
                score = rr - 0.4 * reward_dist_pct * 100  # weight penalty
                pairs.append((score, rr, tgt, s))

        if not pairs:
            return None, None, None

        _, best_rr, best_tgt, best_stop = max(pairs, key=lambda x: x[0])
        return best_tgt, best_stop, best_rr

    def candidate_stops_for_entry(self, entry_price, direction, candidates):
        stops = []
        for cand in candidates:
            if not pd.isna(cand):
                if direction == "long" and cand < entry_price:
                    stops.append(cand * (1.0 - 0.003))
                elif direction == "short" and cand > entry_price:
                    stops.append(cand * (1.0 + 0.003))

        # Add a minimal cushion as last-resort stop
        cushion = entry_price * 0.007
        stops.append(entry_price - cushion if direction == "long" else entry_price + cushion)

        # Deduplicate & sort
        stops = utility.dedup_levels(sorted(set(stops)), min_dist=0.001)

        # ⚠ Strict direction filter to prevent invalid stops
        stops = [s for s in stops if (direction=="long" and s < entry_price) or (direction=="short" and s > entry_price)]

        return stops