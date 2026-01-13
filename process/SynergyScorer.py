import itertools
import pandas as pd

class SynergyScorer:
    def __init__(self, combo_stats_df):
        self.synergy_4way = self._build_synergy_maps(combo_stats_df)

    def _expectancy_to_boost(self, exp):
        """Convert expectancy value to a discrete synergy boost."""
        if exp < 30:
            return -1.0
        else:
            return 0.0

    def _build_synergy_maps(self, df):
        """Build separate synergy maps for 2-flag and 3-flag combos."""
        synergy_4way = {}

        for _, row in df.iterrows():
            exp = row["Winrate"]
            boost = self._expectancy_to_boost(exp)

            # Collect available flags
            flag_list = [f for f in [row.get("Flag1"), row.get("Flag2"), row.get("Flag3"), row.get("Flag4"), row.get("Flag5")] if pd.notna(f)]

            combo = tuple(sorted(flag_list))
            synergy_4way[combo] = boost

        return synergy_4way

    def score_flag_combination(self, flags):
        """
        Compute total synergy score for given active flags (set or list).
        Includes both 2-flag and 3-flag synergies.
        """
        flags = sorted(set(flags))
        total_score = 0.0

        # Check all 3-flag subsets
        if len(flags) >= 5:
            for combo in itertools.combinations(flags, 5):
                total_score += self.synergy_4way.get(combo, 0.0)

        return total_score
