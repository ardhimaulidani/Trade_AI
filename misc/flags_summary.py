import pandas as pd
from itertools import combinations

# Read the parquet file
df = pd.read_parquet('database/parquet/event_trades.parquet')

# Define main trigger flags
main_flags = ["flag_sweep_monday","flag_sweep_pd","flag_sweep_pw","flag_sweep_pm","flag_breakout_va","flag_rotation_va"]

# Define flag columns to analyze (secondary/contextual)
df['flag_volume_spike_any'] = (
    df['flag_volume_spike_c1'] |
    df['flag_volume_spike_c2'] |
    df['flag_volume_spike_c3'] |
    df['flag_volume_spike_c4'] |
    df['flag_volume_spike_c5'] |
    df['flag_volume_spike_c6']
)

df['flag_oi_spike_any'] = (
    df['flag_oi_spike_c1'] |
    df['flag_oi_spike_c2'] |
    df['flag_oi_spike_c3'] |
    df['flag_oi_spike_c4']
)

flag_columns = [
    'flag_volume_spike_any',
    'flag_oi_spike_any',
    'flag_vwap_touch', 'flag_session_bias', 
    'flag_ib_touch', 'flag_value_migration',
    'flag_flow_confirm_short', 'flag_flow_confirm_long', 'flag_oi_building', 'flag_oi_unwinding',
    'flag_bias_bull', 'flag_bias_bear', 
    'flag_climax_multitf', 'flag_climax_top', 'flag_climax_bottom',
    'flag_bullish_reset', 'flag_bearish_reset', "flag_bear_trap", "flag_bull_trap",
    'candle_hammer_shooting', 'candle_engulfing', 'candle_marubozu'
]

print("=" * 120)
print("FLAG COMBINATION ANALYSIS - Including MAIN TRIGGER FLAGS")
print("=" * 120)

# =====================================================================
# PART 0: MAIN FLAG–BASED COMBINATION ANALYSIS
# =====================================================================
print("\n📊 PART 0: MAIN TRIGGER–BASED FLAG COMBINATIONS")
print("-" * 120)

main_results = []
main_combo_results = []
main_combo3_results = []
main_combo4_results = []

# (flags)
single_flag = list(flag_columns)
single_flag.append("flag_countertrade")

for flag in single_flag:
    combo_trades = df[df[flag] == True]
    total_calls = len(combo_trades)
    if total_calls >= 5:
        wins = len(combo_trades[combo_trades['realized_rr'] > 1.0])
        avg_rr = combo_trades['realized_rr'].mean()
        winrate = (wins / total_calls) * 100
        expectancy = avg_rr * (winrate / 100)
        
        main_results.append({
            'Flag': f"{flag.replace('flag_', '')}",
            'Calls': total_calls,
            'Winrate': round(winrate, 2),
            'Avg_RR': round(avg_rr, 3),
            'Expectancy': round(expectancy, 3),
            'Flag': flag
        })

# (flag 3)
for flag1, flag2, flag3, flag4, flag5 in combinations(flag_columns, 5):
    mask = (df[flag1] == True) & (df[flag2] == True) & (df[flag3] == True) & (df[flag4] == True) & (df[flag5] == True)
    combo_trades = df[mask]
    total_calls = len(combo_trades)
    if total_calls >= 10:
        wins = len(combo_trades[combo_trades['realized_rr'] > 1.0])
        avg_rr = combo_trades['realized_rr'].mean()
        winrate = (wins / total_calls) * 100
        expectancy = avg_rr * (winrate / 100)
        main_combo3_results.append({
            'Flag_Combo': f"{flag1.replace('flag_', '')} + {flag2.replace('flag_', '')} + {flag3.replace('flag_', '')} + {flag4.replace('flag_', '')} + {flag5.replace('flag_', '')}",
            'Calls': total_calls,
            'Winrate': round(winrate, 2),
            'Avg_RR': round(avg_rr, 3),
            'Expectancy': round(expectancy, 3),
            'Flag1': flag1,
            'Flag2': flag2,
            'Flag3': flag3,
            'Flag4': flag4,
            'Flag5': flag5
        })

# (main_flag 4)
for flag1, flag2, flag3, flag4 in combinations(flag_columns, 4):
    mask = (df[flag1] == True) & (df[flag2] == True) & (df[flag3] == True) & (df[flag4] == True)
    combo_trades = df[mask]
    total_calls = len(combo_trades)
    if total_calls >= 10:
        wins = len(combo_trades[combo_trades['realized_rr'] > 1.0])
        avg_rr = combo_trades['realized_rr'].mean()
        winrate = (wins / total_calls) * 100
        expectancy = avg_rr * (winrate / 100)
        main_combo4_results.append({
            'Flag_Combo': f"{flag1.replace('flag_', '')} + {flag2.replace('flag_', '')} + {flag3.replace('flag_', '')} + {flag4.replace('flag_', '')}",
            'Calls': total_calls,
            'Winrate': round(winrate, 2),
            'Avg_RR': round(avg_rr, 3),
            'Expectancy': round(expectancy, 3),
            'Flag1': flag1,
            'Flag2': flag2,
            'Flag3': flag3,
            'Flag4': flag4
        })

# Convert to DataFrames
main_df = pd.DataFrame(main_results)
main_combo3_df = pd.DataFrame(main_combo3_results)
main_combo4_df = pd.DataFrame(main_combo4_results)

if not main_df.empty:
    main_df_sorted = main_df.sort_values('Expectancy', ascending=False)

if not main_combo3_df.empty:
    main_combo3_df_sorted = main_combo3_df.sort_values('Expectancy', ascending=False)
    print("\n🏆 TOP MAIN + 3 FLAG COMBINATIONS (by Expectancy)")
    print(main_combo3_df_sorted[['Flag_Combo', 'Calls', 'Winrate', 'Avg_RR', 'Expectancy']].head(5).to_string(index=False))

if not main_combo4_df.empty:
    main_combo4_df_sorted = main_combo4_df.sort_values('Expectancy', ascending=False)
    print("\n🏆 TOP MAIN + 4 FLAG COMBINATIONS (by Expectancy)")
    print(main_combo4_df_sorted[['Flag_Combo', 'Calls', 'Winrate', 'Avg_RR', 'Expectancy']].head(5).to_string(index=False))

# =====================================================================
# SAVE OUTPUTS
# =====================================================================
if not main_df.empty:
    main_df_sorted.to_csv('flag_combinations.csv', index=False)
if not main_combo3_df.empty:
    main_combo3_df_sorted.to_csv('mainflag_combinations_3way.csv', index=False)
if not main_combo4_df.empty:
    main_combo4_df_sorted.to_csv('mainflag_combinations_4way.csv', index=False)

print("\n✅ Main trigger combination results saved:")
print("   - mainflag_combinations.csv  (1 flags)")
print("   - mainflag_combinations_2way.csv  (main + 2 flags)")
print("   - mainflag_combinations_3way.csv  (main + 3 flags)")
print("   - mainflag_combinations_4way.csv  (main + 4 flags)")
print("=" * 120)
