"""
simulate_from_candles_refactor.py

Changes from previous:
 - Balance is NOT modified at entry/stack. Balance updates only on realized PnL (exit events).
 - Stacking limited to MAX_STACK_LEGS (3).
 - Timeout (force close) uses the LAST stacked entry timestamp to compute elapsed bars.
 - Fees are applied on exit (both entry-side and exit-side are charged and deducted when trade closes).
 - Other logic kept consistent: stop-to-BEP rule measured from first entry; move-to-BEP uses consecutive bars > MOVE_AFTER_R.
"""

import sqlite3
import pandas as pd

# -------- USER PARAMETERS (tweak these) ----------
SIGNALS_PQ = "database/parquet/event_trades.parquet"     # path to your signals parquet
DB_FILE = "database/btc_historical.db"  # sqlite DB with OHLCV
TABLE = "btc_15m_ohlcv_clean"           # table name containing columns ts, open, high, low, close, volume
TIMEFRAME = "15m"                       # descriptive only
INITIAL_BALANCE = 5.0
RISK_PER_LEG = 0.025                    # 2.5% per stacked entry (policy A)
FEE_RATE_PER_SIDE = 0.00055             # taker fee per side
MOVE_AFTER_R = 1                        # move stop after position exceeds 2R
MOVE_AFTER_BARS = 4                     # number of bars of being > MOVE_AFTER_R before moving stop
MOVE_TO_R = 0.25                        # new stop placed at entry +/- 0.25R
MAX_STACK_LEGS = 1                      # limit stacking to 3 legs
# Lookahead/timeouts: define in bars (matches your snippet logic)
BARS_PER_DAY = 24 * 4    # if 15m bars => 96 per day
BARS_PER_HOUR = 4        # 15m -> 4
LOOKAHEAD_S_DAYS = 3
LOOKAHEAD_A_DAYS = 1
LOOKAHEAD_B_HOURS = 4
MAX_POSITION_NOTIONAL = 2000

OUTPUT_EQUITY_CSV = "database/csv/equity_curve_from_candles.csv"
# -------------------------------------------------

# small helpers
def compute_r_price_distance(entry_price, stop_price):
    return abs(entry_price - stop_price)

def compute_rr_from_price(entry_price, exit_price, stop_price, direction):
    r_dist = compute_r_price_distance(entry_price, stop_price)
    if r_dist == 0:
        return 0.0
    if direction == "long":
        profit = exit_price - entry_price
    else:
        profit = entry_price - exit_price
    return profit / r_dist

# ------------------ load signals ------------------
signals = pd.read_parquet(SIGNALS_PQ)
required_cols = ['entry_ts','entry_price','stop','target','direction','tier','main_trigger']
for c in required_cols:
    if c not in signals.columns:
        raise ValueError(f"signals parquet missing required column: {c}")

signals['entry_ts'] = pd.to_datetime(signals['entry_ts'])

# cutoff date
start = pd.Timestamp("2025-10-21", tz="UTC")

# day of month helpers
signals["day"] = signals["entry_ts"].dt.day
signals["days_in_month"] = signals["entry_ts"].dt.days_in_month

# filter only A and S + date range + exclude 1st & last days of month
signals = signals[
    (signals["entry_ts"] >= start) &
    (signals["tier"].isin(["A", "S"])) &
    (signals["day"] != 2) &  # remove 1st
    (signals["day"] < signals["days_in_month"] - 1) # remove last day
].sort_values("entry_ts").reset_index(drop=True)


# ------------------ load candles ------------------
first_signal = signals['entry_ts'].min()
last_signal = signals['entry_ts'].max()

max_bars_lookahead = max(
    LOOKAHEAD_S_DAYS * BARS_PER_DAY,
    LOOKAHEAD_A_DAYS * BARS_PER_DAY,
    LOOKAHEAD_B_HOURS * BARS_PER_HOUR
)

conn = sqlite3.connect(DB_FILE)
# use parameterized query and naive datetimes (no timezone token)
start_ts = (first_signal - pd.Timedelta(days=1)).tz_convert(None)
end_ts   = (last_signal + pd.Timedelta(days=LOOKAHEAD_A_DAYS + 2)).tz_convert(None)

query = f"""
SELECT ts, open, high, low, close, volume
FROM {TABLE}
WHERE ts >= '{start_ts}' AND ts <= '{end_ts}'
ORDER BY ts
"""
candles = pd.read_sql(query, conn, parse_dates=["ts"])
conn.close()

if candles.empty:
    raise RuntimeError("No candles pulled from DB. Check DB_FILE, TABLE, and time range.")

# aggregate identical ts rows to VWAP-style OHLCV if needed
candles = candles.groupby("ts").apply(
    lambda x: pd.Series({
        "open": (x["open"] * x["volume"]).sum() / x["volume"].sum(),
        "high": x["high"].max(),
        "low": x["low"].min(),
        "close": (x["close"] * x["volume"]).sum() / x["volume"].sum(),
        "volume": x["volume"].sum()
    })
).reset_index()

ts_to_idx = {ts: idx for idx, ts in enumerate(candles['ts'])}

# ------------------ Simulator ------------------
class Position:
    def __init__(self, first_signal_row, balance_at_first_entry):
        self.direction = first_signal_row['direction']
        self.tier = first_signal_row['tier']
        self.common_stop = float(first_signal_row['stop'])
        self.common_target = float(first_signal_row['target'])
        self.entries = []  # list of dicts: entry_ts, entry_price, size_units, risk_money, entry_notional
        self.first_entry_ts = pd.to_datetime(first_signal_row['entry_ts'])
        self.first_entry_price = float(first_signal_row['entry_price'])
        self.stop_active = self.common_stop
        self.stop_moved_to_bep = False
        self.bars_above_move_R = 0
        self.current_position_risk = 0.0
        self.balance_at_first_entry = balance_at_first_entry

    def add_leg_sized_to_risk_1pct(self, entry_price, balance_now, risk_per_leg=RISK_PER_LEG):
        entry_price = float(entry_price)
        d = abs(entry_price - self.common_stop)
        if d <= 0:
            return None
        R_money = risk_per_leg * balance_now
        size_units = R_money / d
        entry_notional = size_units * entry_price

        # Apply MAX_POSITION_NOTIONAL constraint
        if entry_notional > MAX_POSITION_NOTIONAL:
            entry_notional = MAX_POSITION_NOTIONAL
            size_units = entry_notional / entry_price
            R_money = size_units * d  # adjust risk accordingly

        elif size_units < 0.001:
            size_units = 0.001
            entry_notional = size_units * entry_price
            R_money = size_units * d  # adjust risk accordingly

        self.entries.append({
            'entry_ts': None,  # caller will set
            'entry_price': entry_price,
            'size_units': size_units,
            'risk_money': R_money,
            'entry_notional': entry_notional
        })
        self.current_position_risk += R_money
        return size_units

    def last_entry_ts(self):
        if not self.entries:
            return self.first_entry_ts
        return pd.to_datetime(self.entries[-1]['entry_ts'])

    def compute_unrealized_r(self, mark_price):
        # R-price unit is first_entry_price - common_stop
        R_price = abs(self.first_entry_price - self.common_stop)
        if R_price == 0:
            return 0.0
        unreal_pnl = 0.0
        for leg in self.entries:
            if self.direction == 'long':
                unreal_pnl += (mark_price - leg['entry_price']) * leg['size_units']
            else:
                unreal_pnl += (leg['entry_price'] - mark_price) * leg['size_units']
        unreal_r = unreal_pnl / R_price
        return unreal_r

    def total_entry_notional(self):
        return sum([leg['entry_notional'] for leg in self.entries])

    def total_exit_notional_at_price(self, price):
        return sum([leg['size_units'] * price for leg in self.entries])

# Simulation loop
balance = INITIAL_BALANCE
equity_rows = []
active_pos = None
signal_i = 0
num_signals = len(signals)
stopped_out = False

def get_max_bars_for_tier(tier):
    if tier == "S":
        return LOOKAHEAD_S_DAYS * BARS_PER_DAY
    elif tier == "A":
        return LOOKAHEAD_A_DAYS * BARS_PER_DAY
    else:
        return LOOKAHEAD_B_HOURS * BARS_PER_HOUR

for idx, row in candles.iterrows():
    current_ts = row['ts']
    high = float(row['high'])
    low = float(row['low'])
    close = float(row['close'])
    open_p = float(row['open'])

    # process signals whose entry_ts <= current candle ts
    while signal_i < num_signals and signals.loc[signal_i, 'entry_ts'] <= current_ts:
        sig = signals.loc[signal_i]
        sig_ts = sig['entry_ts']
        sig_price = float(sig['entry_price'])
        sig_direction = sig['direction']
        sig_tier = sig['tier']
        sig_stop = float(sig['stop'])
        sig_target = float(sig['target'])

        # Pass if Error in Stop Placement
        if (sig_direction == "long" and sig_stop >= sig_price):
            signal_i += 1
            continue
        
        if  (sig_direction == "short" and sig_stop <= sig_price):
            signal_i += 1
            continue

        # If there is no active position, create one and add first leg (do NOT change balance)
        if active_pos is None:
            active_pos = Position(sig, balance_at_first_entry=balance)
            size_units = active_pos.add_leg_sized_to_risk_1pct(sig_price, balance_now=balance, risk_per_leg=RISK_PER_LEG)
            if size_units is None:
                # degenerate; skip
                active_pos = None
            else:
                active_pos.entries[-1]['entry_ts'] = sig_ts
                # record entry event (no balance change)
                equity_rows.append({
                    'ts': sig_ts,
                    'event': 'entry',
                    'balance': float(round(balance, 8)),
                    'position_notional': float(active_pos.total_entry_notional()),
                    'num_legs': len(active_pos.entries)
                })
        else:
            # If same direction and we haven't reached max stack, add a leg (do NOT change balance)
            if sig_direction == active_pos.direction:
                if len(active_pos.entries) < MAX_STACK_LEGS:
                    size_units = active_pos.add_leg_sized_to_risk_1pct(sig_price, balance_now=balance, risk_per_leg=RISK_PER_LEG)
                    if size_units is not None:
                        active_pos.entries[-1]['entry_ts'] = sig_ts
                        equity_rows.append({
                            'ts': sig_ts,
                            'event': 'stack_entry',
                            'balance': float(round(balance, 8)),
                            'position_notional': float(active_pos.total_entry_notional()),
                            'num_legs': len(active_pos.entries)
                        })
                    # else degenerate
                else:
                    # reached max stacks, ignore this stacking signal (log it)
                    equity_rows.append({
                        'ts': sig_ts,
                        'event': 'stack_ignored_max_legs',
                        'balance': float(round(balance, 8)),
                        'position_notional': float(active_pos.total_entry_notional()),
                        'num_legs': len(active_pos.entries)
                    })
            else:
                # opposite direction -> force close active position at current candle open, realize PnL and fees, then allow signal to open fresh position
                exit_price = open_p
                pnl = 0.0
                entry_notional_total = active_pos.total_entry_notional()
                exit_notional_total = active_pos.total_exit_notional_at_price(exit_price)
                for leg in active_pos.entries:
                    if active_pos.direction == "long":
                        pnl += (exit_price - leg['entry_price']) * leg['size_units']
                    else:
                        pnl += (leg['entry_price'] - exit_price) * leg['size_units']
                # fees: entry-side fees were not deducted at entry; deduct both sides now using actual notionals
                fee_entry = entry_notional_total * FEE_RATE_PER_SIDE
                fee_exit = exit_notional_total * FEE_RATE_PER_SIDE
                total_fees = fee_entry + fee_exit
                balance += pnl
                balance -= total_fees
                win = True if pnl - total_fees > 0 else False

                equity_rows.append({
                    'ts': current_ts,
                    'event': 'forced_close_on_opposite_win' if win else 'forced_close_on_opposite_lose',
                    'balance': float(round(balance, 8)),
                    'pnl': float(round(pnl, 8)),
                    'fees': float(round(total_fees, 8)),
                    'num_legs': len(active_pos.entries)
                })

                active_pos = None

        if equity_rows:
            last_event = equity_rows[-1]['event']
            if last_event != 'forced_close_on_opposite_lose' and last_event != 'forced_close_on_opposite_win':
                signal_i += 1
            else:
                # do not increment so the same signal will be processed again as a fresh entry
                pass
        else:
            signal_i += 1

    # If there is an active position, evaluate exits
    if active_pos is not None:
        first_entry_ts = active_pos.entries[0]['entry_ts']
        if current_ts == first_entry_ts:
            # don’t check stop/target on entry candle
            pass
        else:
            # normal exit checks here
            # For move-to-BEP: measure consecutive bars where unrealized_R >= MOVE_AFTER_R using mark=close
            if not active_pos.stop_moved_to_bep:
                unreal_r = active_pos.compute_unrealized_r(close)
                if unreal_r >= MOVE_AFTER_R:
                    active_pos.bars_above_move_R += 1
                else:
                    active_pos.bars_above_move_R = 0
                if active_pos.bars_above_move_R >= MOVE_AFTER_BARS:
                    entry_price_first = active_pos.first_entry_price
                    r_price = abs(entry_price_first - active_pos.common_stop)
                    if active_pos.direction == "long":
                        new_stop = entry_price_first + r_price * MOVE_TO_R
                    else:
                        new_stop = entry_price_first - r_price * MOVE_TO_R
                    active_pos.stop_active = new_stop
                    active_pos.stop_moved_to_bep = True
                    equity_rows.append({
                        'ts': current_ts,
                        'event': 'stop_moved_to_bep',
                        'balance': float(round(balance,8)),
                        'new_stop': new_stop,
                        'num_legs': len(active_pos.entries)
                    })

            # Determine intrabar hits (conservative assumption: stop-first if both in same candle)
            tgt = active_pos.common_target
            stop_active = active_pos.stop_active
            hit_target = False
            hit_stop = False

            if active_pos.direction == "long":
                if high >= tgt:
                    hit_target = True
                if low <= stop_active:
                    hit_stop = True
                if hit_target and hit_stop:
                    chosen = 'stop'   # conservative default
                    if chosen == 'target':
                        hit_stop = False
                    else:
                        hit_target = False
            else:
                if low <= tgt:
                    hit_target = True
                if high >= stop_active:
                    hit_stop = True
                if hit_target and hit_stop:
                    chosen = 'stop'
                    if chosen == 'target':
                        hit_stop = False
                    else:
                        hit_target = False

            if hit_target or hit_stop:
                if hit_target:
                    exit_price = tgt
                    outcome = 'target_hit'
                else:
                    exit_price = stop_active
                    outcome = 'stop_hit' if not active_pos.stop_moved_to_bep else 'BEP'

                pnl = 0.0
                entry_notional_total = active_pos.total_entry_notional()
                exit_notional_total = active_pos.total_exit_notional_at_price(exit_price)
                for leg in active_pos.entries:
                    if active_pos.direction == "long":
                        pnl += (exit_price - leg['entry_price']) * leg['size_units']
                    else:
                        pnl += (leg['entry_price'] - exit_price) * leg['size_units']

                fee_entry = entry_notional_total * FEE_RATE_PER_SIDE
                fee_exit = exit_notional_total * FEE_RATE_PER_SIDE
                total_fees = fee_entry + fee_exit

                balance += pnl
                balance -= total_fees

                equity_rows.append({
                    'ts': current_ts,
                    'event': outcome,
                    'balance': float(round(balance,8)),
                    'pnl': float(round(pnl,8)),
                    'fees': float(round(total_fees,8)),
                    'num_legs': len(active_pos.entries)
                })

                active_pos = None

                if balance <= 0:
                    stopped_out = True
                    break

                continue

        # Timeout check measured from LAST stacked entry (not first)
        last_entry_ts = active_pos.last_entry_ts()
        if last_entry_ts in ts_to_idx:
            bars_since_last = idx - ts_to_idx[last_entry_ts]
        else:
            td = (current_ts - last_entry_ts).total_seconds()
            bars_since_last = max(0, int(td / (15*60)))  # assume 15m bars

        max_bars = get_max_bars_for_tier(active_pos.tier)
        if bars_since_last >= max_bars:
            # force close at market (close price)
            exit_price = close
            pnl = 0.0
            entry_notional_total = active_pos.total_entry_notional()
            exit_notional_total = active_pos.total_exit_notional_at_price(exit_price)
            for leg in active_pos.entries:
                if active_pos.direction == "long":
                    pnl += (exit_price - leg['entry_price']) * leg['size_units']
                else:
                    pnl += (leg['entry_price'] - exit_price) * leg['size_units']

            fee_entry = entry_notional_total * FEE_RATE_PER_SIDE
            fee_exit = exit_notional_total * FEE_RATE_PER_SIDE
            total_fees = fee_entry + fee_exit

            balance += pnl
            balance -= total_fees

            equity_rows.append({
                'ts': current_ts,
                'event': 'timeout_force_close',
                'balance': float(round(balance,8)),
                'pnl': float(round(pnl,8)),
                'fees': float(round(total_fees,8)),
                'num_legs': len(active_pos.entries)
            })

            active_pos = None

            if balance <= 0:
                stopped_out = True
                break

# End candles loop

# Final close if still active
if active_pos is not None and not stopped_out:
    first_entry_ts = active_pos.entries[0]['entry_ts']
    if current_ts == first_entry_ts:
        # don’t check stop/target on entry candle
        pass
    else:
        # normal exit checks here
        exit_price = float(candles.iloc[-1]['close'])
        pnl = 0.0
        entry_notional_total = active_pos.total_entry_notional()
        exit_notional_total = active_pos.total_exit_notional_at_price(exit_price)
        for leg in active_pos.entries:
            if active_pos.direction == "long":
                pnl += (exit_price - leg['entry_price']) * leg['size_units']
            else:
                pnl += (leg['entry_price'] - exit_price) * leg['size_units']

        fee_entry = entry_notional_total * FEE_RATE_PER_SIDE
        fee_exit = exit_notional_total * FEE_RATE_PER_SIDE
        total_fees = fee_entry + fee_exit

        balance += pnl
        balance -= total_fees

        equity_rows.append({
            'ts': candles.iloc[-1]['ts'],
            'event': 'final_close_end_of_data',
            'balance': float(round(balance,8)),
            'pnl': float(round(pnl,8)),
            'fees': float(round(total_fees,8)),
            'num_legs': len(active_pos.entries)
        })
        active_pos = None

# Build equity df and stats
equity_df = pd.DataFrame(equity_rows).sort_values('ts').reset_index(drop=True)
pd.options.display.float_format = '{:,.8f}'.format

# --- compute max drawdown ---
equity_df['peak'] = equity_df['balance'].cummax()
equity_df['drawdown'] = equity_df['balance'] / equity_df['peak'] - 1.0
max_drawdown = equity_df['drawdown'].min()  # this will be negative (e.g. -0.25 for -25%)

final_balance = balance
num_trades_executed = equity_df[equity_df['event'].isin([
    'target_hit','stop_hit','BEP',
    'timeout_force_close',
    'forced_close_on_opposite_lose',
    'forced_close_on_opposite_win',
    'final_close_end_of_data'
])].shape[0]

win_events = equity_df[equity_df['event'].isin([
    'target_hit','BEP','forced_close_on_opposite_win'
])]
win_rate = (len(win_events) / max(1, num_trades_executed)) * 100 if num_trades_executed>0 else 0.0

equity_df.to_csv(OUTPUT_EQUITY_CSV, index=False)

print("=== SIMULATION SUMMARY (refactor) ===")
print(f"Initial balance: {INITIAL_BALANCE:.8f}")
print(f"Final balance:   {final_balance:.8f}")
print(f"Trades closed:   {num_trades_executed}")
print(f"Win rate (approx): {win_rate:.2f}%")
print("Max Drawdown: %.2f%%" % (max_drawdown*100))
print(f"Stopped out (balance<=0): {stopped_out}")
print(f"Equity CSV saved to: {OUTPUT_EQUITY_CSV}")
