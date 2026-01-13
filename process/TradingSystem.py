import time
import threading
from datetime import datetime, timedelta, timezone
from utility.bybit import BybitTrader

# ---------------------------
# CONFIG / HELPERS
# ---------------------------

LOOKAHEAD_MAP = {
    "S": timedelta(days=3),
    "A": timedelta(days=1),
    "B": timedelta(hours=4),
    "C": timedelta(hours=4),
}

# persistent settings you can tune
DEFAULTS = {
    "max_risk_pct": 0.025,   # 2.5%
    "profit_rr_to_trail": 1,
    "wait_after_reach": timedelta(hours=1),
    "trail_sl_rr": 0.25,
    "monitor_interval_s": 15,    # background monitor loop interval
}

def utc_now():
    return datetime.now(timezone.utc)

def round_contracts_to_step(contracts: int, step: int = 0.001):
    # contract step is usually integer 1; keep this for extensibility
    return max(0.001, int((contracts // step) * step or step))


# ---------------------------
# Trading System
# ---------------------------

class TradingSystem:
    def __init__(self,
                 bybit: BybitTrader,
                 max_risk_pct=DEFAULTS["max_risk_pct"],
                 n_positions_allowed=2,
                 allow_hedge=False,
                 monitor_interval_s=DEFAULTS["monitor_interval_s"]):
        self.bybit = bybit
        self.max_risk_pct = max_risk_pct
        self.n_positions_allowed = n_positions_allowed
        self.allow_hedge = allow_hedge
        self.monitor_interval_s = monitor_interval_s

        # active trades tracked locally (dictionary of internal_id -> trade_info)
        # trade_info fields: entry_ts, entry_price, sl, tp, side, qty_contracts, tier, lookahead_deadline,
        # opened (True/False), reached_profit_time (datetime or None), rr (computed)
        self.active_trades = {}
        self._id_counter = 0
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._stop_event = threading.Event()

        # instrument filters
        self.instrument = self.bybit.get_instrument_filters()

    def _gen_id(self):
        self._id_counter += 1
        return f"t{self._id_counter}"

    def compute_qty_contracts_from_risk(self, entry_price, sl_price):
        """Compute contracts such that risk <= max_risk_pct * capital.
           Returns integer contracts already rounded to contract step.
        """
        capital = self.bybit.get_capital()
        risk_amount = capital * self.max_risk_pct
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit <= 0:
            raise ValueError("Invalid SL/entry difference")
        qty_btc = risk_amount / risk_per_unit

        if qty_btc < 0.001:
            qty_btc = 0.001
        contracts = round_contracts_to_step(qty_btc, 0.001)
        return contracts

    def _count_active_same_direction(self, side):
        # side is "Buy" or "Sell"
        with self._lock:
            return sum(1 for t in self.active_trades.values() if t["side"] == side and t["opened"])
   
    def _has_conflicting_opposite(self, side):
        # If hedging not allowed and there exists opposite opened trade
        if self.allow_hedge:
            return False
        opposite = "Sell" if side == "Buy" else "Buy"
        return self._count_active_same_direction(opposite) > 0

    def add_signal(self, entry_price: float, sl: float, tp: float, side: str, tier: str, signal_ts: datetime = None):
        """
        Add a trading signal. Returns internal trade id if accepted, else None.
        side: "Buy" or "Sell"
        tier: "S","A","B","C"
        """
        if signal_ts is None:
            signal_ts = utc_now()
        tier = tier.upper()
        lookahead = LOOKAHEAD_MAP.get(tier, timedelta(hours=4))
        deadline = signal_ts + lookahead

        # enforce N positions and hedging rules
        if self._has_conflicting_opposite(side):
            # must force-close opposite positions before entering (policy: reject and caller can choose to close)
            opposite = "Sell" if side == "Buy" else "Buy"
            print(f"Hedging not allowed. Found opposite {opposite} trades -> closing them before opening new {side} signal.")

            with self._lock:
                opposite_trades = [tid for tid, t in self.active_trades.items()
                                if t["side"] == opposite and t["opened"]]

            for tid in opposite_trades:
                try:
                    trade = self.active_trades[tid]
                    close_side = side  # open direction closes opposite
                    print(f"[{utc_now()}] Closing opposite trade {tid} ({opposite}) to allow new {side} entry.")
                    self.bybit.place_market_order(
                        qty_contracts=trade["qty_contracts"],
                        side=close_side,
                        reduceOnly=True
                    )
                    with self._lock:
                        self.active_trades.pop(tid, None)
                except Exception as e:
                    print(f"Error closing opposite trade {tid}: {e}")

        n_same = self._count_active_same_direction(side)
        if n_same >= self.n_positions_allowed:
            print(f"Already {n_same} same-direction positions (limit {self.n_positions_allowed}) --> reject signal")
            return None

        # compute contracts
        try:
            contracts = self.compute_qty_contracts_from_risk(entry_price, sl)
        except Exception as e:
            print("Failed to compute contract size:", e)
            return None

        # create internal trade record (not yet opened on exchange)
        tid = self._gen_id()
        trade = {
            "id": tid,
            "entry_ts": signal_ts,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "side": side,
            "tier": tier,
            "lookahead_deadline": deadline,
            "qty_contracts": contracts,
            "opened": False,
            "opened_at": None,
            "open_exec_price": None,
            "reached_profit_time": None,   # when 1.5RR first reached
            "rr": (tp - entry_price) / (entry_price - sl) if side == "Buy" else (entry_price - tp) / (sl - entry_price)
        }
        with self._lock:
            self.active_trades[tid] = trade

        # place order immediately (market)
        try:
            print(f"[{utc_now()}] Placing market order for trade {tid}, contracts={contracts}, side={side}")
            resp = self.bybit.place_market_order(qty_contracts=contracts, side=side, tp=tp, sl=sl)
            # store exchange response
            trade["opened"] = True
            trade["opened_at"] = utc_now()
            trade["open_exec_price"] = self.bybit.get_ticker_price()
            with self._lock:
                self.active_trades[tid] = trade
            print(f"Placed trade {tid}: {resp}")
            return tid
        except Exception as e:
            print("Error placing order:", e)
            with self._lock:
                # remove failed trade
                self.active_trades.pop(tid, None)
            return None

    def _monitor_once(self):
        now = utc_now()
        try:
            last_price = self.bybit.get_ticker_price()
        except Exception as e:
            print("Ticker fetch error:", e)
            return

        with self._lock:
            trade_items = list(self.active_trades.items())

        for tid, trade in trade_items:
            if not trade["opened"]:
                continue

            # ---------------------------
            # NEW: check if still active on exchange
            # ---------------------------
            try:
                pos_info = self.bybit.get_position_info()
                if pos_info is None:
                    # Trade was closed on exchange (TP/SL/manual/liquidation)
                    last_closed = self.bybit.get_last_closed_pnl()
                    reason = last_closed["reason"] if last_closed else "unknown"
                    print(f"[{now}] Trade {tid} closed on exchange. Reason={reason}. Removing from active_trades.")
                    with self._lock:
                        self.active_trades.pop(tid, None)
                    continue
            except Exception as e:
                print("Position fetch error:", e)
                continue

            if pos_info is None or pos_info.get("size", 0) == 0:
                # means trade was closed on exchange (TP/SL or manual)
                print(f"[{now}] Trade {tid} closed on exchange (TP/SL hit). Removing from active_trades.")
                with self._lock:
                    self.active_trades.pop(tid, None)
                continue

            # ---------------------------
            # Existing monitoring logic
            # ---------------------------
            side = trade["side"]
            entry = trade["entry_price"]
            sl = trade["sl"]
            tp = trade["tp"]
            rr_unit = abs(entry - sl)
            profit = (last_price - entry) if side == "Buy" else (entry - last_price)
            current_rr = profit / rr_unit if rr_unit > 0 else 0

            # 1) Reached 1.5RR
            if trade["reached_profit_time"] is None and current_rr >= DEFAULTS["profit_rr_to_trail"]:
                trade["reached_profit_time"] = now
                print(f"[{now}] Trade {tid} reached {DEFAULTS['profit_rr_to_trail']}RR at price {last_price}")

            # 2) Sustained profit → trail SL
            if trade["reached_profit_time"] is not None:
                if now >= trade["reached_profit_time"] + DEFAULTS["wait_after_reach"]:
                    if current_rr >= DEFAULTS["profit_rr_to_trail"]:
                        new_sl = entry + rr_unit * DEFAULTS["trail_sl_rr"] if side == "Buy" else entry - rr_unit * DEFAULTS["trail_sl_rr"]
                        try:
                            print(f"[{now}] Trailing SL for {tid} to {new_sl} (was {trade['sl']})")
                            self.bybit.set_trading_stop(stopLoss=new_sl)
                            trade["sl"] = new_sl
                        except Exception as e:
                            print("Error updating SL:", e)
                    trade["reached_profit_time"] = None

            # 3) Lookahead expiry
            if now >= trade["lookahead_deadline"]:
                print(f"[{now}] Lookahead expired for {tid} -> force close")
                try:
                    close_side = "Sell" if side == "Buy" else "Buy"
                    self.bybit.place_market_order(qty_contracts=trade["qty_contracts"],
                                                side=close_side, reduceOnly=True)
                except Exception as e:
                    print("Error force closing:", e)
                with self._lock:
                    self.active_trades.pop(tid, None)
            else:
                with self._lock:
                    self.active_trades[tid] = trade

    def start_monitor(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            print("Monitor already running")
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("Monitor started")

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                self._monitor_once()
            except Exception as e:
                print("Monitor iteration error:", e)
            time.sleep(self.monitor_interval_s)

    def stop_monitor(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("Monitor stopped")

    def manual_close_trade(self, tid):
        with self._lock:
            trade = self.active_trades.get(tid)
        if not trade:
            print("No such trade:", tid)
            return
        side = trade["side"]
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            resp = self.bybit.place_market_order(qty_contracts=trade["qty_contracts"],
                                                 side=close_side, reduceOnly=True)
            print("Manual close resp:", resp)
            with self._lock:
                self.active_trades.pop(tid, None)
        except Exception as e:
            print("Error manual closing:", e)

    def list_active_trades(self):
        with self._lock:
            return {k: v.copy() for k, v in self.active_trades.items()}