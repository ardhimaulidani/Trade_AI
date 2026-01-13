import time
import json
from pathlib import Path
import pandas as pd
import utility.yaml_loader as loader
from utility.bybit import BybitTrader
from process.TradingSystem import TradingSystem

cfg = loader.load_config("config/rules_config.yaml")

PARQUET_FILE = Path(cfg.files.out_trades)
STATE_FILE = Path("config/trade_state.parquet")

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                if "last_seen_ts" in state and state["last_seen_ts"] is not None:
                    return pd.to_datetime(state["last_seen_ts"], utc=True)
        except Exception:
            pass
    return None

def save_state(last_seen_ts):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_seen_ts": str(last_seen_ts) if last_seen_ts is not None else None}, f)

def get_new_trades(parquet_file, last_seen_ts):
    """Return only tier A and S trades newer than last_seen_ts."""
    if not parquet_file.exists():
        return [], last_seen_ts

    df = pd.read_parquet(parquet_file)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")

    # filter by timestamp
    if last_seen_ts is not None:
        df = df[df["entry_ts"] > last_seen_ts]

    # filter by tier (only A)
    df = df[df["tier"].str.upper().isin(["A", "S"])]

    trades = []
    for _, row in df.iterrows():
        trades.append({
            "entry_ts": row["entry_ts"],
            "entry_price": float(row["entry_price"]),
            "sl": float(row["stop"]),
            "tp": float(row["target"]),
            "side": "Buy" if str(row["direction"]).lower() == "long" else "Sell",
            "tier": str(row["tier"]).upper(),
        })

    # update last_seen_ts
    if not df.empty:
        last_seen_ts = df["entry_ts"].max()

    return trades, last_seen_ts

if __name__ == "__main__":
    # instantiate wrapper and trading system
    bybit = BybitTrader(key_file="config/bybit_config.json", testnet=False, symbol="BTCUSDT")
    system = TradingSystem(bybit,
                           max_risk_pct=0.025,
                           n_positions_allowed=1,
                           allow_hedge=False)

    # set leverage to max
    try:
        print("Setting leverage to max:", bybit.set_leverage_to_max())
    except Exception as e:
        print("Warning: set leverage error:", e)

    # start background monitor
    system.start_monitor()

    # load state (last seen timestamp)
    last_seen_ts = load_state()
    if last_seen_ts is None and PARQUET_FILE.exists():
        df_init = pd.read_parquet(PARQUET_FILE)
        df_init["entry_ts"] = pd.to_datetime(df_init["entry_ts"], utc=True, errors="coerce")
        if not df_init.empty:
            last_seen_ts = df_init["entry_ts"].max()
            save_state(last_seen_ts)

    print("Starting executor. Last seen entry_ts:", last_seen_ts)

    try:
        while True:
            new_trades, last_seen_ts = get_new_trades(PARQUET_FILE, last_seen_ts)

            for t in new_trades:
                print(f"📢 New trade {t['side']} {t['entry_price']} tp={t['tp']} sl={t['sl']} tier={t['tier']}")

                retries = 2
                tid = None
                while retries > 0 and tid is None:
                    tid = system.add_signal(entry_price=t["entry_price"],
                                            sl=t["sl"],
                                            tp=t["tp"],
                                            side=t["side"],
                                            tier=t["tier"],
                                            signal_ts=t["entry_ts"])
                    if tid:
                        print(f"✅ Trade {tid} executed successfully")
                        break
                    else:
                        retries -= 1
                        print(f"⚠️ Trade execution failed, retrying... ({3-retries}/3)")
                        time.sleep(5)

                if tid is None:
                    print("❌ Trade could not be placed after retries, skipping.")

            if new_trades:
                save_state(last_seen_ts)

            time.sleep(120)  # check every 60s
    except KeyboardInterrupt:
        system.stop_monitor()
        print("Stopped by user")
