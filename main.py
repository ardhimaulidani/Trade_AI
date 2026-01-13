import os
import logging
import builtins

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from process.BarProcessor import BarProcessor
from process.TradeScorer import TradeScorer
from process.TradeSimulator import TradeSimulator
from process.TradePicker import TradePicker

from utility.discord import Discord
import utility.yaml_loader as loader
import utility.helpers as utility

# -------------------------
# These objects will be set in main and passed to worker initializer
# -------------------------
def init_worker(globs, model_counter, model_pred, model_dd, features):
    """Initializer for process pool: set global small objects for worker"""
    utility.setup_logging()
    global G, MLModel_counter, MLModel_pred, MLModel_dd, MLModel_features
    G = globs
    MLModel_counter = model_counter
    MLModel_pred = model_pred
    MLModel_dd = model_dd
    MLModel_features = features

# -------------------------
# Core per-row worker: receives integer index into arrays
# -------------------------
def process_index(cfg, i):
    try:
        # Prepare Row
        ts = G['ts_list'][i]
        entry_price = float(G['close_list'][i])
        trades_out = []

        # Use TradeScorer
        tradeScorer = TradeScorer(G, i, cfg)

        # Get Key Level Suite
        lvl_candidates = tradeScorer.get_keylevel()
        
        # Search for Trigger Reasons (multi-direction dict)
        triggers = tradeScorer.search_trigger()
        if not triggers:
            return None

        # Score Trades (multi-output dict)
        scored = tradeScorer.score_trade(MLModel_counter, MLModel_pred, MLModel_dd, MLModel_features, triggers)

        # Process all directions found
        for direction, trigger_list in triggers.items():
            # Countertrade Flip Direction
            dir = direction
            final_tier, score, flags, MLFeatures = scored[direction]
            if flags["flag_countertrade"] == True:
                if dir == "long": dir = "short"
                elif dir == "short": dir = "long"

            logging.debug(f"[{dir}] tier: {final_tier}, score: {score}")
            logging.debug(f"[{dir}] trigger reasons: {trigger_list}")

            # Stage: stop and target choice
            tradePicker = TradePicker(cfg)
            # BETA: Pick from Prediction Movement Model
            best_tgt, best_stop, best_rr = tradePicker._pick_from_score(
            entry_price, dir, score, final_tier, lvl_candidates
            )

            # best_tgt, best_stop, best_rr = tradePicker._pick_non_rotation(
            # entry_price, dir, final_tier, lvl_candidates
            # )
            logging.debug(f"[{dir}] stop={best_stop}, tgt={best_tgt}, RR={best_rr}")


            if best_tgt is not None and best_stop is not None:
                # --- Format trigger reasons ---
                trigger_reason_pub = ", ".join(trigger_list) if trigger_list else "N/A"
                logging.debug(f"[{dir}] trigger_reason_pub: {trigger_reason_pub}")
                
                signal = {
                    "entry_ts": ts,
                    "entry_price": entry_price,
                    "direction": dir,
                    "tier": final_tier,
                    "stop": best_stop,
                    "target": best_tgt,
                    "trigger": trigger_reason_pub
                }

                # Set Discord Alert
                if final_tier in ("A", "S"):
                    discord_client = Discord(_debug=True)
                    # discord_client.publish_trade_embed(signal)

                logging.debug(f"[{dir}] Published to Discord")

                # Trade Simulation
                tradeSimulator = TradeSimulator(G, cfg)
                    
                if cfg.mode == loader.MODE.DATASET:
                    # Simulate Highest and Lowest Percentage
                    exit_price, exit_ts, highest_move, highest_idx, lowest_move, lowest_idx = tradeSimulator.simulate_all(
                        i, entry_price
                    )

                    MLData = {
                        "entry_ts": ts,
                        "entry_price": entry_price,
                        "direction": dir,
                        "main_trigger": trigger_reason_pub,
                        "raw_score": score,
                        "exit_ts": exit_ts,
                        "exit_price": exit_price,
                        "highest_move": highest_move,
                        "time_to_high": highest_idx * 15,
                        "lowest_move": lowest_move,
                        "time_to_low": lowest_idx * 15
                    }
                    MLData.update(MLFeatures)
                    trades_out.append(MLData)

                elif cfg.mode == loader.MODE.REALTIME or cfg.mode == loader.MODE.DEBUG:
                    if cfg.mode == loader.MODE.REALTIME:
                        # Simulate Trade RR
                        exit_price, exit_ts, outcome, realized_rr, highest_rr = tradeSimulator.simulate(
                            i, entry_price, best_stop, best_tgt, dir, final_tier
                        )
                        lowest_rr = 0
                    else:
                        # Simulate Highest and Lowest RR
                        exit_price, exit_ts, outcome, realized_rr, highest_rr, lowest_rr = tradeSimulator.simulate_no_stop(
                            i, entry_price, best_stop, best_tgt, dir, final_tier
                        )

                        # Append to parquet
                        trade = {
                            "entry_ts": ts,
                            "entry_price": entry_price,
                            "direction": dir,
                            "main_trigger": trigger_reason_pub,   # now it's a list
                            "raw_score": score,
                            "tier": final_tier,
                            "target": best_tgt,
                            "stop": best_stop,
                            "best_rr": best_rr,
                            "outcome": outcome,
                            "exit_ts": exit_ts,
                            "exit_price": exit_price,
                            "realized_rr": realized_rr,
                            "highest_rr": highest_rr,
                            "lowest_rr": lowest_rr
                        }
                        trade.update(flags)
                        trades_out.append(trade)

        # Return multiple trades
        if not trades_out:
            return None
        if len(trades_out) == 1:
            return trades_out[0]
        return trades_out

    except Exception as e:
        logging.error(f"{e}")
        return None


# -------------------------
# Main: load data, precompute arrays and maps, spawn pool
# -------------------------

# BETA: - Load ML Model
import joblib
import lightgbm as lgb
from xgboost import XGBRegressor

def main_once(asset, cfg):
    # resolve parquet paths
    if asset == "btc":
        PARQUET_OUT = cfg.btc_files.out_trades
        SUMMARY_OUT = cfg.btc_files.out_summary
    elif asset == "eth":
        PARQUET_OUT = cfg.eth_files.out_trades
        SUMMARY_OUT = cfg.eth_files.out_summary
    else:  # sol
        PARQUET_OUT = cfg.sol_files.out_trades
        SUMMARY_OUT = cfg.sol_files.out_summary

    model_pred = XGBRegressor()
    model_pred.load_model("database/model/profit_model.json")

    model_dd = XGBRegressor()
    model_dd.load_model("database/model/dd_model.json")

    model_counter = lgb.Booster(model_file="database/model/regressor.json")

    feature_cols = joblib.load("database/model/BTC_features.pkl")
    
    # Init Processor Class
    barProcessor = BarProcessor(cfg)

    # Load All BTC Data
    barProcessor.load_data(asset)

    # Do Preprocessing
    barProcessor.preprocess_bars()

    # Check for new entries
    new_indices = barProcessor.get_new_indices()
    if not new_indices:
        logging.info("No new bars to process.")
        return

    logging.info(f"Processing {len(new_indices)} new bars...")
    globs = barProcessor.prepare_globals()

    # --- Run processing only on new indices ---
    results = []
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker, initargs=(globs,model_counter,model_pred,model_dd,feature_cols)) as exe:
        futures = {exe.submit(process_index, cfg, idx): idx for idx in new_indices}
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            results.extend(r if isinstance(r, builtins.list) else [r])

    if results:
        # --- Save trades incrementally ---
        new_trades = pd.DataFrame(results)
        if os.path.exists(PARQUET_OUT):
            old_trades = pd.read_parquet(PARQUET_OUT)
            all_trades = pd.concat([old_trades, new_trades], ignore_index=True)
        else:
            all_trades = new_trades
        all_trades.to_parquet(PARQUET_OUT, index=False)

        # --- Display Trade Performance Summary ---
        # if cfg.mode == loader.MODE.REALTIME or cfg.mode == loader.MODE.DEBUG:
        #     summary = new_trades.groupby("tier").agg(
        #         total_trades=("tier","count"),
        #         target_hit_rate=("outcome", lambda x: (x=="target_hit").mean()),
        #         avg_realized_rr=("realized_rr","mean"),
        #         avg_max_rr=("highest_rr","mean"),
        #         avg_low_rr=("lowest_rr", "mean"),
        #         max_rr=("highest_rr","max"),
        #         min_rr=("lowest_rr", "min"),
        #         win_rate=("realized_rr", lambda x: (x>0).mean())
        #     ).reset_index().sort_values("tier", ascending=False)
        #     summary.to_csv(SUMMARY_OUT, index=False)
        #     print("Tier summary saved to", SUMMARY_OUT)
        #     print(summary)
    else:
        logging.info("No trade trigger available.")

    # Save Last Timestamp State
    # if cfg.mode == loader.MODE.REALTIME:
    #     barProcessor.save_state()

if __name__ == "__main__":
    # Setup Logging
    utility.setup_logging()

    # Init Config YAML
    cfg = loader.load_config("config/rules_config.yaml")
    
    ASSETS = ["btc"]

    for asset in ASSETS:
        logging.info(f"Main process tiering started for {asset}.")
        main_once(asset, cfg)
