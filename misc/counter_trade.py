import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score

import optuna
from optuna.samplers import TPESampler

# === 1. Load and Preprocess Data ===
print("Loading data...")
df = pd.read_parquet('database/model/event_trades.parquet')

# Set pos (preferred) metrics based on initial preference
df["max_profit_pos"] = np.nan
df["max_dd_pos"] = np.nan
df["time_to_max_profit_pos"] = np.nan
df["time_to_max_dd_pos"] = np.nan

# Initial Guess
df["is_long"] = (df["direction"] == "long").astype(int)

# Preferred long
mask_long = df["direction"] == "long"
df.loc[mask_long, "max_profit_pos"] = df.loc[mask_long, "highest_move"]
df.loc[mask_long, "max_dd_pos"] = df.loc[mask_long, "lowest_move"]
df.loc[mask_long, "time_to_max_profit_pos"] = df.loc[mask_long, "time_to_high"]
df.loc[mask_long, "time_to_max_dd_pos"] = df.loc[mask_long, "time_to_low"]

# Preferred short
mask_short = df["direction"] == "short"
df.loc[mask_short, "max_profit_pos"] = -df.loc[mask_short, "lowest_move"]
df.loc[mask_short, "max_dd_pos"] = -df.loc[mask_short, "highest_move"]
df.loc[mask_short, "time_to_max_profit_pos"] = df.loc[mask_short, "time_to_low"]
df.loc[mask_short, "time_to_max_dd_pos"] = df.loc[mask_short, "time_to_high"]


# Rule 1: Dynamic flip check with threshold
def check_flip(row):
    # Check magnitude threshold: Flip only if DD is significant relative to profit
    profit = abs(row["max_profit_pos"])
    dd = abs(row["max_dd_pos"])
    ratio = profit / dd if dd > 0 else float('inf')
    ratio_counter = dd / profit if profit > 0 else float('inf')

    # Flip if it no profit at all
    if row["time_to_max_profit_pos"] == 0:
        return True
    elif ratio == float('inf'):
        return False
    # Dynamic threshold: Flip if ratio < 3 (profit not 3x+ DD) or DD > 0.1 vol absolute
    elif ratio_counter > 3:
        return True
    elif row["time_to_max_dd_pos"] < row["time_to_max_profit_pos"]:
        if ratio < 2:
            return True

    return False

df["needs_flip"] = df.apply(check_flip, axis=1)
print(f"Number of rows with value 'needs_flip': {len(df[df['needs_flip'] == 1])}")

# Continuous flip_score (0-1): Higher if flip needed (DD fast/large vs profit)
def compute_flip_score(row):
    profit_vol = abs(row["max_profit_pos"])
    dd_vol = abs(row["max_dd_pos"])
    ratio = profit_vol / dd_vol if dd_vol > 0 else float('inf')
    time_ratio = row["time_to_max_dd_pos"] / row["time_to_max_profit_pos"] if row["time_to_max_profit_pos"] > 0 else 0
    
    # Sigmoid: Blends ratio/time (higher if low ratio + fast DD)
    if time_ratio == 0:
        return 1.0 # No Profit -> flip
    if ratio == float('inf'):
        return 0.0  # No DD → no flip
    score = 1 / (1 + np.exp(-(1/ratio - time_ratio)))  # 0-1, high if low ratio + fast DD
    return np.clip(score, 0, 1)

df["flip_score"] = df.apply(compute_flip_score, axis=1)
print(f"Mean flip_score: {df['flip_score'].mean():.4f}")

# Feature engineering: Add interactions/ratios to capture non-linearity
feature_cols = [c for c in df.columns if c.startswith("ml_")]
feature_cols += ["is_long"]

X = df[feature_cols].astype(float)   # convert TRUE/FALSE → 1/0

# === 2. Model Training for Flip Score ===
print("Training model for predicting flip score...")
df = df.sort_values('entry_ts').reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=12)

# === 3. Optuna Objective for Regression ===
def create_regression_objective(Y):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),  # Wider for more learning
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),  # Lower min for nuance
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),  # Added for noise
            "random_state": 33,
            "n_jobs": -1,
            "device": "gpu"
        }

        fold_scores = []
        fold_auc = []  # Extra: Binarized AUC
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(70, verbose=-1)]
            )

            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            fold_scores.append(-mae)

            # Binarized AUC (for classification-like eval)
            y_val_bin = (y_val > 0.5).astype(int)
            preds_bin = (preds > 0.5).astype(int)
            auc = roc_auc_score(y_val_bin, preds_bin) if len(np.unique(y_val_bin)) > 1 else 0.5
            fold_auc.append(auc)

        print(f"Trial mean AUC (binarized): {np.mean(fold_auc):.4f}")  # Log for tuning
        return np.mean(fold_scores)

    return objective

# === Tune Model ===
print("\n=== Tuning Model ===")
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=33))
study.optimize(create_regression_objective(df["flip_score"]), n_trials=100, show_progress_bar=True)  # More trials

print(f"Best CV score (negative MAE): {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Save
best_params = study.best_params
best_params.update({"random_state": 33, "n_jobs": -1})
final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(X, df["flip_score"])
final_model.booster_.save_model('database/model/regressor.json')

import joblib
joblib.dump(feature_cols, "database/model/BTC_features.pkl")
print("\nRegression models trained and saved!")

# Try
pred_flip = final_model.predict(X)

# Add to df
df['pred_flip_score'] = pred_flip

# === Save full results to CSV ===
output_csv = 'flip_inference_results.csv'
df_to_save = df[['entry_ts', 'direction', 'needs_flip',
                 'flip_score', 'pred_flip_score']].copy()

df_to_save.to_csv(output_csv, index=False)
print(f"\nFull inference results saved to: {output_csv}")

# === Summary Statistics ===
print("\n" + "="*50)
print("INFERENCE SUMMARY")
print("="*50)

print(f"Total rows: {len(df)}")

print(f"\nError Metrics:")
print(f"  MAE Flip: {mean_absolute_error(df['flip_score'], df['pred_flip_score']):.4f}")

print(f"\nPercentiles of Predicted Flip:")
print(df['pred_flip_score'].describe()[['min', '25%', '50%', '75%', 'max']])
