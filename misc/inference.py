from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras.backend as K

# === Custom Attention Layer (must be defined before loading) ===
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(name='attention_u',
                                 shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.squeeze(K.dot(uit, K.expand_dims(self.u)), axis=-1)
        ait = K.softmax(ait, axis=1)
        ait = K.expand_dims(ait, axis=-1)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super().get_config()


# GPU setup
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# === Load & Preprocess Data (aligned with training) ===
print("Loading inference data...")
df_btc = pd.read_parquet('database/model/btc/event_trades.parquet')
df_eth = pd.read_parquet('database/model/eth/event_trades.parquet')
df_sol = pd.read_parquet('database/model/sol/event_trades.parquet')

# Add asset column
df_btc['asset'] = 'BTC'
df_eth['asset'] = 'ETH'
df_sol['asset'] = 'SOL'

df = pd.concat([df_btc, df_eth, df_sol], ignore_index=True)

feature_cols = [c for c in df.columns if c.startswith("ml_")]

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

df["is_long"] = (df["direction"] == "long").astype(int)

df["current_vol"] = df["price_delta_z32"].replace(0, np.nan).fillna(method='ffill').clip(lower=0.1)
df["highest_move_vol"] = df["highest_move"] / abs(df["current_vol"])
df["lowest_move_vol"] = df["lowest_move"] / abs(df["current_vol"])

# Keep clipped versions for evaluation
df['highest_move_vol_clipped'] = np.clip(df['highest_move_vol'], -2, 2)
df['lowest_move_vol_clipped']  = np.clip(df['lowest_move_vol'], -2, 2)

# Sequences
seq_vol = df[[f'ml_vol_spike_c{i}' for i in range(1,7)]].values
seq_oi  = df[[f'ml_oi_spike_c{i}' for i in range(1,5)]].values
seq_oi  = np.pad(seq_oi, ((0,0), (0,2)), mode='constant', constant_values=0)

X_seq = np.stack([seq_vol, seq_oi], axis=-1)

# Normalize sequences (must match training!)
scaler_seq = StandardScaler()
for ch in range(2):
    X_seq[:,:,ch] = scaler_seq.fit_transform(X_seq[:,:,ch].reshape(-1,1)).reshape(X_seq.shape[0], -1)

# Static features
static_cols = [c for c in feature_cols 
               if c not in [f'ml_vol_spike_c{i}' for i in range(1,7)] + [f'ml_oi_spike_c{i}' for i in range(1,5)]]
static_cols += ["is_long", "price_delta_z32", "price_delta_z16"]

X_static = df[static_cols].values
scaler_static = StandardScaler()
X_static = scaler_static.fit_transform(X_static)

# === Time scaler (for denormalization of predictions) ===
time_scaler = StandardScaler()
time_scaler.fit(df[['time_to_high', 'time_to_low']])  # same fit as in training

# === Load the multi-task model ===
print("Loading multi-task BiLSTM model...")
model = load_model(
    'database/model/multi_task_bilstm_attention_all_targets.keras',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# === Inference ===
print("Running inference...")
predictions = model.predict([X_seq, X_static], verbose=0)  # returns list of 4 arrays

df['pred_high_vol']      = predictions[0].flatten()   # highest_move_vol
df['pred_low_vol']       = predictions[1].flatten()   # lowest_move_vol
df['pred_time_high_norm']= predictions[2].flatten()   # normalized
df['pred_time_low_norm'] = predictions[3].flatten()   # normalized

# Denormalize time predictions back to minutes
time_preds = np.column_stack([df['pred_time_high_norm'], df['pred_time_low_norm']])
time_preds_minutes = time_scaler.inverse_transform(time_preds)

df['pred_time_to_high'] = time_preds_minutes[:, 0]
df['pred_time_to_low']  = time_preds_minutes[:, 1]

# Price move estimates
df['pred_highest_move'] = df['pred_high_vol'] * abs(df['current_vol'])
df['pred_lowest_move']  = df['pred_low_vol'] * abs(df['current_vol'])

# === Summary Statistics ===
print("\n" + "="*60)
print("MULTI-TASK MODEL INFERENCE SUMMARY")
print("="*60)

total_rows = len(df)
print(f"Total rows: {total_rows}")

print(f"\nHighest Move (vol units):")
print(f"  Mean actual: {df['highest_move_vol'].mean():.4f}")
print(f"  Mean predicted: {df['pred_high_vol'].mean():.4f}")
print(f"  Mean predicted price move: {df['pred_highest_move'].mean() * 100:.2f}%")

print(f"\nLowest Move (vol units):")
print(f"  Mean actual: {df['lowest_move_vol'].mean():.4f}")
print(f"  Mean predicted: {df['pred_low_vol'].mean():.4f}")
print(f"  Mean predicted price move: {df['pred_lowest_move'].mean() * 100:.2f}%")

print(f"\nTime to High/Low (minutes):")
print(f"  MAE Time to High: {mean_absolute_error(df['time_to_high'], df['pred_time_to_high']):.1f}")
print(f"  MAE Time to Low:  {mean_absolute_error(df['time_to_low'], df['pred_time_to_low']):.1f}")

print(f"\nError Metrics (vol units):")
print(f"  MAE highest_move_vol: {mean_absolute_error(df['highest_move_vol'], df['pred_high_vol']):.4f}")
print(f"  MAE lowest_move_vol:  {mean_absolute_error(df['lowest_move_vol'], df['pred_low_vol']):.4f}")

print(f"\nMAE Price Moves (%):")
print(f"  MAE highest_move: {mean_absolute_error(df['highest_move'], df['pred_highest_move']) * 100:.4f}%")
print(f"  MAE lowest_move:  {mean_absolute_error(df['lowest_move'], df['pred_lowest_move']) * 100:.4f}%")

# === Flip Score Analysis (your original logic) ===
df["max_profit_pos"] = np.where(df["direction"] == "long", df["highest_move_vol"], -df["lowest_move_vol"])
df["max_dd_pos"]     = np.where(df["direction"] == "long", df["lowest_move_vol"], -df["highest_move_vol"])
df["time_to_max_profit_pos"] = np.where(df["direction"] == "long", df["time_to_high"], df["time_to_low"])
df["time_to_max_dd_pos"]     = np.where(df["direction"] == "long", df["time_to_low"], df["time_to_high"])

# Predicted side
df["max_profit_pred"] = np.where(df["direction"] == "long", df["pred_high_vol"], -df["pred_low_vol"])
df["max_dd_pred"]     = np.where(df["direction"] == "long", df["pred_low_vol"], -df["pred_high_vol"])
df["time_to_max_profit_pred"] = np.where(df["direction"] == "long", df["pred_time_to_high"], df["pred_time_to_low"])
df["time_to_max_dd_pred"]     = np.where(df["direction"] == "long", df["pred_time_to_low"], df["pred_time_to_high"])

def compute_flip_score(row):
    profit_vol = abs(row["max_profit_pos"])
    dd_vol = abs(row["max_dd_pos"])
    if dd_vol == 0:
        return 0.0
    ratio = profit_vol / dd_vol
    time_ratio = row["time_to_max_dd_pos"] / row["time_to_max_profit_pos"] if row["time_to_max_profit_pos"] > 0 else 0
    if time_ratio == 0:
        return 1.0
    score = 1 / (1 + np.exp(-(1/ratio - time_ratio)))
    return np.clip(score, 0, 1)

df["flip_score"] = df.apply(compute_flip_score, axis=1)
print(f"Mean actual flip_score: {df['flip_score'].mean():.4f}")

def compute_flip_score_pred(row):
    profit_vol = abs(row["max_profit_pred"])
    dd_vol = abs(row["max_dd_pred"])
    if dd_vol == 0:
        return 0.0
    ratio = profit_vol / dd_vol
    time_ratio = row["time_to_max_dd_pred"] / row["time_to_max_profit_pred"] if row["time_to_max_profit_pred"] > 0 else 0
    if time_ratio == 0:
        return 1.0
    score = 1 / (1 + np.exp(-(1/ratio - time_ratio)))
    return np.clip(score, 0, 1)

df["flip_score_pred"] = df.apply(compute_flip_score_pred, axis=1)
print(f"Mean predicted flip_score: {df['flip_score_pred'].mean():.4f}")
print(f"MAE flip_score: {mean_absolute_error(df['flip_score'], df['flip_score_pred']):.4f}")

# === Per-Asset Quarterly Analysis ===
print("\n" + "="*60)
print("PER-ASSET QUARTERLY METRICS")
print("="*60)

assets = ['BTC', 'ETH', 'SOL']
for asset in assets:
    sub_df = df[df['asset'] == asset].copy()
    if sub_df.empty:
        print(f"No data for {asset}")
        continue
    
    # Create quarterly groups
    sub_df['quarter'] = sub_df['entry_ts'].dt.to_period('Q')
    
    quarterly_groups = sub_df.groupby('quarter')
    
    print(f"\n--- {asset} Quarterly Metrics ---")
    
    results = []
    for qtr, group in quarterly_groups:
        if len(group) < 1:
            continue
        
        mae_high_vol = mean_absolute_error(group['highest_move_vol'], group['pred_high_vol'])
        mae_low_vol = mean_absolute_error(group['lowest_move_vol'], group['pred_low_vol'])
        mean_actual_flip = group['flip_score'].mean()
        mae_flip = mean_absolute_error(group['flip_score'], group['flip_score_pred'])
        mean_pred_flip = group['flip_score_pred'].mean()
        
        results.append({
            'Quarter': str(qtr),
            'Num Events': len(group),
            'MAE High Vol': f"{mae_high_vol:.4f}",
            'MAE Low Vol': f"{mae_low_vol:.4f}",
            'Mean Actual Flip': f"{mean_actual_flip:.4f}",
            'MAE Flip': f"{mae_flip:.4f}",
            'Mean Pred Flip': f"{mean_pred_flip:.4f}"
        })
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
    else:
        print("No quarterly data available.")

# === Save results ===
output_file = 'multi_task_inference_results.xlsx'
df_to_save = df[['entry_ts', 'direction', 'highest_move_vol', 'pred_high_vol',
                 'lowest_move_vol', 'pred_low_vol', 'pred_highest_move', 'pred_lowest_move',
                 'time_to_high', 'pred_time_to_high', 'time_to_low', 'pred_time_to_low',
                 'flip_score', 'flip_score_pred']].copy()

df_to_save['entry_ts'] = df_to_save['entry_ts'].dt.tz_localize(None)
df_to_save.to_excel(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\nInference complete!")