import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Masking, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Force GPU
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU enabled")
    except RuntimeError as e:
        print(e)
        
# === 1. Load and Preprocess Data ===
print("Loading data...")
df = pd.read_parquet('database/model/event_trades.parquet')
feature_cols = [c for c in df.columns if c.startswith("ml_")]
tscv = TimeSeriesSplit(n_splits=12)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)
print("Any NaN in df:", df.isnull().any().any())

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

# Preprocess sequences
seq_vol = df[[f'ml_vol_spike_c{i}' for i in range(1,7)]].values  # [rows, 6]
seq_oi = df[[f'ml_oi_spike_c{i}' for i in range(1,5)]].values
seq_oi = np.pad(seq_oi, ((0,0), (0,2)), mode='constant', constant_values=0)  # Pad to 6

# Stack as [rows, timesteps=6, channels=2 (vol+oi)]
X_seq = np.stack([seq_vol, seq_oi], axis=-1)  # [n_rows, 6, 2]

# Normalize (per channel, robust to outliers)
scaler = StandardScaler()
X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
X_seq_scaled = scaler.fit_transform(X_seq_reshaped)
X_seq = X_seq_scaled.reshape(X_seq.shape)

# Static features (non-seq)
static_cols = [c for c in feature_cols if c not in [f'ml_vol_spike_c{i}' for i in range(1,7)] + [f'ml_oi_spike_c{i}' for i in range(1,5)]]
X_static = df[static_cols].values

# Scale static
scaler_static = StandardScaler()
X_static = scaler_static.fit_transform(X_static)

# LSTM model with fixed params
units = 64
layers = 1
dropout = 0.2
lr = 0.001
epochs = 50
batch_size = 32

fold_mae = []
fold_auc = []
for train_idx, val_idx in tscv.split(X_seq):
    X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
    X_static_train, X_static_val = X_static[train_idx], X_static[val_idx]
    y_train, y_val = df["flip_score"].iloc[train_idx].values, df["flip_score"].iloc[val_idx].values

    # Functional model
    seq_input = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
    static_input = Input(shape=(X_static.shape[1],))

    x = Masking(mask_value=0.0)(seq_input)  # Mask 0s
    x = LSTM(units, activation='tanh', recurrent_activation='sigmoid')(x)  # Tanh stable
    x = Dropout(dropout)(x)

    combined = Concatenate()([x, static_input])
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)  # Clip gradients!
    model.compile(optimizer=optimizer, loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit([X_seq_train, X_static_train], y_train,
              epochs=epochs, batch_size=batch_size,
              validation_data=([X_seq_val, X_static_val], y_val),
              callbacks=[early_stop], verbose=1)

    preds = model.predict([X_seq_val, X_static_val]).flatten()
    preds = np.clip(preds, 0, 1)  # Force 0-1

    mae = mean_absolute_error(y_val, preds)
    fold_mae.append(mae)

    y_val_bin = (y_val > 0.5).astype(int)
    auc = roc_auc_score(y_val_bin, preds)
    fold_auc.append(auc)

    print(f"Fold MAE: {mae:.4f}, AUC: {auc:.4f}")

print(f"Mean MAE: {np.mean(fold_mae):.4f}")
print(f"Mean AUC: {np.mean(fold_auc):.4f}")

# Final train
model.fit([X_seq, X_static], df["flip_score"].values, epochs=epochs, batch_size=batch_size, verbose=1)
model.save('lstm_flip_model_fixed.h5')