import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

import tensorflow as tf

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

# === Custom Attention Layer ===
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


def create_head(x, name):
    y = Dense(64, activation='relu', name=f'{name}_dense1')(x)
    y = Dropout(0.3, name=f'{name}_dropout')(y)
    return Dense(1, name=name)(y)


# === Load & Preprocess ===
print("Loading data...")
df_btc = pd.read_parquet('database/model/btc/event_trades.parquet')
df_eth = pd.read_parquet('database/model/eth/event_trades.parquet')
df_sol = pd.read_parquet('database/model/sol/event_trades.parquet')
df = pd.concat([df_btc, df_eth, df_sol], ignore_index=True)

feature_cols = [c for c in df.columns if c.startswith("ml_")]

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

df["is_long"] = (df["direction"] == "long").astype(int)

df["current_vol"] = df["price_delta_z32"].replace(0, np.nan).fillna(method='ffill').clip(lower=0.1)
df["hm_vol"] = df["highest_move"] / abs(df["current_vol"])
df["lm_vol"] = df["lowest_move"] / abs(df["current_vol"])

df['hm_vol'] = np.clip(df['hm_vol'], -2, 2)
df['lm_vol'] = np.clip(df['lm_vol'], -2, 2)

df['time_to_high'] = np.clip(df['time_to_high'], 0, 1440)
df['time_to_low'] = np.clip(df['time_to_low'], 0, 1440)

# Sequences
seq_vol = df[[f'ml_vol_spike_c{i}' for i in range(1,7)]].values
seq_oi  = df[[f'ml_oi_spike_c{i}' for i in range(1,5)]].values
seq_oi  = np.pad(seq_oi, ((0,0), (0,2)), mode='constant', constant_values=0)

X_seq = np.stack([seq_vol, seq_oi], axis=-1)

# Normalize sequences per channel
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

# Sort by time
df = df.sort_values('entry_ts').reset_index(drop=True)

# Normalize time targets
time_scaler = StandardScaler()
df[['tth_norm', 'ttl_norm']] = time_scaler.fit_transform(
    df[['time_to_high', 'time_to_low']]
)

targets = ['hm_vol', 'lm_vol', 'tth_norm', 'ttl_norm']

# Hyperparameters
UNITS = 192
DROPOUT = 0.45
BATCH_SIZE = 96
EPOCHS = 120
LR = 0.0015

tscv = TimeSeriesSplit(n_splits=12)

print("\n" + "="*70)
print("Starting Multi-Task BiLSTM + Attention Training")
print("="*70)

fold_maes = {t: [] for t in targets}

for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X_seq), 1):
    print(f"\nFold {fold_num}/{tscv.n_splits}")
    
    X_seq_tr, X_seq_val = X_seq[train_idx], X_seq[val_idx]
    X_stat_tr, X_stat_val = X_static[train_idx], X_static[val_idx]
    
    y_tr = {t: df[t].iloc[train_idx].values for t in targets}
    y_val = {t: df[t].iloc[val_idx].values for t in targets}
    
    # ── Model Definition ── (using LIST outputs!)
    seq_input = Input(shape=(X_seq.shape[1], X_seq.shape[2]), name='seq_input')
    static_input = Input(shape=(X_static.shape[1],), name='static_input')
    
    x = Bidirectional(LSTM(UNITS, return_sequences=True,
                           activation='tanh', recurrent_activation='sigmoid'),
                      name='bi_lstm_1')(seq_input)
    x = Dropout(DROPOUT, name='dropout_1')(x)
    
    x = Bidirectional(LSTM(UNITS//2, return_sequences=True,
                           activation='tanh', recurrent_activation='sigmoid'),
                      name='bi_lstm_2')(x)
    x = Dropout(DROPOUT, name='dropout_2')(x)
    
    x = AttentionLayer(name='attention')(x)
    x = Dropout(DROPOUT, name='dropout_attention')(x)
    
    # Optional shared dense (recommended)
    x = Dense(128, activation='relu', name='shared_dense')(x)
    x = Dropout(0.3, name='shared_dropout')(x)
    
    combined = Concatenate(name='concat')([x, static_input])
    
    # Outputs as LIST
    outputs = [create_head(combined, f'output_{t}') for t in targets]
    
    model = Model(inputs=[seq_input, static_input], outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss=['mse'] * len(targets),
        loss_weights=[2.0, 2.0, 0.8, 0.8],   # higher weight on moves
        metrics=['mae'] * len(targets)
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=18, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        [X_seq_tr, X_stat_tr],
        [y_tr[t] for t in targets],
        validation_data=([X_seq_val, X_stat_val], [y_val[t] for t in targets]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    preds = model.predict([X_seq_val, X_stat_val], verbose=0)  # returns list
    for i, t in enumerate(targets):
        mae = mean_absolute_error(y_val[t], preds[i].flatten())
        fold_maes[t].append(mae)
        print(f"{t} - MAE: {mae:.4f}")

# CV Summary
print("\n" + "="*70)
print("Cross-Validation Results (Mean MAE ± std)")
print("-"*70)
for t in targets:
    mean_mae = np.mean(fold_maes[t])
    std_mae = np.std(fold_maes[t])
    print(f"{t:18} : {mean_mae:.4f} ± {std_mae:.4f}")
print("="*70)

# === Final model ===
print("\nTraining FINAL multi-task model on full dataset...")

seq_input = Input(shape=(X_seq.shape[1], X_seq.shape[2]), name='seq_input')
static_input = Input(shape=(X_static.shape[1],), name='static_input')

x = Bidirectional(LSTM(UNITS, return_sequences=True, name='bi_lstm_1_final'))(seq_input)
x = Dropout(DROPOUT, name='dropout_1_final')(x)
x = Bidirectional(LSTM(UNITS//2, return_sequences=True, name='bi_lstm_2_final'))(x)
x = Dropout(DROPOUT, name='dropout_2_final')(x)
x = AttentionLayer(name='attention_final')(x)
x = Dropout(DROPOUT, name='dropout_attention_final')(x)
x = Dense(128, activation='relu', name='shared_dense_final')(x)
x = Dropout(0.3, name='shared_dropout_final')(x)

combined = Concatenate(name='concat_final')([x, static_input])

outputs = [create_head(combined, f'output_{t}') for t in targets]

final_model = Model(inputs=[seq_input, static_input], outputs=outputs)

final_model.compile(
    optimizer=Adam(learning_rate=LR, clipnorm=1.0),
    loss=['mse'] * len(targets),
    loss_weights=[2.0, 2.0, 0.8, 0.8],
    metrics=['mae'] * len(targets)
)

final_model.fit(
    [X_seq, X_static],
    [df[t].values for t in targets],
    epochs=150,
    batch_size=BATCH_SIZE,
    verbose=1
)

final_model.save('database/model/multi_task_bilstm_attention_all_targets.keras')
print("Final multi-task model saved!")

print("\nTraining complete.")