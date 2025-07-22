import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Φόρτωση και επεξεργασία δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X = df.drop(columns=exclude_cols, errors='ignore')
X = X.loc[:, X.dtypes != 'object']
y = df['target']

# === 2. Split ΠΡΙΝ την κανονικοποίηση ===
SEQ_LEN = 24
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

# Εφαρμογή raw split
X_values = X.values
y_values = y.values

X_seq, y_seq = create_sequences(X_values, y_values, SEQ_LEN)
split = int(len(X_seq) * 0.8)
X_train_raw, X_test_raw = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === 3. Κανονικοποίηση ΜΕΤΑ το split ===
n_features = X_train_raw.shape[2]
scaler = StandardScaler()

# Χρειαζόμαστε reshaping για να κανονικοποιήσουμε per feature
X_train_flat = X_train_raw.reshape(-1, n_features)
X_test_flat = X_test_raw.reshape(-1, n_features)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_raw.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_raw.shape)

# === 4. Μοντέλο CNN → BiLSTM → Double Attention ===
def cnn_bilstm_double_attention(seq_len, input_dim):
    inp = Input(shape=(seq_len, input_dim))
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # 1ο Attention
    attn1 = layers.Dense(1, activation='tanh')(x)
    w1 = layers.Softmax(axis=1)(attn1)
    c1 = layers.Multiply()([x, w1])
    c1 = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(c1)

    # 2ο Attention
    x2 = layers.RepeatVector(seq_len)(c1)
    x2 = layers.Concatenate()([x, x2])
    x2 = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x2)

    attn2 = layers.Dense(1, activation='tanh')(x2)
    w2 = layers.Softmax(axis=1)(attn2)
    c2 = layers.Multiply()([x2, w2])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(c2)

    x = layers.Dense(64, activation='relu')(context)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return models.Model(inputs=inp, outputs=out)

# === 5. Εκπαίδευση ===
model = cnn_bilstm_double_attention(SEQ_LEN, X_train_scaled.shape[2])
model.compile(optimizer='adam', loss='mae')

callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# === 6. Αξιολόγηση ===
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ CNN → BiLSTM → Double Attention RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 7. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("CNN → BiLSTM → Double Attention Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
