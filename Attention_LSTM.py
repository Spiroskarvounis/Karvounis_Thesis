import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Επιλογή αριθμητικών ===
exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X = df.drop(columns=exclude_cols, errors='ignore')
X = X.loc[:, X.dtypes != 'object']
y = df['target'].values

# === 3. Sliding windows ===
SEQ_LEN = 24
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

X_values = X.values
X_seq_all, y_seq_all = create_sequences(X_values, y, SEQ_LEN)

# === 4. Split ΠΡΙΝ το scaling ===
split = int(len(X_seq_all) * 0.8)
X_train_raw, X_test_raw = X_seq_all[:split], X_seq_all[split:]
y_train, y_test = y_seq_all[:split], y_seq_all[split:]

# === 5. Scaling ΜΕΤΑ το split ===
n_features = X_train_raw.shape[2]
scaler = StandardScaler()

X_train_flat = X_train_raw.reshape(-1, n_features)
X_test_flat = X_test_raw.reshape(-1, n_features)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_raw.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_raw.shape)

# === 6. LSTM + Attention ===
def attention_block(inputs):
    score = layers.Dense(1, activation='tanh')(inputs)
    weights = layers.Softmax(axis=1)(score)
    context = layers.Multiply()([inputs, weights])
    context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    return context

inputs = layers.Input(shape=(SEQ_LEN, n_features))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = attention_block(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1)(x)

model = models.Model(inputs, output)
model.compile(optimizer='adam', loss='mse')

# === 7. Εκπαίδευση ===
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# === 8. Αξιολόγηση ===
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ LSTM + Attention RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 9. Γράφημα ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("LSTM + Attention Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
