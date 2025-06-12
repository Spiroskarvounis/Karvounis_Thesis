import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models

# === 1. Φόρτωση και προεπεξεργασία δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X = df.drop(columns=exclude_cols, errors='ignore')
X = X.loc[:, X.dtypes != 'object']
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 2. Δημιουργία χρονοπαραθύρων ===
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 24
X_seq, y_seq = create_sequences(X_scaled, y.values, SEQ_LEN)
X_train, X_test = X_seq[:int(0.8*len(X_seq))], X_seq[int(0.8*len(X_seq)):]
y_train, y_test = y_seq[:int(0.8*len(y_seq))], y_seq[int(0.8*len(y_seq)):]

# === 3. LSTM + Attention Layer ===
def attention_block(inputs):
    score = layers.Dense(1, activation='tanh')(inputs)
    weights = layers.Softmax(axis=1)(score)
    context = layers.Multiply()([inputs, weights])
    context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    return context

inputs = layers.Input(shape=(SEQ_LEN, X_train.shape[2]))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = attention_block(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1)(x)

model = models.Model(inputs, output)
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# === 4. Αξιολόγηση ===
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ LSTM + Attention RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 5. Γράφημα ===
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
