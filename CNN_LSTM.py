import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === 1. Φόρτωση & προετοιμασία δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Επιλογή αριθμητικών χαρακτηριστικών ===
X = df.select_dtypes(include=[np.number]).drop(columns=['target'])
y = df['target'].values.reshape(-1, 1)

# === 3. Δημιουργία ακολουθιών ===
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24
X_np = X.values
X_seq_all, y_seq_all = create_sequences(X_np, y, TIME_STEPS)

# === 4. Split ===
split = int(0.8 * len(X_seq_all))
X_train_raw, X_test_raw = X_seq_all[:split], X_seq_all[split:]
y_train_raw, y_test_raw = y_seq_all[:split], y_seq_all[split:]

# === 5. Scaling ΜΕΤΑ το split ===
n_features = X_train_raw.shape[2]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# reshape για scaling
X_train_flat = X_train_raw.reshape(-1, n_features)
X_test_flat = X_test_raw.reshape(-1, n_features)

X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train_raw.shape)
X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test_raw.shape)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# === 6. CNN-LSTM Μοντέλο ===
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# === 7. Εκπαίδευση ===
model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, validation_split=0.1)

# === 8. Πρόβλεψη & αποσκαλιμάρισμα ===
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

# === 9. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ CNN-LSTM RMSE: {rmse:.2f}")

# === 10. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.legend()
plt.title("CNN-LSTM Forecast vs Actual")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.show()
