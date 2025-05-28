import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === 1. Φόρτωση Δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === 2. Δημιουργία στόχου (επόμενη ώρα παραγωγής) ===
df['target'] = df['generation'].shift(-1)
df = df.dropna()

# === 3. Επιλογή αριθμητικών χαρακτηριστικών ===
X = df.select_dtypes(include=[np.number]).drop(columns=['target'])
y = df['target']

# === 4. Κανονικοποίηση ===
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# === 5. Δημιουργία ακολουθιών για LSTM ===
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24  # 24 ώρες
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# === 6. Split σε train/test ===
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === 7. Δημιουργία LSTM μοντέλου ===
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# === 8. Εκπαίδευση ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# === 9. Πρόβλεψη ===
y_pred = model.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# === 10. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"✅ RMSE: {rmse:.2f}")

# === 11. Plot predicted vs actual ===
plt.figure(figsize=(12, 4))
plt.plot(y_test_inv[:200], label='Actual')
plt.plot(y_pred_inv[:200], label='Predicted')
plt.legend()
plt.title("LSTM Forecast vs Actual")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.show()
