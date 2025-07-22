import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# === 1. Φόρτωση enriched dataset ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Επιλογή αριθμητικών χαρακτηριστικών ===
X = df.select_dtypes(include=[np.number]).drop(columns=['target']).values
y = df['target'].values.reshape(-1, 1)

# === 3. Δημιουργία ακολουθιών ===
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24
X_seq_all, y_seq_all = create_sequences(X, y, TIME_STEPS)

# === 4. Split ΠΡΙΝ το scaling ===
split = int(len(X_seq_all) * 0.8)
X_train_raw, X_test_raw = X_seq_all[:split], X_seq_all[split:]
y_train_raw, y_test_raw = y_seq_all[:split], y_seq_all[split:]

# === 5. Scaling ΜΕΤΑ το split ===
n_features = X_train_raw.shape[2]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_flat = X_train_raw.reshape(-1, n_features)
X_test_flat = X_test_raw.reshape(-1, n_features)

X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train_raw.shape)
X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test_raw.shape)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# === 6. LSTM για feature extraction ===
inputs = Input(shape=(TIME_STEPS, n_features))
x = LSTM(64, activation='relu')(inputs)
features = Dense(16, activation='relu')(x)
output = Dense(1)(features)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')

# === 7. Εκπαίδευση ===
model.fit(X_train_scaled, y_train_scaled, epochs=10, batch_size=32, validation_split=0.1)

# === 8. Εξαγωγή deep features ===
feature_extractor = Model(inputs=inputs, outputs=features)
X_train_feat = feature_extractor.predict(X_train_scaled)
X_test_feat = feature_extractor.predict(X_test_scaled)

# === 9. Εκπαίδευση Random Forest ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_feat, y_train_scaled.ravel())

# === 10. Πρόβλεψη και αποσκαλιμάρισμα ===
y_pred_scaled = rf_model.predict(X_test_feat).reshape(-1, 1)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Hybrid LSTM → RF RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 11. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Hybrid DeepTree (LSTM → Random Forest)")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
