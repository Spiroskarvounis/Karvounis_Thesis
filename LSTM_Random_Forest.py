import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# === 1. Φόρτωση enriched dataset ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Επιλογή αριθμητικών χαρακτηριστικών ===
X_raw = df.select_dtypes(include=[np.number]).drop(columns=['target'])
y = df['target']

# Κανονικοποίηση
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# === 3. Δημιουργία ακολουθιών ===
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# === 4. Train-test split ===
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === 5. LSTM για feature extraction ===
inputs = Input(shape=(TIME_STEPS, X_train.shape[2]))
x = LSTM(64, activation='relu')(inputs)
features = Dense(16, activation='relu')(x)
output = Dense(1)(features)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')

# === 6. Εκπαίδευση ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# === 7. Εξαγωγή deep features ===
feature_extractor = Model(inputs=inputs, outputs=features)
X_train_feat = feature_extractor.predict(X_train)
X_test_feat = feature_extractor.predict(X_test)

# === 8. Εκπαίδευση Random Forest ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_feat, y_train)

# === 9. Πρόβλεψη και αξιολόγηση ===
y_pred = rf_model.predict(X_test_feat)
y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"✅ Hybrid LSTM → RF RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 10. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test_inv[:200], label='Actual')
plt.plot(y_pred_inv[:200], label='Predicted')
plt.title("Hybrid DeepTree (LSTM → Random Forest)")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
