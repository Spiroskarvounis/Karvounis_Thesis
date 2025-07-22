import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import layers, models
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pywt

# === 1. Φόρτωση και επεξεργασία δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X_raw = df.drop(columns=exclude_cols, errors='ignore')
X_raw = X_raw.loc[:, X_raw.dtypes != 'object'].values
y_raw = df['target'].values

# === 2. Split πριν το scaling ===
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# === 3. Scaling μετά το split ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# === 4. Wavelet Transform ===
def apply_wavelet_transform(X, wavelet='db1'):
    transformed = []
    for i in range(X.shape[1]):
        cA, cD = pywt.dwt(X[:, i], wavelet)
        min_len = min(len(cA), len(cD))
        transformed.append(cA[:min_len])
        transformed.append(cD[:min_len])
    return np.stack(transformed, axis=1).T

X_train_wavelet = apply_wavelet_transform(X_train_scaled)
X_test_wavelet = apply_wavelet_transform(X_test_scaled)

X_train_wavelet = X_train_wavelet.reshape(-1, X_train_wavelet.shape[1], 1)
X_test_wavelet = X_test_wavelet.reshape(-1, X_test_wavelet.shape[1], 1)

y_train = y_train[:X_train_wavelet.shape[0]]
y_test = y_test[:X_test_wavelet.shape[0]]

# === 5. LSTM Feature Extractor ===
def build_lstm_feature_extractor(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(32, activation='relu')
    ])
    return model

lstm_model = build_lstm_feature_extractor((X_train_wavelet.shape[1], 1))
features_train = lstm_model.predict(X_train_wavelet)
features_test = lstm_model.predict(X_test_wavelet)

# === 6. XGBoost ===
xgb = XGBRegressor()
xgb.fit(features_train, y_train)
y_pred = xgb.predict(features_test)

# === 7. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Wavelet → LSTM → XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 8. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Wavelet → LSTM → XGBoost Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
