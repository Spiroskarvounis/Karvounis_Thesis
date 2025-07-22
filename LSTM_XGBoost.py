import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
import matplotlib.pyplot as plt

# === 1. Load dataset ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Select features and labels ===
X_raw = df.select_dtypes(include=[np.number]).drop(columns=['target'])
y_raw = df['target']

# === 3. Train-test split BEFORE scaling ===
TIME_STEPS = 24
split = int(len(df) * 0.8)

X_train_raw = X_raw.iloc[:split + TIME_STEPS]
X_test_raw = X_raw.iloc[split + TIME_STEPS:]
y_train_raw = y_raw.iloc[:split + TIME_STEPS]
y_test_raw = y_raw.iloc[split + TIME_STEPS:]

# === 4. Scale separately ===
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test_raw.values.reshape(-1, 1))

# === 5. Sequence creation ===
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i-time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)

# === 6. LSTM Model ===
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# === 7. Extract LSTM features and train XGBoost ===
lstm_train_features = model.predict(X_train_seq)
lstm_test_features = model.predict(X_test_seq)

xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
xgb_model.fit(lstm_train_features, y_train_seq)
y_pred_scaled = xgb_model.predict(lstm_test_features)

# === 8. Invert scaling for evaluation ===
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_true = scaler_y.inverse_transform(y_test_seq)

# === 9. Metrics ===
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"✅ LSTM → XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 10. Plot ===
plt.figure(figsize=(12, 4))
plt.plot(y_true[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("LSTM → XGBoost Forecasting")
plt.xlabel("Time steps")
plt.ylabel("Solar Generation (kW)")
plt.legend()
plt.tight_layout()
plt.show()
