import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from xgboost import XGBRegressor

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Επιλογή χαρακτηριστικών ===
exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X = df.drop(columns=exclude_cols, errors='ignore')
X = X.loc[:, X.dtypes != 'object']
y = df['target'].values

# === 3. Δημιουργία sliding windows ===
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 24
X_np = X.values
X_seq_all, y_seq_all = create_sequences(X_np, y, SEQ_LEN)

# === 4. Split πριν το scaling ===
split = int(0.8 * len(X_seq_all))
X_train_raw, X_test_raw = X_seq_all[:split], X_seq_all[split:]
y_train, y_test = y_seq_all[:split], y_seq_all[split:]

# === 5. Scaling μετά το split ===
n_features = X_train_raw.shape[2]
scaler = StandardScaler()

X_train_flat = X_train_raw.reshape(-1, n_features)
X_test_flat = X_test_raw.reshape(-1, n_features)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_raw.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_raw.shape)

# === 6. Transformer feature extractor ===
def transformer_feature_extractor(seq_len, input_dim):
    inputs = Input(shape=(seq_len, input_dim))
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.1)(x, x)
    x = layers.Dropout(0.1)(x)
    x = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(filters=128, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=input_dim, kernel_size=1)(x)
    x = x + inputs

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    model = models.Model(inputs, x)
    return model

# === 7. Εξαγωγή χαρακτηριστικών και πρόβλεψη με XGBoost ===
feature_model = transformer_feature_extractor(SEQ_LEN, n_features)
X_train_feats = feature_model.predict(X_train_scaled)
X_test_feats = feature_model.predict(X_test_scaled)

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_feats, y_train)
y_pred = xgb_model.predict(X_test_feats)

# === 8. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Transformer → XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 9. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Transformer → XGBoost Hybrid Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
