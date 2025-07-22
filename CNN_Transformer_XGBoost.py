import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# === 1. Φόρτωση και επεξεργασία δεδομένων ===
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

# === 2. Sliding windows ===
SEQ_LEN = 24
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y.values, SEQ_LEN)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# === 3. CNN + Transformer Feature Extractor ===
def cnn_transformer_feature_extractor(seq_len, input_dim):
    inp = Input(shape=(seq_len, input_dim))
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inp)
    x = layers.LayerNormalization()(x)

    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    return models.Model(inp, x)

model_feat = cnn_transformer_feature_extractor(SEQ_LEN, X_train.shape[2])

X_train_feat = model_feat.predict(X_train, verbose=0)
X_test_feat = model_feat.predict(X_test, verbose=0)

# === 4. XGBoost ===
xgb_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_feat, y_train)
y_pred = xgb_model.predict(X_test_feat)

# === 5. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ CNN → Transformer → XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 6. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.title("CNN → Transformer → XGBoost Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
