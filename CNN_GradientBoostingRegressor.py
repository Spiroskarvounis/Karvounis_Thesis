import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === 1. Load and prepare data ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.sort_index()
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Feature selection ===
exclude = ['generation', 'datetzname', 'conds', 'icon', 'metar']
features = df.drop(columns=exclude + ['target'], errors='ignore')
X = features.select_dtypes(include=[np.number])
y = df['target']

# === 3. Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Create sliding windows ===
SEQ_LEN = 24
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y.values, SEQ_LEN)

# === 5. Split train/test ===
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === 6. DL feature extractor on time windows ===
def cnn_feature_extractor(seq_len, input_dim):
    inp = Input(shape=(seq_len, input_dim))
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inp)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    return models.Model(inp, x)

extractor = cnn_feature_extractor(SEQ_LEN, X_train.shape[2])
X_train_feat = extractor.predict(X_train, verbose=0)
X_test_feat = extractor.predict(X_test, verbose=0)

# === 7. Train ML model on DL features ===
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X_train_feat, y_train)
y_pred = gbr.predict(X_test_feat)

# === 8. Evaluation ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ DL Feature Extractor → Gradient Boosting RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 9. Plot ===
plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.title("DL Feature Extractor → Gradient Boosting Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
