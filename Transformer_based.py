import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
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

# === 2. Κανονικοποίηση και δημιουργία χρονικών παραθύρων ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

# === 3. Ορισμός Transformer Encoder για εξαγωγή χαρακτηριστικών ===
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

# === 4. Εξαγωγή χαρακτηριστικών και πρόβλεψη με XGBoost ===
feature_model = transformer_feature_extractor(SEQ_LEN, X_train.shape[2])
X_train_feats = feature_model.predict(X_train)
X_test_feats = feature_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_feats, y_train)
y_pred = xgb_model.predict(X_test_feats)

# === 5. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Transformer → XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 6. Οπτικοποίηση ===
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
