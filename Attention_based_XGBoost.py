import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

# === 2. Επιλογή και κανονικοποίηση χαρακτηριστικών ===
exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X = df.drop(columns=exclude_cols, errors='ignore')
X = X.loc[:, X.dtypes != 'object']
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# === 4. Attention block ===
input_layer = layers.Input(shape=(X_train.shape[1],))
dense = layers.Dense(64, activation='relu')(input_layer)
attention_probs = layers.Dense(X_train.shape[1], activation='softmax')(dense)
attended_features = layers.Multiply()([input_layer, attention_probs])
encoder = models.Model(inputs=input_layer, outputs=attended_features)

# === 5. Προεξαγωγή attended features ===
X_train_att = encoder.predict(X_train)
X_test_att = encoder.predict(X_test)

# === 6. XGBoost ===
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train_att, y_train)
y_pred = model.predict(X_test_att)

# === 7. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Attention + XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 8. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Attention-enhanced XGBoost Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
