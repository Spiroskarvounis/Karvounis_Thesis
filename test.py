import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.sort_index()
df['target'] = df['generation'].shift(-1)

# === 2. Drop NaNs ===
df.dropna(inplace=True)

# === 3. Χωρισμός features ===
exclude = ['generation', 'datetzname', 'conds', 'icon', 'metar']
features = df.drop(columns=exclude + ['target'], errors='ignore')

# === 4. Ορισμός υπο-ομάδων features ===
time_features = [col for col in features.columns if 'hour' in col or 'day' in col or 'month' in col or 'sin' in col or 'cos' in col]
lag_features = [col for col in features.columns if 'lag' in col or 'roll' in col or 'mean' in col or 'std' in col]

X_time = features[time_features]
X_lag = features[lag_features]
y = df['target']

# === 5. Train/Test Split ===
X_time_train, X_time_test, X_lag_train, X_lag_test, y_train, y_test = train_test_split(
    X_time, X_lag, y, test_size=0.2, shuffle=False)

# === 6. Κανονικοποίηση ===
scaler_time = StandardScaler()
scaler_lag = StandardScaler()

X_time_train_scaled = scaler_time.fit_transform(X_time_train)
X_time_test_scaled = scaler_time.transform(X_time_test)

X_lag_train_scaled = scaler_lag.fit_transform(X_lag_train)
X_lag_test_scaled = scaler_lag.transform(X_lag_test)

# === 7. Εκπαίδευση μοντέλων ===
ridge = Ridge(alpha=1.0)
ridge.fit(X_time_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_time_test_scaled)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_lag_train_scaled, y_train)
y_pred_rf = rf.predict(X_lag_test_scaled)

# === 8. Συνδυασμός προβλέψεων ===
alpha = 0.4  # βάρος στο Ridge
y_pred_combo = alpha * y_pred_ridge + (1 - alpha) * y_pred_rf

# === 9. Αξιολόγηση ===
def evaluate_model(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {label} Evaluation:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.4f}")

evaluate_model(y_train, rf.predict(X_lag_train_scaled), "RF Train Set")
evaluate_model(y_test, y_pred_rf, "RF Test Set")
evaluate_model(y_test, y_pred_combo, "Hybrid Ridge + RF Test Set")

# === 10. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(y_pred_combo[:200], label="Hybrid Predicted")
plt.title("Hybrid Ridge + Random Forest Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
