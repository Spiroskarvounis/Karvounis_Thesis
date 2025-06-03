import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# === 1. Φόρτωση enriched dataset ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === 2. Δημιουργία στόχου (παραγωγή επόμενης ώρας) ===
df['target'] = df['generation'].shift(-1)
df = df.dropna()

# === 3. Επιλογή χαρακτηριστικών από NWP + calendar ===
nwp_features = ['tempm', 'dewptm', 'hum', 'wspdm', 'pressurem']
calendar_features = ['hour', 'dayofweek', 'month', 'is_daylight']
X = df[nwp_features + calendar_features].copy()
X.columns = X.columns.str.replace(r"[\[\]<>]", "", regex=True)

y = df['target']

# === 4. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 5. XGBoost training ===
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === 6. Πρόβλεψη & αξιολόγηση ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ NWP + XGBoost RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 7. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("NWP + XGBoost Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
