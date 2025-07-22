import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === 1. Φόρτωση και Προεπεξεργασία Δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df['target'] = df['generation'].shift(-1)
df.dropna(inplace=True)

exclude_cols = ['generation', 'target', 'datetzname', 'conds', 'icon', 'metar']
X = df.drop(columns=exclude_cols, errors='ignore')
X = X.loc[:, X.dtypes != 'object']
y = df['target']

# === Split BEFORE scaling ===
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === Fit scaler ONLY on training data ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# === 2. Ορισμός Base Learners ===
xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=100, max_depth=8)
lgb = LGBMRegressor(n_estimators=100)

# === 3. Stacking Ensemble ===
estimators = [
    ('xgb', xgb),
    ('rf', rf),
    ('lgb', lgb)
]

stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)

# === 4. Εκπαίδευση και Αξιολόγηση ===
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === 5. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.title("Stacked (XGB + RF + LGBM) Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.tight_layout()
plt.show()

# === 6. Εμφάνιση Μετρικών ===
print(f"✅ Stacked RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")
print(stacked_model.final_estimator_.coef_)
