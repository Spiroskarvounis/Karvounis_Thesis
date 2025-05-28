import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === 1. Φόρτωση enriched αρχείου ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === 2. Δημιουργία στόχου (επόμενη ώρα παραγωγής) ===
df['target'] = df['generation'].shift(-1)
df = df.dropna()

# === 3. Επιλογή αριθμητικών χαρακτηριστικών ===
X = df.select_dtypes(include=[np.number]).drop(columns=['target'])
X.columns = X.columns.str.replace(r"[\[\]<>]", "", regex=True)  # καθαρισμός ονομάτων
y = df['target']

# === 4. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 5. Εκπαίδευση Random Forest μοντέλου ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Πρόβλεψη ===
y_pred = model.predict(X_test)

# === 7. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Random Forest RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 8. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Random Forest Forecast vs Actual")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
