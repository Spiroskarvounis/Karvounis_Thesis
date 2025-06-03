import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("merged_SunDance_1007_features.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === 2. Επιλογή της παραγωγής (μόνο univariate σειρά) ===
series = df['generation'].dropna()

# === 3. Χωρισμός σε train/test ===
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# === 4. Εκπαίδευση ARIMA ===
# Οι παράμετροι p,d,q μπορούν να βελτιστοποιηθούν (εδώ: p=2, d=1, q=2)
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()

# === 5. Πρόβλεψη για όλο το test set ===
forecast = model_fit.forecast(steps=len(test))
forecast = forecast.values

# === 6. Αξιολόγηση ===
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
r2 = r2_score(test, forecast)

print(f"✅ ARIMA RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === 7. Οπτικοποίηση ===
plt.figure(figsize=(12, 4))
plt.plot(test.values[:200], label='Actual')
plt.plot(forecast[:200], label='Predicted')
plt.title("ARIMA Forecast")
plt.xlabel("Samples")
plt.ylabel("Solar Generation [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
