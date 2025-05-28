import pandas as pd

# === Φόρτωση αρχείου ===
df = pd.read_csv("merged_SunDance_1007.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === Αντικατάσταση ονόματος στήλης παραγωγής (π.χ. 'gen [kW]' -> 'generation') ===
if 'generation' not in df.columns:
    for col in df.columns:
        if 'gen' in col.lower():
            df.rename(columns={col: 'generation'}, inplace=True)
            break

# === Δημιουργία νέου DataFrame για feature engineering ===
df_feat = df.copy()

# --- Χρονικά χαρακτηριστικά ---
df_feat['hour'] = df_feat.index.hour
df_feat['dayofweek'] = df_feat.index.dayofweek
df_feat['month'] = df_feat.index.month
df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
df_feat['is_daylight'] = df_feat['hour'].between(6, 18).astype(int)

# --- Lag χαρακτηριστικά (1h, 2h, 3h, 24h πριν) ---
for lag in [1, 2, 3, 24]:
    df_feat[f'generation_lag_{lag}h'] = df_feat['generation'].shift(lag)

# --- Rolling averages ---
df_feat['generation_roll_mean_3h'] = df_feat['generation'].rolling(window=3).mean()
if 'tempm' in df_feat.columns:
    df_feat['tempm_roll_mean_3h'] = df_feat['tempm'].rolling(window=3).mean()

# --- Αφαίρεση NaNs που δημιουργήθηκαν από shift/rolling ---
df_feat_clean = df_feat.dropna()

# === Αποθήκευση εμπλουτισμένου αρχείου ===
df_feat_clean.to_csv("merged_SunDance_1007_features.csv")

print("✅ Ολοκληρώθηκε το Feature Engineering!")
print(f"📁 Αποθηκεύτηκε ως: merged_SunDance_1007_features.csv")
