import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- ΡΥΘΜΙΣΗ ΑΡΧΕΙΩΝ ----------
merged_file = "merged_SunDance_1007.csv"
output_dir = "merged_plots"
os.makedirs(output_dir, exist_ok=True)

# ---------- ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ----------
df = pd.read_csv(merged_file, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# ---------- FEATURE EXTRACTION ----------
df['hour'] = df.index.hour
df['month'] = df.index.month

df.rename(columns={'gen [kW]': 'generation'}, inplace=True)

# ---------- 1. Χρονική εξέλιξη παραγωγής ----------
plt.figure(figsize=(14, 5))
df['generation'].plot()
plt.title("Ηλιακή Παραγωγή στο Χρόνο")
plt.xlabel("Ημερομηνία")
plt.ylabel("Παραγωγή (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/generation_over_time.png")
plt.close()

# ---------- 2. Temp vs Generation ----------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tempm', y='generation', data=df, alpha=0.5)
plt.title("Θερμοκρασία vs Ηλιακή Παραγωγή")
plt.xlabel("Θερμοκρασία (°C)")
plt.ylabel("Παραγωγή (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/temp_vs_generation.png")
plt.close()

# ---------- 3. Humidity vs Generation ----------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='hum', y='generation', data=df, alpha=0.5)
plt.title("Υγρασία vs Ηλιακή Παραγωγή")
plt.xlabel("Υγρασία (%)")
plt.ylabel("Παραγωγή (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/humidity_vs_generation.png")
plt.close()

# ---------- 4. Boxplot παραγωγής ανά Ώρα ----------
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='generation', data=df)
plt.title("Ηλιακή Παραγωγή ανά Ώρα Ημέρας")
plt.xlabel("Ώρα")
plt.ylabel("Παραγωγή (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/generation_by_hour.png")
plt.close()

# ---------- 5. Histogram παραγωγής ----------
plt.figure(figsize=(10, 5))
df['generation'].hist(bins=50)
plt.title("Κατανομή Ηλιακής Παραγωγής")
plt.xlabel("Παραγωγή (kWh)")
plt.ylabel("Συχνότητα")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/generation_distribution.png")
plt.close()

# ---------- 6. Correlation Heatmap ----------
numeric_cols = ['generation', 'tempm', 'dewptm', 'hum', 'wspdm', 'pressurem']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Συσχέτιση Παραγωγής & Μετεωρολογικών Παραμέτρων")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

print("✅ Οπτικοποιήσεις ολοκληρώθηκαν! Αρχεία αποθηκεύτηκαν στο:", output_dir)
