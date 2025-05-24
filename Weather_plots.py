import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- ΑΡΧΙΚΕΣ ΡΥΘΜΙΣΕΙΣ ----------
# Path προς αρχείο weather ενός site
weather_path = "C:\\Users\\karvs\\Downloads\\sundance-data-release\\SunDance_data_release\\weather\\SunDance_1007.csv"

# Δημιουργία φακέλου για αποθήκευση διαγραμμάτων
output_dir = "C:\\Users\\karvs\\Downloads\\sundance-data-release\\SunDance_data_release\\weather\\weather_plots"
os.makedirs(output_dir, exist_ok=True)

# ---------- ΣΤΗΛΕΣ ΟΠΩΣ ΟΡΙΖΟΝΤΑΙ ΣΤΟ README ----------
columns = [
    'dateyear', 'datemon', 'datemday', 'datehour', 'datemin', 'datetzname',
    'tempm', 'tempi', 'dewptm', 'dewpti',
    'hum', 'wspdm', 'wspdi', 'wgustm', 'wgusti',
    'wdird', 'wdire', 'vism', 'visi',
    'pressurem', 'pressurei',
    'windchillm', 'windchilli', 'heatindexm', 'heatindexi',
    'precipm', 'precipi',
    'conds', 'icon',
    'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado',
    'metar'
]

# ---------- ΔΙΑΒΑΣΜΑ ΔΕΔΟΜΕΝΩΝ ----------
df = pd.read_csv(weather_path, header=None, names=columns)

# Δημιουργία timestamp
df['timestamp'] = pd.to_datetime(
    df[['dateyear', 'datemon', 'datemday', 'datehour', 'datemin']].rename(
        columns={'dateyear': 'year', 'datemon': 'month', 'datemday': 'day', 'datehour': 'hour', 'datemin': 'minute'}
    ),
    errors='coerce'
)

df.set_index('timestamp', inplace=True)

# ---------- FEATURE EXTRACTION ----------
df['hour'] = df.index.hour
df['month'] = df.index.month

# ---------- 1. Θερμοκρασία στο χρόνο ----------
plt.figure(figsize=(14, 5))
df['tempm'].plot()
plt.title("Θερμοκρασία στο Χρόνο")
plt.xlabel("Ημερομηνία")
plt.ylabel("Θερμοκρασία (°C)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/temperature_over_time.png")
plt.close()

# ---------- 2. Boxplot Θερμοκρασίας ανά Ώρα ----------
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='tempm', data=df)
plt.title("Κατανομή Θερμοκρασίας ανά Ώρα")
plt.xlabel("Ώρα")
plt.ylabel("Θερμοκρασία (°C)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/temperature_by_hour.png")
plt.close()

# ---------- 3. Boxplot Υγρασίας ανά Μήνα ----------
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='hum', data=df)
plt.title("Κατανομή Σχετικής Υγρασίας ανά Μήνα")
plt.xlabel("Μήνας")
plt.ylabel("Υγρασία (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/humidity_by_month.png")
plt.close()

# ---------- 4. Scatter: Θερμοκρασία vs Υγρασία ----------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tempm', y='hum', data=df, alpha=0.5)
plt.title("Θερμοκρασία vs Σχετική Υγρασία")
plt.xlabel("Θερμοκρασία (°C)")
plt.ylabel("Υγρασία (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/temp_vs_humidity.png")
plt.close()

# ---------- 5. Συχνότητα Καιρικών Συνθηκών ----------
plt.figure(figsize=(10, 6))
sns.countplot(y='conds', data=df, order=df['conds'].value_counts().index[:10])
plt.title("10 Πιο Συχνές Καιρικές Συνθήκες")
plt.xlabel("Συχνότητα")
plt.ylabel("Συνθήκη")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/weather_condition_counts.png")
plt.close()

print("✅ Ολοκληρώθηκε. Τα γραφήματα αποθηκεύτηκαν στον φάκελο:", output_dir)
