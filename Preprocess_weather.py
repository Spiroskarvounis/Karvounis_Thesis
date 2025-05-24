import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Ονόματα στηλών όπως δίνονται στο README
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

# 📥 Φόρτωση αρχείου χωρίς header και με σωστές στήλες
df = pd.read_csv(
    "C:\\Users\\karvs\\Downloads\\sundance-data-release\\SunDance_data_release\\weather\\SunDance_1007.csv",
    header=None,
    names=columns
)

# 🕒 Δημιουργία στήλης timestamp από τις στήλες ημερομηνίας-ώρας
df['timestamp'] = pd.to_datetime(
    df[['dateyear', 'datemon', 'datemday', 'datehour', 'datemin']].rename(
        columns={
            'dateyear': 'year',
            'datemon': 'month',
            'datemday': 'day',
            'datehour': 'hour',
            'datemin': 'minute'
        }
    ),
    errors='coerce'  # αγνοεί invalid ημερομηνίες αν υπάρχουν
)

# 📌 Προβολή βασικών πληροφοριών
#print(df[['timestamp', 'tempm', 'hum', 'conds']].head())

# 👉 Θέσε timestamp ως index για μελλοντική ανάλυση
df.set_index('timestamp', inplace=True)

# 👉 Βήμα 2: Γρήγορη ματιά στα δεδομένα
print("📌 Πρώτες 5 γραμμές:")
print(df.head())

# print("\n📌 Τύποι δεδομένων:")
# print(df.dtypes)
#
# print("\n📌 Διαστάσεις dataset:", df.shape)
#
# # 👉 Βήμα 3: Έλεγχος για null/NaN τιμές
# print("\n📌 Απουσία τιμών:")
# print(df.isnull().sum())
#
# # 👉 Βήμα 4: Στατιστικά περιγραφικά
# print("\n📌 Στατιστικά χαρακτηριστικά:")
# print(df.describe())

# df['tempm'].plot(figsize=(14,5), title='Θερμοκρασία στο Χρόνο')
# df['hum'].plot(figsize=(14,5), title='Σχετική Υγρασία στο Χρόνο')
# df['hour'] = df.index.hour
# df['month'] = df.index.month
#
# sns.boxplot(x='hour', y='tempm', data=df)
# plt.title('Κατανομή Θερμοκρασίας ανά Ώρα Ημέρας')
# plt.show()

df['hour'] = df.index.hour

plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='tempm', data=df)
plt.title("Κατανομή Θερμοκρασίας ανά Ώρα")
plt.xlabel("Ώρα")
plt.ylabel("Θερμοκρασία (°C)")
plt.grid(True)
plt.show()  # ← αυτό είναι απολύτως απαραίτητο σε script
