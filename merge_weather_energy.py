import pandas as pd

# ----- ΟΡΙΣΜΟΣ ΑΡΧΕΙΩΝ -----
weather_path = "C:\\Users\\karvs\\Downloads\\sundance-data-release\\SunDance_data_release\\weather\\SunDance_1007.csv"
energy_path = "C:\\Users\\karvs\\Downloads\\sundance-data-release\\SunDance_data_release\\energy\\SunDance_1007.csv"
output_path = "merged_SunDance_1007.csv"

# ----- ΣΤΗΛΕΣ WEATHER ΣΥΜΦΩΝΑ ΜΕ README -----
weather_columns = [
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

# ----- ΦΟΡΤΩΣΗ WEATHER -----
df_weather = pd.read_csv(weather_path, header=None, names=weather_columns)

# Δημιουργία timestamp
df_weather['timestamp'] = pd.to_datetime(
    df_weather[['dateyear', 'datemon', 'datemday', 'datehour', 'datemin']].rename(
        columns={'dateyear': 'year', 'datemon': 'month', 'datemday': 'day',
                 'datehour': 'hour', 'datemin': 'minute'}
    ),
    errors='coerce'
)

df_weather.set_index('timestamp', inplace=True)

# 🟢 Στρογγυλοποίηση στην πλησιέστερη ώρα
df_weather.index = df_weather.index.round('H')

# ----- ΦΟΡΤΩΣΗ ENERGY -----
df_energy = pd.read_csv(energy_path, parse_dates=['Date & Time'])
df_energy.rename(columns={'Date & Time': 'timestamp'}, inplace=True)
df_energy.set_index('timestamp', inplace=True)

# ----- ΣΥΓΧΩΝΕΥΣΗ -----
merged_df = df_energy.join(df_weather, how='inner')  # τώρα θα ταιριάξουν οι ώρες

# ----- ΑΠΟΘΗΚΕΥΣΗ -----
merged_df.to_csv(output_path)
print("✅ Συγχώνευση επιτυχής! Αποθηκεύτηκε ως:", output_path)

# Δείγμα
print("\n📌 Δείγμα από το νέο merged αρχείο:")
print(merged_df[['gen [kW]', 'tempm', 'hum', 'conds']].head())
