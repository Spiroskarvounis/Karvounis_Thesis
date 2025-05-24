# 📦 Βασικές βιβλιοθήκες
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("C:\\Users\\karvs\\Downloads\\sundance-data-release\\SunDance_data_release\\energy\\SunDance_1007.csv")
print(df.columns)

# Για ημερομηνίες
from datetime import datetime
#
# 👉 Βήμα 1: Φόρτωση δεδομένων
# Προσαρμόστε το path ανάλογα με το αρχείο σας
df = pd.read_csv("C:\\Users\karvs\Downloads\sundance-data-release\SunDance_data_release\energy\SunDance_1007.csv", parse_dates=['Date & Time'])
df.rename(columns={'Date & Time': 'timestamp'}, inplace=True)


# #
# # 👉 Βήμα 2: Γρήγορη ματιά στα δεδομένα
# print("📌 Πρώτες 5 γραμμές:")
# print(df.head())
#
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
#
# # 👉 Βήμα 5: Ανάλυση χρονικής κατανομής
# df.set_index('timestamp', inplace=True)
# df = df.sort_index()
# #
# # Plot ηλιακής παραγωγής (αν υπάρχει η στήλη)
# if 'Solar+Wind [kW]' in df.columns:
#     plt.figure(figsize=(14, 5))
#     df['Solar+Wind [kW]'].plot()
#     plt.title("Ηλιακή Παραγωγή στο Χρόνο")
#     plt.xlabel("Ημερομηνία")
#     plt.ylabel("kWh")
#     plt.grid(True)
#     plt.show()

df['gen [kW]'].plot(figsize=(14,5), title='Ηλιακή Παραγωγή στο Χρόνο')
