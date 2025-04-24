import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

regions = {
    "Маңғыстау": {"temp": (25, 38), "salinity": (8000, 10000), "hard": 1},
    "Қызылорда": {"temp": (20, 33), "salinity": (5000, 8000), "hard": 0.8},
    "Алматы": {"temp": (10, 25), "salinity": (1500, 4000), "hard": 0.3},
    "Атырау": {"temp": (22, 35), "salinity": (6000, 9000), "hard": 0.9},
    "Солтүстік Қазақстан": {"temp": (5, 20), "salinity": (1000, 3000), "hard": 0.2},
    "Жамбыл": {"temp": (15, 30), "salinity": (2000, 6000), "hard": 0.6},
    "Шымкент": {"temp": (18, 36), "salinity": (2500, 7000), "hard": 0.7},
    "Ақтөбе": {"temp": (10, 28), "salinity": (3000, 6000), "hard": 0.5},
    "Түркістан": {"temp": (20, 37), "salinity": (4000, 8000), "hard": 0.8},
    "Павлодар": {"temp": (5, 22), "salinity": (2000, 5000), "hard": 0.4}
}

start_time = datetime(2025, 4, 1, 0, 0, 0)
data = []
n = 2000

for i in range(n):
    region = random.choice(list(regions.keys()))
    props = regions[region]
    code = list(regions.keys()).index(region)
    time = start_time + timedelta(minutes=30 * i)
    hour = time.hour

    temp_shift = -2 if hour < 6 else (1 if hour > 12 else 0)
    temp = np.clip(np.random.uniform(*props["temp"]) + temp_shift, 0, 45)
    salinity = np.random.uniform(*props["salinity"])
    ph = np.random.uniform(6.4, 8.3)
    pressure = np.random.uniform(2.0, 6.0)
    level = np.random.uniform(20, 100)

    anomaly = 1 if np.random.rand() < 0.04 else 0
    if anomaly:
        ph = np.random.uniform(4.0, 5.8)

    output_pressure = 0.0025 * salinity + 0.05 * temp - 0.3 * ph + np.random.normal(0, 0.3)

    if salinity > 7000:
        method = "кері осмос"
        stage = "тұщыландыру"
    elif salinity > 4000:
        method = "нанофильтрация"
        stage = "алдын ала сүзу"
    else:
        method = "мембраналық сүзу"
        stage = "алдын ала сүзу"

    data.append({
        "уақыт": time.strftime("%Y-%m-%d %H:%M"),
        "өңір": region,
        "өңір_код": code,
        "температура": temp,
        "тұздылық": salinity,
        "pH": ph,
        "кіру_қысымы": pressure,
        "су_деңгейі": level,
        "шығыс_қысымы": output_pressure,
        "аномалия": anomaly,
        "әдіс": method,
        "процесс_сатысы": stage,
        "қатты_су": int(salinity > 6000),
        "жоғары_қысым_талап": int(output_pressure > 5)
    })

# Сақтау
output = pd.DataFrame(data)
output.to_csv("data/sensor_data_kz_realistic.csv", index=False)
print("✅ Kazakhstan regional dataset generated: data/sensor_data_kz_realistic.csv")