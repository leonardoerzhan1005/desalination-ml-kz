
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# Extended regions with method probabilities
regions = {
    "Маңғыстау": {"temp": (25, 42), "salinity": (9000, 12000), "hard": 1, "methods": {"кері осмос": 0.7, "нанофильтрация": 0.25, "мембраналық сүзу": 0.05}},
    "Қызылорда": {"temp": (20, 37), "salinity": (6000, 9000), "hard": 0.8, "methods": {"кері осмос": 0.5, "нанофильтрация": 0.4, "мембраналық сүзу": 0.1}},
    "Алматы": {"temp": (10, 27), "salinity": (1200, 4000), "hard": 0.3, "methods": {"кері осмос": 0.1, "нанофильтрация": 0.3, "мембраналық сүзу": 0.6}},
    "Атырау": {"temp": (22, 39), "salinity": (8000, 11000), "hard": 0.9, "methods": {"кері осмос": 0.65, "нанофильтрация": 0.3, "мембраналық сүзу": 0.05}},
    "Солтүстік Қазақстан": {"temp": (5, 23), "salinity": (1000, 3000), "hard": 0.2, "methods": {"кері осмос": 0.05, "нанофильтрация": 0.15, "мембраналық сүзу": 0.8}},
    "Жамбыл": {"temp": (15, 32), "salinity": (2000, 6000), "hard": 0.6, "methods": {"кері осмос": 0.3, "нанофильтрация": 0.4, "мембраналық сүзу": 0.3}},
    "Шымкент": {"temp": (18, 38), "salinity": (3000, 7500), "hard": 0.7, "methods": {"кері осмос": 0.4, "нанофильтрация": 0.4, "мембраналық сүзу": 0.2}},
    "Ақтөбе": {"temp": (10, 30), "salinity": (2500, 6000), "hard": 0.5, "methods": {"кері осмос": 0.2, "нанофильтрация": 0.5, "мембраналық сүзу": 0.3}},
    "Түркістан": {"temp": (20, 37), "salinity": (4500, 8500), "hard": 0.8, "methods": {"кері осмос": 0.5, "нанофильтрация": 0.4, "мембраналық сүзу": 0.1}},
    "Павлодар": {"temp": (5, 25), "salinity": (2000, 4800), "hard": 0.4, "methods": {"кері осмос": 0.15, "нанофильтрация": 0.35, "мембраналық сүзу": 0.5}}
}

start_time = datetime(2025, 1, 1, 0, 0, 0)  # Start earlier for seasonal variation
data = []
n = 10000  # Increased dataset size

for i in range(n):
    region = random.choice(list(regions.keys()))
    props = regions[region]
    code = list(regions.keys()).index(region)
    time = start_time + timedelta(minutes=30 * i)
    hour = time.hour
    month = time.month

    # Seasonal temperature adjustment
    seasonal_temp_shift = 5 * np.sin(2 * np.pi * (month - 3) / 12)  # Peak in summer (June)
    temp = np.clip(np.random.uniform(*props["temp"]) + seasonal_temp_shift + (-2 if hour < 6 else 3 if 12 < hour < 17 else 0), 0, 50)

    # Salinity with regional and seasonal variation
    base_salinity = np.random.normal(np.mean(props["salinity"]), 500)
    seasonal_salinity_shift = 200 * np.sin(2 * np.pi * (month - 6) / 12)  # Higher salinity in late summer
    salinity = np.clip(base_salinity + seasonal_salinity_shift, 500, 15000)

    # pH with correlation to salinity
    ph = np.clip(np.random.normal(7.4 - 0.00005 * salinity, 0.3), 6.0, 8.5)
    pressure = np.clip(np.random.normal(4.5, 0.8), 2.0, 7.0)
    level = np.clip(np.random.normal(60, 20), 10, 120)

    # Anomaly (5% chance, varied types)
    anomaly_type = random.choices(["none", "high_salinity", "low_ph", "pressure_spike"], weights=[0.95, 0.02, 0.02, 0.01])[0]
    anomaly = 1 if anomaly_type != "none" else 0
    if anomaly_type == "high_salinity":
        salinity *= np.random.uniform(1.3, 1.8)
    elif anomaly_type == "low_ph":
        ph = np.random.uniform(4.5, 5.5)
    elif anomaly_type == "pressure_spike":
        pressure *= np.random.uniform(1.5, 2.0)

    # Output pressure with improved formula
    output_pressure = np.clip(0.002 * salinity + 0.04 * temp - 0.25 * ph + 0.1 * pressure + np.random.normal(0, 0.3), 0, 10)

    # Method assignment with regional probabilities
    method = random.choices(
        ["кері осмос", "нанофильтрация", "мембраналық сүзу"],
        weights=[props["methods"]["кері осмос"], props["methods"]["нанофильтрация"], props["methods"]["мембраналық сүзу"]]
    )[0]

    # Additional feature: filter efficiency
    filter_efficiency = np.clip(np.random.normal(0.85, 0.05), 0.7, 0.95) if method != "мембраналық сүзу" else np.clip(np.random.normal(0.9, 0.03), 0.8, 0.98)

    data.append({
        "уақыт": time.strftime("%Y-%m-%d %H:%M"),
        "өңір": region,
        "өңір_код": code,
        "температура": round(temp, 2),
        "тұздылық": round(salinity, 2),
        "pH": round(ph, 2),
        "кіру_қысымы": round(pressure, 2),
        "су_деңгейі": round(level, 2),
        "шығыс_қысымы": round(output_pressure, 2),
        "аномалия": anomaly,
        "аномалия_түрі": anomaly_type,
        "әдіс": method,
        "фильтр_тиімділігі": round(filter_efficiency, 2)
    })

output = pd.DataFrame(data)
output.to_csv("data/sensor_data_kz_realistic.csv", index=False)
print(f"✅ Kazakhstan realistic dataset updated: data/sensor_data_kz_realistic.csv with {len(output)} records")
