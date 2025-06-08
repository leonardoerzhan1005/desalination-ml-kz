import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# Define regions with properties
regions = {
    "Маңғыстау": {"temp": (25, 42), "salinity": (9000, 12000), "hard": 1, "methods": {"кері осмос": 0.7368, "нанофильтрация": 0.2632}, "capacity": 5000},
    "Қызылорда": {"temp": (20, 37), "salinity": (6000, 9000), "hard": 0.8, "methods": {"кері осмос": 0.5556, "нанофильтрация": 0.4444}, "capacity": 3000},
    "Алматы": {"temp": (10, 27), "salinity": (1200, 4000), "hard": 0.3, "methods": {"кері осмос": 0.1429, "нанофильтрация": 0.8571}, "capacity": 1000},
    "Атырау": {"temp": (22, 39), "salinity": (8000, 11000), "hard": 0.9, "methods": {"кері осмос": 0.6842, "нанофильтрация": 0.3158}, "capacity": 4000},
    "Солтүстік Қазақстан": {"temp": (5, 23), "salinity": (1000, 3000), "hard": 0.2, "methods": {"кері осмос": 0.25, "нанофильтрация": 0.75}, "capacity": 800},
    "Жамбыл": {"temp": (15, 32), "salinity": (2000, 6000), "hard": 0.6, "methods": {"кері осмос": 0.4286, "нанофильтрация": 0.5714}, "capacity": 2000},
    "Шымкент": {"temp": (18, 38), "salinity": (3000, 7500), "hard": 0.7, "methods": {"кері осмос": 0.5, "нанофильтрация": 0.5}, "capacity": 2500},
    "Ақтөбе": {"temp": (10, 30), "salinity": (2500, 6000), "hard": 0.5, "methods": {"кері осмос": 0.2857, "нанофильтрация": 0.7143}, "capacity": 1800},
    "Түркістан": {"temp": (20, 37), "salinity": (4500, 8500), "hard": 0.8, "methods": {"кері осмос": 0.5556, "нанофильтрация": 0.4444}, "capacity": 3000},
    "Павлодар": {"temp": (5, 25), "salinity": (2000, 4800), "hard": 0.4, "methods": {"кері осмос": 0.3, "нанофильтрация": 0.7}, "capacity": 1500}
}

start_time = datetime(2024, 7, 1, 0, 0, 0)
data = []
n = 500

for i in range(n):
    region = random.choice(list(regions.keys()))
    props = regions[region]
    code = list(regions.keys()).index(region)
    time = start_time + timedelta(minutes=30 * i)
    hour = time.hour
    month = time.month

    seasonal_temp_shift = 6 * np.sin(2 * np.pi * (month - 6) / 12)
    seasonal_salinity_shift = 300 * np.sin(2 * np.pi * (month - 8) / 12)
    temp = np.clip(np.random.uniform(*props["temp"]) + seasonal_temp_shift + (-2 if hour < 6 else 3 if 12 < hour < 17 else 0), 0, 50)
    salinity = np.clip(np.random.normal(np.mean(props["salinity"]), 600) + seasonal_salinity_shift, 500, 16000)

    ph = np.clip(np.random.normal(7.4 - 0.00004 * salinity, 0.3), 6.0, 8.5)
    pressure = np.clip(np.random.normal(4.5, 0.9), 2.0, 7.5)
    level = np.clip(np.random.normal(60, 20), 10, 120)

    method = random.choices(
        ["кері осмос", "нанофильтрация"],
        weights=[props["methods"]["кері осмос"], props["methods"]["нанофильтрация"]]
    )[0]
    filter_efficiency = np.clip(np.random.normal(0.85, 0.05), 0.7, 0.95)

    membrane_age = np.random.randint(30, 730)
    maintenance_status = 1 if np.random.rand() < 0.02 else 0
    plant_capacity = props["capacity"]

    anomaly_type = random.choices(
        ["none", "high_salinity", "low_ph", "pressure_spike", "membrane_fouling"],
        weights=[0.92, 0.03, 0.02, 0.02, 0.01]
    )[0]
    anomaly = 1 if anomaly_type != "none" else 0
    if anomaly_type == "high_salinity":
        salinity *= np.random.uniform(1.3, 1.8)
    elif anomaly_type == "low_ph":
        ph = np.random.uniform(4.5, 5.5)
    elif anomaly_type == "pressure_spike":
        pressure *= np.random.uniform(1.5, 2.0)
    elif anomaly_type == "membrane_fouling":
        filter_efficiency *= np.random.uniform(0.6, 0.8)

    output_pressure = np.clip(
        0.002 * salinity + 0.04 * temp - 0.25 * ph + 0.1 * pressure - 0.05 * (membrane_age / 365) + np.random.normal(0, 0.3),
        0, 10
    )

    energy_base = (2.5 if method == "кері осмос" else 1.5) * pressure / filter_efficiency
    energy_consumption = np.clip(energy_base + np.random.normal(0, 0.2), 0.5, 5.0)

    energy_cost = energy_consumption * 0.1
    maintenance_cost = (0.2 if maintenance_status else 0.05) + 0.01 * (membrane_age / 365)
    operational_cost = np.clip(energy_cost + maintenance_cost, 0.1, 2.0)

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
        "фильтр_тиімділігі": round(filter_efficiency, 2),
        "мембрана_жасы": membrane_age,
        "техникалық_жағдай": maintenance_status,
        "зауыт_сыйымдылығы": plant_capacity,
        "энергия_шығыны": round(energy_consumption, 2),
        "операциялық_шығын": round(operational_cost, 2)
    })

output = pd.DataFrame(data)
output.to_csv("data/sensor_data_kz_realistic.csv", index=False)
print(f"✅ Dataset updated: data/sensor_data_kz_realistic.csv with {len(output)} records")