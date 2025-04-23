import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(77)

regions = {
    "Маңғыстау": {"temp": (25, 38), "salinity": (7000, 10000)},
    "Қызылорда": {"temp": (20, 33), "salinity": (5000, 8000)},
    "Алматы": {"temp": (15, 30), "salinity": (2000, 4000)},
    "Қостанай": {"temp": (5, 20), "salinity": (1000, 3000)},
    "Атырау": {"temp": (22, 35), "salinity": (6000, 9000)}
}

def generate_data(n=1000):
    data = []
    start_time = datetime(2025, 4, 1, 0, 0, 0)

    for i in range(n):
        region = random.choice(list(regions.keys()))
        r_code = list(regions.keys()).index(region)
        temp_range = regions[region]["temp"]
        sal_range = regions[region]["salinity"]

        time = start_time + timedelta(minutes=30 * i)
        hour = time.hour

        # Температурный фактор по времени суток
        if 0 <= hour <= 6:
            temp_factor = -3
        elif 7 <= hour <= 12:
            temp_factor = +0
        elif 13 <= hour <= 17:
            temp_factor = +2
        else:
            temp_factor = -1

        температура = np.clip(np.random.uniform(*temp_range) + temp_factor, 0, 45)
        тұздылық = np.random.uniform(*sal_range)
        pH = np.random.uniform(6.5, 8.2)
        кіру_қысымы = np.random.uniform(2.0, 6.0)
        су_деңгейі = np.random.uniform(20, 100)

        # Аномалия генерациясы
        anomaly = 0
        if np.random.rand() < 0.05:
            pH = np.random.uniform(4.5, 5.5)
            anomaly = 1

        # Шығыс қысымын есептеу
        шығыс_қысымы = (
            0.0025 * тұздылық +
            0.05 * температура -
            0.3 * pH +
            np.random.normal(0, 0.3)
        )

        # Әдіс пен саты логикасы
        if тұздылық > 7000:
            method = "кері осмос"
            stage = "тұщыландыру"
        elif тұздылық > 4000:
            method = "нанофильтрация"
            stage = "алдын ала сүзу"
        elif температура > 30:
            method = "қайнау"
            stage = "буландыру"
        else:
            method = "мембраналық сүзу"
            stage = "алдын ала сүзу"

        қатты_су = 1 if тұздылық > 6000 else 0
        жоғары_қысым_талап = 1 if шығыс_қысымы > 5 else 0

        data.append({
            "уақыт": time.strftime("%Y-%m-%d %H:%M"),
            "өңір": region,
            "өңір_код": r_code,
            "температура": температура,
            "тұздылық": тұздылық,
            "pH": pH,
            "кіру_қысымы": кіру_қысымы,
            "су_деңгейі": су_деңгейі,
            "шығыс_қысымы": шығыс_қысымы,
            "аномалия": anomaly,
            "әдіс": method,
            "процесс_сатысы": stage,
            "қатты_су": қатты_су,
            "жоғары_қысым_талап": жоғары_қысым_талап
        })

    return pd.DataFrame(data)

df = generate_data(1000)
df.to_csv("data/sensor_data_kz_realistic.csv", index=False)
print("✅ Реалистичный датасет сохранён: data/sensor_data_kz_realistic.csv")
