import pandas as pd
import numpy as np

regions = ['Маңғыстау', 'Қызылорда', 'Алматы', 'Қостанай', 'Атырау']
n = 300
np.random.seed(77)

data = []
for _ in range(n):
    region = np.random.choice(regions)
    temp = np.random.uniform(10, 30)
    salinity = np.random.uniform(3000, 10000)
    ph = np.random.uniform(6.5, 8.5)
    pressure = np.random.uniform(2, 6)
    level = np.random.uniform(20, 100)
    region_code = regions.index(region)
    target = 0.0025 * salinity + 0.05 * temp - 0.3 * ph + np.random.normal(0, 0.3)
    
    data.append({
        'өңір': region,
        'өңір_код': region_code,
        'температура': temp,
        'тұздылық': salinity,
        'pH': ph,
        'кіру_қысымы': pressure,
        'су_деңгейі': level,
        'шығыс_қысымы_болжам': target
    })

df = pd.DataFrame(data)
df.to_csv("data/sensor_data_kz.csv", index=False)
print("✅ sensor_data_kz.csv сәтті генерацияланды")