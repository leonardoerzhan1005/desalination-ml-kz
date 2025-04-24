import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF

# Модельді жүктеу
model = joblib.load("models/kz_model.pkl")

# Жоғары дәлдікті дата
history_path = "data/prediction_history.csv"
if os.path.exists(history_path):
    history_df = pd.read_csv(history_path)
else:
    history_df = pd.DataFrame(columns=[
        "өңір", "температура", "тұздылық", "pH", "кіру_қысымы", "су_деңгейі", "шығыс_қысымы"
    ])

# Реалистік датасетті жүктеу (анализ үшін)
df = pd.read_csv("data/sensor_data_kz_realistic.csv")
df["уақыт"] = pd.to_datetime(df["уақыт"])

st.set_page_config(page_title="Суды тұщыландыру жүйесі", layout="centered")
st.title("💧 Суды тұщыландыру – қысым болжамы")

with st.expander("ℹ️ Жүйе туралы ақпарат"):
    st.markdown("""
    Бұл жүйе **суды тұщыландыру процесін автоматты басқаруға арналған**.

    🧠 **Жасанды интеллект** арқылы енгізілген параметрлерге қарап,
    қажетті шығыс қысымын болжайды.

    **Параметрлер:** температура, тұздылық, pH, қысым, су деңгейі, өңір
    """)

# 📐 Формулалар мен теория
with st.expander("📐 Формулалар мен теория"):
    st.latex(r"""
    \text{P}_{шығыс} = 0.0025 \cdot \text{TDS} + 0.05 \cdot \text{Temp} - 0.3 \cdot \text{pH} + \varepsilon
    """)
    st.latex(r"""
    \pi = iMRT
    """)
    st.markdown("""
    **Мұндағы:**
    - **TDS** — тұздылық (ppm)
    - **Temp** — температура (°C)
    - **pH** — су қышқылдығы
    - **\( \varepsilon \)** — модельдегі кездейсоқ ауытқу (нормаль бөлінген)
    - **\( \pi \)** — осмостық қысым
    - **i** — Ван-Гофф коэффициенті
    - **M** — молярлық концентрация
    - **R** — газ тұрақтысы
    - **T** — температура (К)
    """)

# 🔄 Су тұщыландыру процесін имитациялау (2 кезең)
st.header("🚰 Су тұщыландыру кезеңдері")
initial_salinity = st.number_input("Бастапқы тұздылық (ppm)", value=6000, step=100)
temperature = st.slider("Температура (°C)", 0.0, 45.0, 25.0)
ph = st.slider("pH", 5.0, 9.0, 7.0)
input_pressure = st.slider("Кіру қысымы (бар)", 1.0, 6.0, 3.0)

# Нанофильтрация нәтижесі
nano = {
    "Тұздылық": initial_salinity * 0.5,
    "pH": ph - 0.2,
    "Қысым": input_pressure + 0.5,
    "Температура": temperature
}

# Кері осмос нәтижесі
osmosis = {
    "Тұздылық": nano["Тұздылық"] * 0.1,
    "pH": nano["pH"] - 0.1,
    "Қысым": nano["Қысым"] + 1.5,
    "Температура": nano["Температура"]
}

# Кестелік көрініс және график
params_df = pd.DataFrame([
    {"Кезең": "Бастапқы", **{"Тұздылық": initial_salinity, "pH": ph, "Қысым": input_pressure, "Температура": temperature}},
    {"Кезең": "Нанофильтрация", **nano},
    {"Кезең": "Кері осмос", **osmosis}
])

st.subheader("Су параметрлерінің кезең бойынша өзгерісі")
st.dataframe(params_df.set_index("Кезең"))
fig = px.line(params_df, x="Кезең", y=["Тұздылық", "pH", "Қысым", "Температура"], markers=True,
              title="Әр әдістен кейінгі параметрлердің өзгерісі")
st.plotly_chart(fig)

# Қорытынды
st.subheader("Қорытынды")
final_salinity = osmosis["Тұздылық"]
if final_salinity <= 500:
    st.success(f"🟢 Соңғы тұздылық: {final_salinity:.2f} ppm — бұл су ауыз су үшін жарамды ✅")
else:
    st.error(f"🔴 Соңғы тұздылық: {final_salinity:.2f} ppm — бұл су әлі де жарамсыз ⚠️")
