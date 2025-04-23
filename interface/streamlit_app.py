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

# Параметрлерді енгізу
region_mapping = {
    "Маңғыстау": 0,
    "Қызылорда": 1,
    "Алматы": 2,
    "Қостанай": 3,
    "Атырау": 4
}
region_name = st.selectbox("Өңірді таңдаңыз", list(region_mapping.keys()))
region_code = region_mapping[region_name]
temperature = st.slider("Температура (°C)", 0.0, 45.0, 25.0)
salinity = st.slider("Тұздылық (ppm)", 1000, 10000, 4000)
ph = st.slider("pH", 5.0, 9.0, 7.0)
input_pressure = st.slider("Кіру қысымы (бар)", 1.0, 6.0, 3.0)
water_level = st.slider("Су деңгейі (см)", 0.0, 100.0, 60.0)

if st.button("🔍 Болжам жасау"):
    input_data = pd.DataFrame([{
        "өңір_код": region_code,
        "температура": temperature,
        "тұздылық": salinity,
        "pH": ph,
        "кіру_қысымы": input_pressure,
        "су_деңгейі": water_level
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"🔧 Болжанған шығыс қысымы: {prediction:.2f} бар")

    new_row = {
        "өңір": region_name,
        "температура": temperature,
        "тұздылық": salinity,
        "pH": ph,
        "кіру_қысымы": input_pressure,
        "су_деңгейі": water_level,
        "шығыс_қысымы": prediction
    }
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv(history_path, index=False)

# Графиктер және анализ бөлімі
st.header("📊 Анализ бөлімі")

if st.checkbox("Гистограмма (таралуы)"):
    st.subheader("Су параметрлерінің таралуы")
    fig1 = px.histogram(df, x="шығыс_қысымы", nbins=50, title="Шығыс қысымының таралуы")
    st.plotly_chart(fig1)

if st.checkbox("Әдістер бойынша қысым (Boxplot)"):
    st.subheader("Әдістер мен қысым арасындағы байланыс")
    fig2 = px.box(df, x="әдіс", y="шығыс_қысымы", title="Әдістер бойынша шығыс қысымы")
    st.plotly_chart(fig2)

if st.checkbox("Уақыт бойынша қысым (Line)"):
    st.subheader("Уақыт бойынша қысым динамикасы")
    fig3 = px.line(df, x="уақыт", y="шығыс_қысымы", title="Шығыс қысымының уақыт бойынша өзгерісі")
    st.plotly_chart(fig3)

if st.checkbox("Аномалиялар (pH және қысым)"):
    st.subheader("Аномалиялық мәндер бойынша диаграмма")
    anom_df = df[df["аномалия"] == 1]
    fig4 = px.scatter(anom_df, x="pH", y="шығыс_қысымы", color="әдіс",
                      title="Аномалиялық жағдайлардағы pH пен қысым")
    st.plotly_chart(fig4)

if st.checkbox("Қатты су жиілігі"):
    st.subheader("Әдістердегі қатты судың үлесі")
    pivot = df.pivot_table(index="әдіс", values="қатты_су", aggfunc="mean")
    st.bar_chart(pivot)

if st.checkbox("Жоғары қысым талап етілетін әдістер"):
    st.subheader("Әдістер бойынша жоғары қысым жиілігі")
    pivot2 = df.pivot_table(index="әдіс", values="жоғары_қысым_талап", aggfunc="mean")
    st.bar_chart(pivot2)

if st.checkbox("📥 Деректерді жүктеу (CSV)"):
    st.download_button("⬇️ CSV-ті жүктеу", data=df.to_csv(index=False), file_name="kz_data.csv")
