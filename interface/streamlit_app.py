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

with st.expander("📊 🔬 Анализ кезеңдері (визуалды)"):
    st.subheader("1. 📁 Датасет құрылымы") 
    st.dataframe(df.head())

    st.subheader("2. 🌍 Барлық өңірлер бойынша визуализация")
    st.markdown("Өңірді таңдаңыз:")
    selected_region = st.selectbox("Өңір", sorted(df["өңір"].unique()))
    filtered_df = df[df["өңір"] == selected_region]

    fig_map = px.histogram(filtered_df, x="әдіс",
                           title=f"{selected_region} өңіріндегі тұщыландыру әдістерінің жиілігі",
                           labels={"әдіс": "Әдіс", "count": "Жазба саны"})
    st.plotly_chart(fig_map)

    st.subheader("3. 📊 Параметрлер арасындағы байланыс") 
    numeric_cols = ["температура", "тұздылық", "pH", "кіру_қысымы", "шығыс_қысымы"]
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Корреляциялық матрица")
    st.plotly_chart(fig_corr)

    st.subheader("4. 🧠 Модель қалай үйретілді") 
    st.markdown("""
    - Модель: `RandomForestRegressor`
    - Нысана (target): `шығыс_қысымы`
    - Тәуелсіз айнымалылар (X): `температура`, `тұздылық`, `pH`, `қысым`, `су деңгейі`, `өңір_код`
    - Тренинг: `data/sensor_data_kz_realistic.csv` файлы негізінде
    """)

    ...
    st.subheader("5. 🧪 Модель қалай болжам береді?")
    valid_df = filtered_df[filtered_df["әдіс"].isin(["нанофильтрация", "кері осмос"])]

    if len(valid_df) == 0:
        st.warning(f"{selected_region} өңірінде 'нанофильтрация' немесе 'кері осмос' әдістері тіркелмеген.")
    else:
        example = valid_df.sample(1).iloc[0]
        st.markdown(f"""
        Мысал жазба:
        - Өңір: **{example['өңір']}**
        - Температура: **{example['температура']}** °C
        - Тұздылық: **{example['тұздылық']}** ppm
        - pH: **{example['pH']}**
        - Кіру қысымы: **{example['кіру_қысымы']}** бар
        - Су деңгейі: **{example['су_деңгейі']}** см
        - Қолданылған әдіс: **{example['әдіс']}**
        """)

        predict_input = pd.DataFrame([{
            "өңір_код": example['өңір_код'],
            "температура": example['температура'],
            "тұздылық": example['тұздылық'],
            "pH": example['pH'],
            "кіру_қысымы": example['кіру_қысымы'],
            "су_деңгейі": example['су_деңгейі']
        }])

        predicted = model.predict(predict_input)[0]
        st.success(f"📌 Модель болжаған шығыс қысымы: **{predicted:.2f} бар**")

        st.subheader("6. 💧 Екі кезеңді тұщыландыру визуалды")
        R_nano, R_ro = 0.6, 0.95
        sal_0 = example['тұздылық']
        sal_nano = sal_0 * (1 - R_nano)
        sal_ro = sal_nano * (1 - R_ro)

        stages_df = pd.DataFrame({
            "Кезең": ["Бастапқы", "Нанофильтрация", "Кері осмос"],
            "Тұздылық (ppm)": [sal_0, sal_nano, sal_ro]
        })
        fig_stages = px.line(stages_df, x="Кезең", y="Тұздылық (ppm)", markers=True,
                             title="Тұщыландыру кезеңдеріндегі тұздылықтың төмендеуі")
        st.plotly_chart(fig_stages)
        if sal_ro <= 500:
            st.success(f"🟢 Соңғы тұздылық: {sal_ro:.2f} ppm — су ауыз суға жарамды")
        else:
            st.error(f"🔴 Соңғы тұздылық: {sal_ro:.2f} ppm — су әлі де жарамсыз")
#sdasda