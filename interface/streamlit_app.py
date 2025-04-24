
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load model and data
model = joblib.load("models/kz_model.pkl")
df = pd.read_csv("data/sensor_data_kz_realistic.csv")
df["уақыт"] = pd.to_datetime(df["уақыт"])

# Set page configuration
st.set_page_config(page_title="Суды тұщыландыру жүйесі", layout="wide", page_icon="💧")

# Custom CSS for dark mode compatibility
st.markdown("""
<style>
    :root {
        --text-color: #333;
        --bg-color: #F5F7FA;
        --info-bg: #E3F2FD;
        --button-bg: #1E88E5;
        --button-text: #FFFFFF;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    .main-title { 
        font-size: 2.5em; 
        color: #1E88E5; 
        text-align: center; 
        margin-bottom: 20px; 
    }
    .stage-title { 
        font-size: 1.8em; 
        color: #1565C0; 
        margin-top: 20px; 
    }
    .info-box { 
        background-color: var(--info-bg); 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px; 
        border:  direttamente 1px solid #BBDEFB;
    }
    .stButton>button { 
        background-color: var(--button-bg); 
        color: var(--button-text); 
        border-radius: 8px; 
        border: none;
    }
    .stTabs { 
        background-color: var(--info-bg); 
        padding: 10px; 
        border-radius: 10px; 
    }
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #E0E0E0;
            --bg-color: #1A1A1A;
            --info-bg: #263238;
            --button-bg: #42A5F5;
            --button-text: #FFFFFF;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .main-title, .stage-title {
            color: #42A5F5;
        }
        .info-box {
            border: 1px solid #546E7A;
        }
    }
</style>
""", unsafe_allow_html=True)

# Detect theme for Plotly
theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"

# Main title
st.markdown('<div class="main-title">💧 Суды тұщыландыру – Интерактивті қысым болжамы</div>', unsafe_allow_html=True)
st.markdown("Бұл бағдарлама суды тұщыландыру процесін талдауға және қысымды болжауға арналған. Әр кезеңді зерттеу үшін төмендегі қойындыларды қолданыңыз.")

# Tabs for each stage
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Датасет", 
    "2. Өңірлер", 
    "3. Корреляция", 
    "4. Модельді үйрету", 
    "5. Болжам", 
    "6. Тұщыландыру", 
    "7. Өңір статистикасы", 
    "8. Параметр маңыздылығы"
])

# Stage 1: Dataset Structure
with tab1:
    st.markdown('<div class="stage-title">📁 1. Датасет құрылымы</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Сенсорлардан жиналған деректер: температура, тұздылық, pH, қысым, су деңгейі және әдіс.</div>', unsafe_allow_html=True)
    rows = st.slider("Көрсетілетін жолдар саны", 5, 20, 5, key="dataset_rows")
    st.dataframe(df.head(rows), use_container_width=True)

# Stage 2: Regional Visualization
with tab2:
    st.markdown('<div class="stage-title">🌍 2. Өңірлер бойынша визуализация</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Өңір таңдап, тұщыландыру әдістерінің таралуын гистограммада көріңіз.</div>', unsafe_allow_html=True)
    selected_region = st.selectbox("Өңірді таңдаңыз:", sorted(df["өңір"].unique()), key="region_select")
    filtered_df = df[df["өңір"] == selected_region]
    fig_map = px.histogram(filtered_df, x="әдіс", title=f"{selected_region} өңіріндегі әдістер жиілігі", color="әдіс", template=theme)
    st.plotly_chart(fig_map, use_container_width=True)

# Stage 3: Parameter Correlation
with tab3:
    st.markdown('<div class="stage-title">📊 3. Параметрлер арасындағы байланыс</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Корреляциялық матрица параметрлердің өзара байланысын көрсетеді.</div>', unsafe_allow_html=True)
    numeric_cols = ["температура", "тұздылық", "pH", "кіру_қысымы", "шығыс_қысымы"]
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Корреляциялық матрица", color_continuous_scale="RdBu", template=theme)
    st.plotly_chart(fig_corr, use_container_width=True)

# Stage 4: Model Training
with tab4:
    st.markdown('<div class="stage-title">🧠 4. Модель қалай үйретілді?</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">RandomForestRegressor шығыс қысымын болжау үшін параметрлерді талдайды.</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Модель**: RandomForestRegressor  
    - **Нысана**: Шығыс қысымы  
    - **Параметрлер**: Температура, Тұздылық, pH, Кіру қысымы, Су деңгейі, Өңір коды
    """)
    st.button("Модель туралы толығырақ", help="RandomForestRegressor – ағаштар ансамбліне негізделген алгоритм.")

# Stage 5: Model Prediction
with tab5:
    st.markdown('<div class="stage-title">🧪 5. Модель болжамы</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Модель шығыс қысымын болжайды және формуламен салыстырылады.</div>', unsafe_allow_html=True)
    valid_df = filtered_df[filtered_df["әдіс"].isin(["нанофильтрация", "кері осмос"])]
    if len(valid_df) == 0:
        st.warning(f"{selected_region} өңірінде 'нанофильтрация' немесе 'кері осмос' әдістері жоқ.")
        example = None
    else:
        example = valid_df.sample(1).iloc[0]
        st.markdown(f"""
        **Мысал деректер:**  
        - Өңір: {example['өңір']}  
        - Температура: {example['температура']} °C  
        - Тұздылық: {example['тұздылық']} ppm  
        - pH: {example['pH']}  
        - Кіру қысымы: {example['кіру_қысымы']} бар  
        - Су деңгейі: {example['су_деңгейі']} см
        """)
        predict_input = pd.DataFrame([{
            "өңір_код": example["өңір_код"],
            "температура": example["температура"],
            "тұздылық": example["тұздылық"],
            "pH": example["pH"],
            "кіру_қысымы": example["кіру_қысымы"],
            "су_деңгейі": example["су_деңгейі"]
        }])
        predicted = model.predict(predict_input)[0]
        st.success(f"📌 Болжанған қысым: **{predicted:.2f} бар**")
        
        if predicted < 3:
            st.warning("⚠️ Қысым тым төмен – сүзу тиімсіз.")
        elif predicted > 6:
            st.error("❗ Қысым тым жоғары – мембрана зақымдалуы мүмкін!")
        else:
            st.info("✅ Қысым оптималды!")
        
        pressure_formula = 0.0025 * example['тұздылық'] + 0.05 * example['температура'] - 0.3 * example['pH']
        st.write(f"Формула бойынша қысым: **{pressure_formula:.2f} бар**")
        
        weights = {
            "Тұздылық": 0.0025 * example['тұздылық'],
            "Температура": 0.05 * example['температура'],
            "pH (теріс)": -0.3 * example['pH']
        }
        df_weights = pd.DataFrame.from_dict(weights, orient="index", columns=["Қосқан үлесі (бар)"]).reset_index()
        df_weights.rename(columns={"index": "Фактор"}, inplace=True)
        fig_formula = px.bar(df_weights, x="Фактор", y="Қосқан үлесі (бар)", title="Формула параметрлерінің үлесі", text_auto=True, template=theme)
        st.plotly_chart(fig_formula, use_container_width=True)

# Stage 6: Two-Stage Desalination
with tab6:
    st.markdown('<div class="stage-title">💧 6. Екі кезеңді тұщыландыру</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Нанофильтрация және кері осмос арқылы тұздылықтың төмендеуі.</div>', unsafe_allow_html=True)
    st.markdown("""
    **Формулалар:**  
    - Нанофильтрация: $$TDS_{nano} = TDS_0 \times (1 - R_{nano})$$  
    - Кері осмос: $$TDS_{ro} = TDS_{nano} \times (1 - R_{ro})$$
    """)
    if example is None:
        st.warning("Алдыңғы кезеңде деректер таңдалмаған. Өңірде 'нанофильтрация' немесе 'кері осмос' әдістері болуы керек.")
    else:
        R_nano, R_ro = 0.6, 0.95
        sal_0 = example['тұздылық']
        sal_nano = sal_0 * (1 - R_nano)
        sal_ro = sal_nano * (1 - R_ro)
        stages_df = pd.DataFrame({
            "Кезең": ["Бастапқы", "Нанофильтрация", "Кері осмос"],
            "Тұздылық (ppm)": [sal_0, sal_nano, sal_ro]
        })
        fig_stages = px.line(stages_df, x="Кезең", y="Тұздылық (ppm)", markers=True, title="Тұздылықтың төмендеуі", template=theme)
        st.plotly_chart(fig_stages, use_container_width=True)
        if sal_ro <= 500:
            st.success(f"🟢 Тұздылық: {sal_ro:.2f} ppm – ауыз суға жарамды!")
        else:
            st.error(f"🔴 Тұздылық: {sal_ro:.2f} ppm – су жарамсыз.")

# Stage 7: Regional Statistics
with tab7:
    st.markdown('<div class="stage-title">📊 7. Өңірлер бойынша статистика</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Өңірлердің орташа параметрлері. Ең жоғары мәндер ерекшеленеді.</div>', unsafe_allow_html=True)
    region_summary = df.groupby("өңір")[["температура", "тұздылық", "pH", "кіру_қысымы", "шығыс_қысымы"]].mean().round(2).reset_index()
    st.dataframe(region_summary.style.highlight_max(axis=0), use_container_width=True)

# Stage 8: Feature Importance
with tab8:
    st.markdown('<div class="stage-title">🧠 8. Параметрлердің маңыздылығы</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Модельдің болжамға қай параметрлер көбірек әсер ететіні.</div>', unsafe_allow_html=True)
    features = ["өңір_код", "температура", "тұздылық", "pH", "кіру_қысымы", "су_деңгейі"]
    importances = model.feature_importances_
    df_feat = pd.DataFrame({"Фактор": features, "Маңыздылығы": importances})
    fig_feat = px.bar(df_feat.sort_values("Маңыздылығы", ascending=True), x="Маңыздылығы", y="Фактор", orientation="h", title="Параметрлердің маңыздылығы", template=theme)
    st.plotly_chart(fig_feat, use_container_width=True)
