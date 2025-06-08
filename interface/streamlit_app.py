import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit.components.v1 as components

# Load models and data
try:
    model = joblib.load("models/kz_model.pkl")
    anomaly_model = joblib.load("models/anomaly_model.pkl")
except FileNotFoundError:
    st.error("Модель файлдары табылмады. 'kz_model.pkl' және 'anomaly_model.pkl' файлдарын 'models/' қалтасына орналастырыңыз.")
    st.stop()

try:
    df = pd.read_csv("data/sensor_data_kz_realistic.csv")
    df["уақыт"] = pd.to_datetime(df["уақыт"])
except FileNotFoundError:
    st.error("Деректер файлы табылмады. 'data/sensor_data_kz_realistic.csv' файлын 'data/' қалтасына орналастырыңыз.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Суды тұщыландыру жүйесі", layout="wide", page_icon="💧")

# Custom CSS for modern UI
st.markdown("""
<style>
    :root {
        --text-color: #333;
        --bg-color: #F5F7FA;
        --info-bg: #E3F2FD;
        --button-bg: #1E88E5;
        --button-text: #FFFFFF;
        --border-color: #BBDEFB;
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
        padding: 20px; 
        border-radius: 15px; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px; 
        border: 1px solid var(--border-color);
    }
    .stButton>button { 
        background-color: var(--button-bg); 
        color: var(--button-text); 
        border-radius: 8px; 
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .stSlider > div > div > div {
        background-color: #42A5F5;
        border-radius: 10px;
    }
    .stTabs { 
        background-color: var(--info-bg); 
        padding: 10px; 
        border-radius: 15px; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #E0E0E0;
            --bg-color: #1A1A1A;
            --info-bg: #263238;
            --button-bg: #42A5F5;
            --button-text: #FFFFFF;
            --border-color: #546E7A;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .main-title, .stage-title {
            color: #42A5F5;
        }
        .info-box {
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
        }
    }
</style>
""", unsafe_allow_html=True)

# Detect theme for Plotly
theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"

# Keyboard navigation
components.html("""
<script>
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight') {
            Streamlit.setComponentValue('next_tab');
        } else if (e.key === 'ArrowLeft') {
            Streamlit.setComponentValue('prev_tab');
        }
    });
</script>
""")
tab_action = st.text_input("Tab action", label_visibility="hidden")
if tab_action == "next_tab" and st.session_state.get('current_tab', 0) < 7:
    st.session_state['current_tab'] += 1
    st.rerun()
elif tab_action == "prev_tab" and st.session_state.get('current_tab', 0) > 0:
    st.session_state['current_tab'] -= 1
    st.rerun()

# Main title and help
st.markdown('<div class="main-title">💧 Суды тұщыландыру – Интерактивті қысым болжамы</div>', unsafe_allow_html=True)
st.markdown("Бұл бағдарлама суды тұщыландыру процесін талдауға, қысымды, энергияны және шығындарды болжауға арналған.")

with st.expander("ℹ️ Анықтама"):
    st.markdown("""
    - **1-кезең**: Деректер құрылымын зерттеңіз.
    - **5-кезең**: Қысым, энергия және шығын болжамдарын көріңіз.
    - **6-кезең**: Тұщыландыру процесін реттеңіз және шығындарды есептеңіз.
    - **7-кезең**: Өңірлер бойынша шығындарды салыстырыңыз.
    """)

# Display the "Interactive Tutorial" label without functionality
st.markdown("📖 Интерактивті нұсқаулық")

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
    st.markdown('<div class="info-box">Сенсорлардан жиналған деректер: температура, тұздылық, pH, қысым, су деңгейі, шығындар және әдіс.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Сенсор деректерін жүктеу (CSV)", type="csv")
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)
        required_cols = ["өңір", "температура", "тұздылық", "pH", "кіру_қысымы", "су_деңгейі", "фильтр_тиімділігі", "мембрана_жасы", "техникалық_жағдай", "зауыт_сыйымдылығы"]
        if all(col in uploaded_df.columns for col in required_cols):
            df = uploaded_df
            st.success("Деректер сәтті жүктелді!")
        else:
            st.error("CSV файлы қажетті бағандарды қамтымайды.")
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
    numeric_cols = ["температура", "тұздылық", "pH", "кіру_қысымы", "шығыс_қысымы", "энергия_шығыны", "операциялық_шығын"]
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Корреляциялық матрица", color_continuous_scale="RdBu", template=theme)
    st.plotly_chart(fig_corr, use_container_width=True)

# Stage 4: Model Training
with tab4:
    st.markdown('<div class="stage-title">🧠 4. Модель қалай үйретілді?</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">RandomForestRegressor қысымды, энергияны және шығындарды болжау үшін параметрлерді талдайды.</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Модель**: MultiOutputRegressor(RandomForestRegressor)  
    - **Нысаналар**: Шығыс қысымы, Энергия шығыны, Операциялық шығын  
    - **Параметрлер**: Өңір коды, Температура, Тұздылық, pH, Кіру қысымы, Су деңгейі, Фильтр тиімділігі, Мембрана жасы, Техникалық жағдай, Зауыт сыйымдылығы
    """)
    st.button("Модель туралы толығырақ", help="MultiOutputRegressor бірнеше нысананы болжауға мүмкіндік береді.")

# Stage 5: Model Prediction
with tab5:
    st.markdown('<div class="stage-title">🧪 5. Модель болжамы</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Модель қысымды, энергияны және шығындарды болжайды, аномалияларды анықтайды.</div>', unsafe_allow_html=True)
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
        - Фильтр тиімділігі: {example['фильтр_тиімділігі']}  
        - Мембрана жасы: {example['мембрана_жасы']} күн  
        - Техникалық жағдай: {'Қызмет көрсетуде' if example['техникалық_жағдай'] else 'Қалыпты'}  
        - Зауыт сыйымдылығы: {example['зауыт_сыйымдылығы']} м³/тәулік
        """)
        predict_input_dict = {
            "өңір_код": example["өңір_код"],
            "температура": example["температура"],
            "тұздылық": example["тұздылық"],
            "pH": example["pH"],
            "кіру_қысымы": example["кіру_қысымы"],
            "су_деңгейі": example["су_деңгейі"],
            "фильтр_тиімділігі": example["фильтр_тиімділігі"],
            "мембрана_жасы": example["мембрана_жасы"],
            "техникалық_жағдай": example["техникалық_жағдай"],
            "зауыт_сыйымдылығы": example["зауыт_сыйымдылығы"]
        }
        predict_input = pd.DataFrame([predict_input_dict])
        is_anomaly = anomaly_model.predict(predict_input)[0] == -1
        if is_anomaly:
            st.warning("⚠️ Аномалия анықталды! Болжам дәл болмауы мүмкін.")
        predictions = model.predict(predict_input)[0]
        st.success(f"""
        📌 Болжамдар:
        - Шығыс қысымы: **{predictions[0]:.2f} бар**
        - Энергия шығыны: **{predictions[1]:.2f} кВт·сағ/м³**
        - Операциялық шығын: **{predictions[2]:.2f} $/м³**
        """)
        
        if predictions[0] < 3:
            st.warning("⚠️ Қысым тым төмен – сүзу тиімсіз.")
        elif predictions[0] > 6:
            st.error("❗ Қысым тым жоғары – мембрана зақымдалуы мүмкін!")
        else:
            st.info("✅ Қысым оптималды!")
        if predictions[1] > 2.5:
            st.warning("⚠️ Энергия шығыны жоғары – тиімділікті арттырыңыз.")
        
        pressure_formula = 0.002 * example['тұздылық'] + 0.04 * example['температура'] - 0.25 * example['pH'] + 0.1 * example['кіру_қысымы'] - 0.05 * (example['мембрана_жасы'] / 365)
        st.write(f"Формула бойынша қысым: **{pressure_formula:.2f} бар**")
        
        weights = {
            "Тұздылық": 0.002 * example['тұздылық'],
            "Температура": 0.04 * example['температура'],
            "pH (теріс)": -0.25 * example['pH'],
            "Кіру қысымы": 0.1 * example['кіру_қысымы'],
            "Мембрана жасы (теріс)": -0.05 * (example['мембрана_жасы'] / 365)
        }
        df_weights = pd.DataFrame.from_dict(weights, orient="index", columns=["Қосқан үлесі (бар)"]).reset_index()
        df_weights.rename(columns={"index": "Фактор"}, inplace=True)
        fig_formula = px.bar(df_weights, x="Фактор", y="Қосқан үлесі (бар)", title="Формула параметрлерінің үлесі", text_auto=True, template=theme)
        st.plotly_chart(fig_formula, use_container_width=True)

# Stage 6: Two-Stage Desalination (Enhanced)
with tab6:
    st.markdown('<div class="stage-title">💧 6. Екі кезеңді тұщыландыру – Толық талдау</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Нанофильтрация және кері осмос арқылы судың тұздылығын азайту. Өңір мен әдісті таңдап, тиімділікті салыстырыңыз.</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### 📚 Процесс түсіндірмесі
    Екі кезеңді тұщыландыру:
    1. **Нанофильтрация**: Орташа молекулалы тұздарды (50-70% тиімділік) жояды.
    2. **Кері осмос**: Кіші молекулалы қоспаларды (90-98% тиімділік) жояды.

    **Формулалар:**
    - Нанофильтрация: $$TDS_{nano} = TDS_0 \times (1 - R_{nano})$$
    - Кері осмос: $$TDS_{ro} = TDS_{nano} \times (1 - R_{ro})$$
    - Су шығыны: $$RR = \frac{Q_{out}}{Q_{in}} \times 100\%$$
    - Энергия шығыны: $$E = \frac{P \times Q_{in}}{η \times 3600}$$
    """)

    if example is None:
        st.warning("Алдыңғы кезеңде деректер таңдалмаған. 'Нанофильтрация' немесе 'кері осмос' әдісі бар өңірді таңдаңыз.")
    else:
        st.subheader("⚙️ Өңір және әдіс бойынша тиімділікті салыстыру")
        # Select region and method
        region = st.selectbox("Өңірді таңдаңыз:", sorted(df["өңір"].unique()), key="region_select_tab6")
        method = st.selectbox("Әдісті таңдаңыз:", ["кері осмос", "нанофильтрация"], key="method_select")

        # Calculate region-specific averages from dataset
        region_data = df[df["өңір"] == region]
        avg_salinity = region_data["тұздылық"].mean()
        avg_pressure = region_data["кіру_қысымы"].mean()
        avg_efficiency = region_data["фильтр_тиімділігі"].mean()
        avg_flow_rate = 5.0  # Default value
        avg_r_nano = 0.6 if method == "нанофильтрация" else 0.5
        avg_r_ro = 0.95 if method == "кері осмос" else 0.9

        # Initialize session state for parameters
        if 'initial_salinity' not in st.session_state:
            st.session_state.initial_salinity = float(avg_salinity)
        if 'r_nano' not in st.session_state:
            st.session_state.r_nano = avg_r_nano
        if 'r_ro' not in st.session_state:
            st.session_state.r_ro = avg_r_ro
        if 'input_pressure' not in st.session_state:
            st.session_state.input_pressure = float(avg_pressure)
        if 'flow_rate' not in st.session_state:
            st.session_state.flow_rate = avg_flow_rate
        if 'energy_efficiency' not in st.session_state:
            st.session_state.energy_efficiency = float(avg_efficiency)

        # Button to load region averages
        if st.button("Өңірдің орташа параметрлерін жүктеу"):
            st.session_state.initial_salinity = float(avg_salinity)
            st.session_state.r_nano = avg_r_nano
            st.session_state.r_ro = avg_r_ro
            st.session_state.input_pressure = float(avg_pressure)
            st.session_state.flow_rate = avg_flow_rate
            st.session_state.energy_efficiency = float(avg_efficiency)
            st.rerun()

        # Sliders for manual parameter adjustment
        st.markdown("### Параметрлерді реттеу")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.initial_salinity = st.slider("Бастапқы тұздылық (ppm)", 1000.0, 15000.0, st.session_state.initial_salinity, step=100.0)
            st.session_state.r_nano = st.slider("Нанофильтрация тиімділігі (R_nano)", 0.5, 0.8, st.session_state.r_nano, step=0.01)
            st.session_state.r_ro = st.slider("Кері осмос тиімділігі (R_ro)", 0.9, 0.98, st.session_state.r_ro, step=0.01)
        with col2:
            st.session_state.input_pressure = st.slider("Кіріс қысымы (бар)", 2.0, 7.0, st.session_state.input_pressure, step=0.1)
            st.session_state.flow_rate = st.slider("Кіріс ағыны (м³/сағ)", 1.0, 10.0, st.session_state.flow_rate, step=0.5)
            st.session_state.energy_efficiency = st.slider("Энергия тиімділігі (η)", 0.7, 0.9, st.session_state.energy_efficiency, step=0.01)

        # Retrieve values from session state
        initial_salinity = st.session_state.initial_salinity
        r_nano = st.session_state.r_nano
        r_ro = st.session_state.r_ro
        input_pressure = st.session_state.input_pressure
        flow_rate = st.session_state.flow_rate
        energy_efficiency = st.session_state.energy_efficiency

        # Calculations
        sal_nano = initial_salinity * (1 - r_nano)
        sal_ro = sal_nano * (1 - r_ro)
        recovery_nano = 0.6
        recovery_ro = 0.4
        total_recovery = recovery_nano * recovery_ro
        output_flow = flow_rate * total_recovery
        energy_nano = (input_pressure * flow_rate) / (energy_efficiency * 3600) * 0.5
        energy_ro = (input_pressure * flow_rate * 1.5) / (energy_efficiency * 3600)
        total_energy = energy_nano + energy_ro
        operational_cost = total_energy * 0.1 + (0.2 if example['техникалық_жағдай'] else 0.05) + 0.01 * (example['мембрана_жасы'] / 365)

        # Visualization
        st.subheader("📊 Тұздылықтың төмендеуі")
        stages_df = pd.DataFrame({
            "Кезең": ["Бастапқы", "Нанофильтрация", "Кері осмос"],
            "Тұздылық (ppm)": [initial_salinity, sal_nano, sal_ro]
        })
        fig_stages = px.line(stages_df, x="Кезең", y="Тұздылық (ppm)", markers=True, title="Тұздылықтың екі кезеңде төмендеуі", template=theme)
        fig_stages.add_bar(x=stages_df["Кезең"], y=[initial_salinity, sal_nano, sal_ro], name="Тұздылық", opacity=0.3)
        st.plotly_chart(fig_stages, use_container_width=True)

        # Results
        st.subheader("📈 Нәтижелер")
        st.markdown(f"""
        - **Соңғы тұздылық**: {sal_ro:.2f} ppm ({ '🟢 Ауыз суға жарамды' if sal_ro <= 500 else '🔴 Ауыз суға жарамсыз' })
        - **Су шығыны**: {output_flow:.2f} м³/сағ ({total_recovery*100:.1f}%)
        - **Энергия шығыны**: {total_energy:.2f} кВт·сағ/м³  
          - Нанофильтрация: {energy_nano:.2f} кВт·сағ/м³  
          - Кері осмос: {energy_ro:.2f} кВт·сағ/м³
        - **Операциялық шығын**: {operational_cost:.2f} $/м³
        """)

        # Optimization
        st.subheader("🔧 Оптимизация")
        st.markdown("Оптимизируйте процесс, выбрав цель и диапазон целевой солёности.")

        # Select optimization objective
        optimization_goal = st.selectbox(
            "Цель оптимизации:",
            ["Минимизация затрат", "Минимизация энергопотребления", "Баланс затрат и энергии"],
            key="optimization_goal"
        )

        # Input for salinity range
        col_sal1, col_sal2 = st.columns(2)
        with col_sal1:
            min_salinity = st.number_input("Минимальная целевая солёность (ppm)", 100, 500, 300, step=10)
        with col_sal2:
            max_salinity = st.number_input("Максимальная целевая солёность (ppm)", min_salinity, 500, 500, step=10)

        # Define objective function
        def objective_function(params, goal=optimization_goal):
            r_nano, r_ro, pressure = params
            sal_nano = initial_salinity * (1 - r_nano)
            sal_ro = sal_nano * (1 - r_ro)
            energy = (pressure * flow_rate * (1.5 if r_ro > 0.95 else 1.0)) / (energy_efficiency * 3600)
            cost = energy * 0.1 + (0.2 if example['техникалық_жағдай'] else 0.05) + 0.01 * (example['мембрана_жасы'] / 365)
            
            # Penalty for salinity outside target range
            penalty = 0
            if sal_ro < min_salinity or sal_ro > max_salinity:
                penalty = 1000 + abs(sal_ro - (min_salinity + max_salinity) / 2) * 10
            
            # Define objective based on goal
            if goal == "Минимизация затрат":
                return cost + penalty
            elif goal == "Минимизация энергопотребления":
                return energy + penalty
            else:  # Баланс затрат и энергии
                return (cost + energy) / 2 + penalty

        # Run optimization
        initial_guess = [r_nano, r_ro, input_pressure]
        bounds = [(0.5, 0.8), (0.9, 0.98), (2.0, 7.0)]
        result = minimize(objective_function, initial_guess, bounds=bounds, method='SLSQP')

        if result.success:
            opt_r_nano, opt_r_ro, opt_pressure = result.x
            # Calculate optimized values
            opt_sal_nano = initial_salinity * (1 - opt_r_nano)
            opt_sal_ro = opt_sal_nano * (1 - opt_r_ro)
            opt_energy = (opt_pressure * flow_rate * (1.5 if opt_r_ro > 0.95 else 1.0)) / (energy_efficiency * 3600)
            opt_cost = opt_energy * 0.1 + (0.2 if example['техникалық_жағдай'] else 0.05) + 0.01 * (example['мембрана_жасы'] / 365)

            st.success(f"""
            Оптималды параметрлер:
            - R_nano: **{opt_r_nano:.2f}**
            - R_ro: **{opt_r_ro:.2f}**
            - Қысым: **{opt_pressure:.2f} бар**
            - Соңғы тұздылық: **{opt_sal_ro:.2f} ppm**
            - Энергия шығыны: **{opt_energy:.2f} кВт·сағ/м³**
            - Операциялық шығын: **{opt_cost:.2f} $/м³**
            """)

            # Comparison table
            comparison_df = pd.DataFrame({
                "Параметр": ["R_nano", "R_ro", "Қысым (бар)", "Соңғы тұздылық (ppm)", "Энергия шығыны (кВт·сағ/м³)", "Операциялық шығын ($/м³)"],
                "Текущие": [r_nano, r_ro, input_pressure, sal_ro, total_energy, operational_cost],
                "Оптимальные": [opt_r_nano, opt_r_ro, opt_pressure, opt_sal_ro, opt_energy, opt_cost]
            })
            comparison_df["Текущие"] = comparison_df["Текущие"].round(2)
            comparison_df["Оптимальные"] = comparison_df["Оптимальные"].round(2)
            st.subheader("Сравнение текущих и оптимальных параметров")
            st.dataframe(comparison_df, use_container_width=True)

            # Visualization
            st.subheader("Визуализация оптимизации")
            fig_opt = go.Figure(data=[
                go.Bar(name="Текущие", x=["R_nano", "R_ro", "Қысым", "Тұздылық", "Энергия", "Шығын"],
                       y=[r_nano, r_ro, input_pressure, sal_ro, total_energy, operational_cost]),
                go.Bar(name="Оптимальные", x=["R_nano", "R_ro", "Қысым", "Тұздылық", "Энергия", "Шығын"],
                       y=[opt_r_nano, opt_r_ro, opt_pressure, opt_sal_ro, opt_energy, opt_cost])
            ])
            fig_opt.update_layout(
                title="Сравнение текущих и оптимальных параметров",
                barmode='group',
                template=theme,
                yaxis_title="Значение",
                height=500
            )
            st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.error("Оптимизация не удалась. Возможные причины:")
            st.markdown("""
            - Целевая солёность недостижима с текущими параметрами. Попробуйте:
              - Увеличить диапазон целевой солёности.
              - Увеличить эффективность фильтрации (R_nano, R_ro) в слайдерах.
            - Ограничения на параметры слишком строгие. Проверьте входные значения.
            """)

        # Recommendations
        st.subheader("💡 Ұсыныстар")
        if sal_ro > 500:
            st.warning("Тұздылық жоғары. R_nano немесе R_ro тиімділігін арттырыңыз.")
        if total_energy > 2.5:
            st.warning("Энергия шығыны жоғары. Энергия тиімділігін (η) арттырыңыз немесе қысымды азайтыңыз.")
        if total_recovery < 0.2:
            st.warning("Су шығыны төмен. Фильтрлерді тексеріңіз.")
        if operational_cost > 1.0:
            st.warning("Операциялық шығын жоғары. Техникалық қызметті жоспарлаңыз немесе мембрананы ауыстырыңыз.")

# Stage 7: Regional Statistics
with tab7:
    st.markdown('<div class="stage-title">📊 7. Өңірлер бойынша статистика</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Өңірлердің орташа параметрлері мен шығындары. Ең жоғары мәндер ерекшеленеді.</div>', unsafe_allow_html=True)
    region_summary = df.groupby("өңір")[["температура", "тұздылық", "pH", "кіру_қысымы", "шығыс_қысымы", "энергия_шығыны", "операциялық_шығын"]].mean().round(2).reset_index()
    st.dataframe(region_summary.style.highlight_max(axis=0), use_container_width=True)
    cost_summary = df.groupby("өңір")[["энергия_шығыны", "операциялық_шығын"]].mean().reset_index()
    fig_cost = px.bar(cost_summary, x="өңір", y=["энергия_шығыны", "операциялық_шығын"], barmode="group", title="Өңірлер бойынша шығындар", template=theme)
    st.plotly_chart(fig_cost, use_container_width=True)

# Stage 8: Feature Importance
with tab8:
    st.markdown('<div class="stage-title">🧠 8. Параметрлердің маңыздылығы</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Модельдің болжамға қай параметрлер көбірек әсер ететіні.</div>', unsafe_allow_html=True)
    features = model.estimators_[0].feature_names_in_ if hasattr(model.estimators_[0], 'feature_names_in_') else ["өңір_код", "температура", "тұздылық", "pH", "кіру_қысымы", "су_деңгейі", "фильтр_тиімділігі", "мембрана_жасы", "техникалық_жағдай", "зауыт_сыйымдылығы"]
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    df_feat = pd.DataFrame({"Фактор": features, "Маңыздылығы": importances})
    fig_feat = px.bar(df_feat.sort_values("Маңыздылығы", ascending=True), x="Маңыздылығы", y="Фактор", orientation="h", title="Параметрлердің маңыздылығы", template=theme)
    st.plotly_chart(fig_feat, use_container_width=True)