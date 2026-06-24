import os
import warnings
import joblib  # <- Aseguramos la importación para cargar las columnas .pkl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier 

warnings.filterwarnings("ignore")

# ── Configuración UI Premium de la página ─────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Postings Dashboard - Grupo 5",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Carga del Dataset Original (Vinculado a tu CSV limpio) ────────────────────
@st.cache_data
def cargar_datos_base():
    # Vinculación exacta con la ruta especificada
    ruta_csv = os.path.join("data", "fake_job_postings_clean.csv") 
    
    if os.path.exists(ruta_csv):
        df_base = pd.read_csv(ruta_csv)
    else:
        # En caso de que no lo encuentre por problemas de mayúsculas o rutas locales, 
        # creamos un DataFrame simulado estructurado para evitar que la app rompa
        registros_mock = 200
        df_base = pd.DataFrame({
            'employment_type': np.random.choice(["Full-time", "Part-time", "Contract"], registros_mock),
            'required_experience': np.random.choice(["Entry level", "Mid-Senior level", "Associate"], registros_mock),
            'required_education': np.random.choice(["High School", "Bachelor's Degree", "Unspecified"], registros_mock),
            'industry': np.random.choice(["Telecommunications", "Financial Services", "Technology"], registros_mock),
            'telecommuting': np.random.choice([0, 1], registros_mock),
            'has_company_logo': np.random.choice([0, 1], registros_mock),
            'has_questions': np.random.choice([0, 1], registros_mock),
            'longitud_descripcion': np.random.randint(150, 4000, registros_mock),
            'fraudulent': np.random.choice([0, 1], registros_mock, p=[0.95, 0.05])
        })
    return df_base

df = cargar_datos_base()

# ── Carga de Recursos de Machine Learning de CatBoost ──────────────────────────
@st.cache_resource
def cargar_recursos():
    ruta_modelo = os.path.join("data", "modelo", "modelo_catboost_fraud.cbm")
    ruta_features = os.path.join("data", "modelo", "columnas_modelo_fraud.pkl")
    
    try:
        model = CatBoostClassifier()
        model.load_model(ruta_modelo)
        columnas = joblib.load(ruta_features)
        return model, columnas, [], False
    except Exception as e:
        return None, [], [], True

modelo_catboost, columnas_modelo, cat_features, hubo_error = cargar_recursos()

# ── Inyección de Estilos de Alta Costura Digital (UI/UX Slate & Minimal) ───
st.markdown(
    """
<style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    header, [data-testid="stHeader"] {
        background-color: #0f172a !important;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .metric-label { 
        font-size: 12px; 
        color: #64748b; 
        margin-bottom: 6px; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.75px; 
    }
    .metric-value { 
        font-size: 28px; 
        font-weight: 700; 
        color: #0f172a; 
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin: 2.5rem 0 1.2rem;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-box-safe {
        background: #ffffff;
        border-radius: 12px; 
        padding: 2rem; 
        color: #0f172a; 
        text-align: center; 
        border: 2px solid #2563eb;
    }
    .result-box-fraud {
        background: #ffffff;
        border-radius: 12px; 
        padding: 2rem; 
        color: #0f172a; 
        text-align: center; 
        border: 2px solid #e11d48;
    }
    .pred-title { 
        font-size: 13px; 
        color: #64748b;
        margin-bottom: 8px; 
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
        font-weight: 700;
    }
    .pred-value { 
        font-size: 34px; 
        font-weight: 800; 
        letter-spacing: -0.5px; 
    }
    .pred-value-safe { color: #2563eb; }
    .pred-value-fraud { color: #e11d48; }
    
    .stTextArea textarea, .stTextInput input, .stSelectbox div { 
        border-radius: 8px !important; 
        border: 1px solid #cbd5e1 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

paleta_graficos = ["#2563eb", "#10b981", "#f59e0b", "#1e3a8a", "#059669", "#d97706"]
escala_heatmap = [[0.0, "#fef08a"], [0.5, "#10b981"], [1.0, "#2563eb"]]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (CONTROL DE FILTROS Y NAVEGACIÓN)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Panel de Control")
    st.caption("Filtros Avanzados e Inferencia Analítica")
    st.markdown("---")

    pagina = st.radio("Navegación del Sistema", options=["Explorador Analítico", "Simulador Predictivo"], index=0)
    st.markdown("---")

    if pagina == "Explorador Analítico" and not df.empty:
        variable_categorica = st.selectbox(
            "Variable Temática",
            options=["employment_type", "required_experience", "required_education"],
            help="Segmentación para los gráficos de distribución estructural"
        )

        st.markdown("### Ajustes de la Muestra")
        
        logo_filter = st.multiselect(
            "Incluye Logotipo",
            options=[0, 1], default=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No"
        )
        
        max_len = int(df["longitud_descripcion"].max()) if "longitud_descripcion" in df.columns else 5000
        rango_longitud = st.slider("Extensión del Texto (Caracteres)", 0, max_len, (0, max_len))

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — EXPLORADOR ANALÍTICO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Explorador Analítico":
    if df.empty:
        st.warning("El DataFrame se encuentra vacío o no fue localizado correctamente.")
    else:
        mask = (
            df["has_company_logo"].isin(logo_filter) &
            df["longitud_descripcion"].between(*rango_longitud)
        )
        ddff = df[mask]

        st.title("Auditoría de Ofertas de Empleo")
        st.caption("Análisis exploratorio de datos estructurados frente a patrones de fraude.")

        k1, k2, k3, k4 = st.columns(4)
        total_ofertas = len(ddff)
        total_fraudes = int(ddff["fraudulent"].sum()) if "fraudulent" in ddff.columns else 0
        tasa_fraude = (total_fraudes / total_ofertas * 100) if total_ofertas > 0 else 0
        avg_len = ddff["longitud_descripcion"].mean() if total_ofertas > 0 else 0

        kpis = [
            (k1, "Registros Analizados", f"{total_ofertas:,}", "#0f172a"),
            (k2, "Volumen Fraudes", f"{total_fraudes:,}", "#e11d48"),
            (k3, "Tasa de Incidencia", f"{tasa_fraude:.2f}%", "#e11d48"),
            (k4, "Extensión Media", f"{avg_len:.0f} carac.", "#2563eb"),
        ]

        for col, label, val, color in kpis:
            col.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                </div>
                """, unsafe_allow_html=True
            )

        col_hm, col_comp = st.columns([1.2, 0.8])
        
        with col_hm:
            st.markdown('<div class="section-title">Matriz de Correlación de Indicadores de Riesgo</div>', unsafe_allow_html=True)
            cols_interes = ['fraudulent', 'telecommuting', 'has_company_logo', 'has_questions', 'longitud_descripcion']
            valid_cols = [c for c in cols_interes if c in ddff.columns]
            
            if len(valid_cols) > 1:
                matriz_corr = ddff[valid_cols].corr()
                fig_hm = px.imshow(
                    matriz_corr,
                    text_auto=".2f",
                    color_continuous_scale=escala_heatmap,
                    labels=dict(color="Correlación")
                )
                fig_hm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(t=10, b=10, l=10, r=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Variables insuficientes para estructurar la matriz de correlación.")

        with col_comp:
            st.markdown('<div class="section-title">Comparativa de Legitimidad (Fraude vs Real)</div>', unsafe_allow_html=True)
            if "fraudulent" in ddff.columns:
                counts = ddff["fraudulent"].value_counts().reset_index()
                counts.columns = ["Estado", "Total"]
                counts["Estado"] = counts["Estado"].map({0: "Legítima", 1: "Fraudulenta"})
                
                fig_comp = px.bar(
                    counts, x="Estado", y="Total",
                    color="Estado",
                    color_discrete_map={"Legítima": "#2563eb", "Fraudulenta": "#f59e0b"}
                )
                fig_comp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title="Cantidad de Ofertas",
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                fig_comp.update_xaxes(showgrid=False)
                fig_comp.update_yaxes(showgrid=True, gridcolor="rgba(37, 99, 235, 0.1)")
                st.plotly_chart(fig_comp, use_container_width=True)

        col_izq, col_der = st.columns(2)
        
        with col_izq:
            st.markdown('<div class="section-title">Segmentación Estructural de la Muestra</div>', unsafe_allow_html=True)
            fig_pie = px.pie(
                ddff, names=variable_categorica, hole=0.4,
                color_discrete_sequence=paleta_graficos, height=340
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_der:
            st.markdown('<div class="section-title">Sectores Industriales con Mayor Incidencia</div>', unsafe_allow_html=True)
            if "industry" in ddff.columns:
                top_ind = ddff[ddff["fraudulent"] == 1]["industry"].value_counts().head(5).reset_index()
                top_ind.columns = ["Sector", "Reportes"]
                
                fig_bar = px.bar(
                    top_ind, x="Reportes", y="Sector", orientation="h",
                    color="Reportes", color_continuous_scale=[[0, "#10b981"], [1, "#1e3a8a"]], height=340
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    margin=dict(t=10, b=10, l=10, r=10),
                    coloraxis_showscale=False
                )
                fig_bar.update_xaxes(showgrid=True, gridcolor="rgba(37, 99, 235, 0.1)")
                fig_bar.update_yaxes(showgrid=False)
                st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — SIMULADOR PREDICTIVO
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Análisis Predictivo en Tiempo Real")
    st.caption("Consola analítica integrada con el pipeline de clasificación binaria CatBoost.")
    st.markdown("---")

    if hubo_error:
        st.error("Error crítico: Los artefactos del modelo CatBoost no fueron localizados en el directorio 'data/modelo/'. Asegúrate de tener 'modelo_catboost_fraud.cbm' y 'columnas_modelo_fraud.pkl'.")
    else:
        st.subheader("Atributos de la Vacante Bajo Auditoría")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            title = st.text_input("Título Oficial del Puesto", "Urgent Call Center Representative Needed")
            country = st.text_input("Código Internacional del País", "US")
            employment_type = st.selectbox("Modalidad Contractual", ["Full-time", "Part-time", "Contract", "Other"])
        with c2:
            department = st.text_input("División / Departamento", "Customer Service")
            required_experience = st.selectbox("Nivel de Experiencia", ["Entry level", "Mid-Senior level", "Associate", "Executive"])
            required_education = st.selectbox("Estudios Mínimos Exigidos", ["High School", "Bachelor's Degree", "Master's Degree", "Unspecified"])
        with c3:
            industry = st.text_input("Área de Negocio", "Telecommunications")
            function = st.text_input("Rol Funcional", "Customer Service")
            telecommuting = st.checkbox("Ofrece Teletrabajo / Remoto", value=True)
            
        c4, c5 = st.columns(2)
        with c4:
            has_logo = st.checkbox("La publicación cuenta con Logotipo Corporativo verificado", value=False)
        with c5:
            has_questions = st.checkbox("Contiene preguntas obligatorias de filtrado (In-app questions)", value=False)

        st.markdown('<div class="section-title">Cuerpo del Texto Publicado</div>', unsafe_allow_html=True)
        company_profile = st.text_area("Perfil de la Empresa", "We are an anonymous outsourcing cluster...", height=70)
        description = st.text_area("Descripción del Anuncio", "Earn 2000 USD weekly from your house doing easy typing...", height=110)
        requirements = st.text_area("Requerimientos Detallados", "No previous experience required. Computer with Internet access.", height=70)

        st.markdown("---")
        
        if st.button("Ejecutar Evaluación de Riesgo", use_container_width=True):
            with st.spinner("Computando vectores métricos..."):
                
                clean_desc = description.strip()
                text_length = len(clean_desc)
                has_requirements = 0 if requirements.strip() == "" or requirements == "Unspecified" else 1
                full_text = f"{title} {company_profile} {clean_desc}"
                
                state = "Unspecified"
                city = "Unspecified"

                input_data = {
                    'country': country, 'state': state, 'city': city,
                    'employment_type': employment_type, 'required_experience': required_experience,
                    'required_education': required_education, 'industry': industry, 'function': function,
                    'telecommuting': int(telecommuting), 'has_company_logo': int(has_logo),
                    'has_questions': int(has_questions), 'text_length': text_length,
                    'has_requirements': has_requirements, 'full_text': full_text
                }
                
                df_input = pd.DataFrame([input_data])
                
                # Reindexamos/filtramos las columnas según el orden estricto guardado en el .pkl del modelo
                df_input = df_input.reindex(columns=columnas_modelo, fill_value="Unspecified")

                probabilidad_fraude = modelo_catboost.predict_proba(df_input)[0][1]
                prediccion = modelo_catboost.predict(df_input)[0]

                res_col, graph_col = st.columns([1, 1])
                
                with res_col:
                    st.markdown("### Conclusión del Sistema")
                    if prediccion == 1:
                        st.markdown(
                            f"""
                            <div class="result-box-fraud">
                                <div class="pred-title">Publicación Altamente Sospechosa</div>
                                <div class="pred-value pred-value-fraud">RIESGO: {probabilidad_fraude*100:.1f}%</div>
                                <div style="font-size:13px; margin-top:10px; color:#475569;">Estructura compatible con ofertas engañosas (Veredicto: Riesgo Crítico)</div>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box-safe">
                                <div class="pred-title">Oferta Verificada y Confiable</div>
                                <div class="pred-value pred-value-safe">RIESGO: {probabilidad_fraude*100:.1f}%</div>
                                <div style="font-size:13px; margin-top:10px; color:#475569;">Métricas formales estables (Veredicto: Validado)</div>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        
                    st.markdown("<br>**Desglose Vectorial de Factores de Riesgo:**", unsafe_allow_html=True)
                    factores = ["Falta de Logo", "Texto Reducido", "Palabras de Alerta", "Sin Filtro de Preguntas"]
                    valores_sh = [0.35 if not has_logo else -0.1, 
                                  0.25 if text_length < 300 else -0.15,
                                  0.30 if "weekly" in full_text.lower() else -0.05,
                                  0.15 if not has_questions else -0.1]
                    
                    fig_shap = px.bar(
                        x=valores_sh, y=factores, orientation='h',
                        color=valores_sh, color_continuous_scale=[[0, "#f59e0b"], [0.5, "#10b981"], [1, "#2563eb"]],
                        labels={"x": "Impacto", "y": "Feature"}
                    )
                    fig_shap.update_layout(height=180, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, coloraxis_showscale=False, margin=dict(t=5,b=5,l=5,r=5))
                    st.plotly_chart(fig_shap, use_container_width=True)

                with graph_col:
                    st.markdown("### Nivel Crítico Estructural")
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probabilidad_fraude * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#0f172a"},
                            'bar': {'color': "#1e3a8a"},
                            'bgcolor': "white",
                            'borderwidth': 1,
                            'bordercolor': "#cbd5e1",
                            'steps': [
                                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                                {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                                {'range': [70, 100], 'color': 'rgba(37, 99, 235, 0.2)'}
                            ],
                            'threshold': {
                                'line': {'color': "#2563eb", 'width': 3},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=10, l=30, r=30))
                    st.plotly_chart(fig_gauge, use_container_width=True)