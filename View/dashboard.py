import os
import warnings
import joblib  # Para cargar el modelo de machine learning
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression  # creación de modeo mockeado
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuracion de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Postings Dashboard - Proyecto 7",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modelo mockeado (Mantener estructura original sin eliminar) ───────────────


def crear_recursos_mock():
    modelo_mock = LinearRegression()
    scaler_mock = StandardScaler()
    X_ficticio = np.array([[7, 1500, 1990, 2, 900, 2, 1]])
    y_ficticio = np.array([12.2])
    scaler_mock.fit(X_ficticio)
    scaler_mock.feature_names_in_ = [
        "OverallQual",
        "GrLivArea",
        "YearBuilt",
        "GarageCars",
        "TotalBsmtSF",
        "FullBath",
        "Fireplaces",
    ]
    modelo_mock.fit(scaler_mock.transform(X_ficticio), y_ficticio)
    modelo_mock.coef_ = np.zeros(258)
    modelo_mock.coef_[0] = 0.35
    modelo_mock.coef_[1] = 0.25
    columnas_mock = [
        "OverallQual",
        "GrLivArea",
        "YearBuilt",
        "GarageCars",
        "TotalBsmtSF",
        "FullBath",
        "Fireplaces",
    ] + [f"col_{i}" for i in range(251)]
    return modelo_mock, scaler_mock, columnas_mock, True


# ── Carga de datos (CAMBIO 1: Apuntando a tu CSV limpio) ───────────────────────


@st.cache_data
def load_data():
    # Usamos la ruta limpia libre de outliers extremos
    path = os.path.join("data", "fake_job_postings_clean.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # Fallback de emergencia por si ejecutas local en otra estructura
        st.error(f"No se encontró el archivo en {path}")
        return pd.DataFrame()


df = load_data()

# Crear métrica en tiempo de ejecución para la previsualización del texto
if not df.empty and "description" in df.columns:
    df["longitud_descripcion"] = df["description"].fillna("").str.len()


@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load(
            r"data/utiles/modelo/modelo_ridge_house_prices.pkl"
        )
        escalador = joblib.load(
            r"data/utiles/modelo/escalador_house_prices.pkl"
        )
        columnas = joblib.load(r"data/utiles/modelo/columnas_modelo.pkl")
        return modelo, escalador, columnas, False
    except Exception as e:
        return crear_recursos_mock()


modelo_ml, scaler, columnas_modelo, hubo_error = cargar_recursos()


# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #212529; }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #212529;
        margin: 1.5rem 0 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #f0f0f0;
    }
    .pred-result {
        background: linear-gradient(135deg, #424242 0%, #616161 100%);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-result .label { font-size: 14px; opacity: 0.85; margin-bottom: 6px; }
    .pred-result .value { font-size: 32px; font-weight: 700; letter-spacing: -1px; }
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div { gap: 8px; }
    div[data-testid="stRadio"] > div > label {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 10px 14px;
        width: 100%;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        border: 1.5px solid transparent;
    }
    div[data-testid="stRadio"] > div > label:hover {
        background: #e2e8f5;
        border-color: #6a0dad;
    }
    [data-testid="stSidebar"] { background: #fafafa; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💼 Proyecto 7")
    st.caption("Filtros de Análisis Extrayendo Outliers")
    st.markdown("---")

    pagina = st.radio(
        "Navegación", options=["📊  Explorador", "🔮  Predicción"], index=0
    )

    st.markdown("---")

    if pagina == "📊  Explorador":
        variable_categorica = st.selectbox(
            "🎨 Variable Temática (Color)",
            options=[
                "employment_type",
                "required_experience",
                "required_education",
                "has_company_logo",
            ],
            help="Segmentación categórica de ofertas laborales",
        )

        st.markdown("---")
        st.markdown("### Ajustes de Previsualización")

        # Filtros adaptados a los datos que tenemos
        logotipo = st.multiselect(
            "¿Tiene Logotipo de la Empresa?",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No",
        )

        longitud_maxima = int(df["longitud_descripcion"].max())
        rango_longitud = st.slider(
            "Extensión de la Oferta (Caracteres)",
            0,
            longitud_maxima,
            (0, longitud_maxima),
        )

    else:
        st.markdown("### Variables del Clasificador")
        st.caption(
            "Bloqueado temporalmente para la integración final del modelo."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — EXPLORADOR (CAMBIO 2: KPIs y Gráficos con Datos Reales)
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📊  Explorador" and not df.empty:

    # Aplicación de los filtros dinámicos basados en la limpieza previa
    ddff = df[
        df["has_company_logo"].isin(logotipo)
        & df["longitud_descripcion"].between(*rango_longitud)
    ]

    st.title("💼 Detección de Ofertas de Empleo Fraudulentas")
    st.caption("Previsualización analítica basada en el dataset procesado.")

    # ── KPIs Nuevos Ajustados al CSV Real ─────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    total_ofertas = len(ddff)
    total_fraudes = int(ddff["fraudulent"].sum()) if "fraudulent" in ddff.columns else 0
    porcentaje_fraude = (
        (total_fraudes / total_ofertas * 100) if total_ofertas > 0 else 0
    )
    longitud_promedio = (
        ddff["longitud_descripcion"].mean() if total_ofertas > 0 else 0
    )

    kpis = [
        (k1, "Registros Cargados", f"{total_ofertas:,}", "#1565c0"),
        (k2, "Ofertas Fraudulentas", f"{total_fraudes:,}", "#d32f2f"),
        (k3, "Tasa de Fraude Detectada", f"{porcentaje_fraude:.2f}%", "#d32f2f"),
        (k4, "Longitud Media de Texto", f"{longitud_promedio:.0f} carac.", "#6a0dad"),
    ]

    for col, label, val, color in kpis:
        col.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{val}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Gráfico 1: Análisis Distribución de la Longitud Limpia ────────────────
    st.markdown(
        '<div class="section-title">📊 Perfil de Distribución de Ofertas Limpias de Outliers</div>',
        unsafe_allow_html=True,
    )

    fig_hist = px.histogram(
        ddff,
        x="longitud_descripcion",
        color="fraudulent",
        nbins=40,
        title="Frecuencia de ofertas según extensión de la descripción (Separado por legitimidad)",
        color_discrete_map={0: "#6a0dad", 1: "#ffcc00"},  # Morado y Amarillo
        labels={
            "longitud_descripcion": "Número de Caracteres",
            "count": "Cantidad",
            "fraudulent": "Fraude",
        },
        barmode="overlay",
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Gráficos 2 y 3: Distribuciones Categóricas Combinadas ──────────────────
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("**Segmentación del Dataset por Variable Elegida**")
        fig_pie = px.pie(
            ddff,
            names=variable_categorica,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Agsunset,
            height=320,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_der:
        st.markdown("**Top 5 Industrias Afectadas por Fraude**")
        if "industry" in ddff.columns:
            top_industrias = (
                ddff[ddff["fraudulent"] == 1]["industry"]
                .value_counts()
                .head(5)
                .reset_index()
            )
            top_industrias.columns = ["Industria", "Reportes"]

            fig_bar = px.bar(
                top_industrias,
                x="Reportes",
                y="Industria",
                orientation="h",
                color="Reportes",
                color_continuous_scale="Purp",  # Escala morada consistente
                height=320,
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(t=20, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — PREDICCIÓN (CAMBIO 3: Desactivada temporalmente sin borrar lógica)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🔮 Módulo de Inferencia en Tiempo Real")
    st.caption("Simulador predictivo del pipeline técnico.")
    st.markdown("---")

    # Mensaje informativo elegante para que sepa tu equipo que está retenido ahí a propósito
    st.warning(
        "🔒 **Funcionalidad bloqueada temporalmente para la Entrega:** La lógica de predicción "
        "se encuentra inactiva mientras se completa el entrenamiento del modelo definitivo usando "
        "las nuevas variables estructuradas libres de outliers."
    )

    # El contenedor visual sigue ahí para ver cómo lucirá el proyecto en la presentación
    st.markdown(
        """
    <div class="pred-result">
        <div class="label">Simulador de Clasificación de Ofertas</div>
        <div class="value">MÓDULO EN MANTENIMIENTO</div>
        <div style="font-size:12px; margin-top:8px; opacity:0.8;">[La integración final con Scikit-Learn se desplegará a mediados de junio]</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # CÓDIGO INTERNO COMENTADO (NO SE ELIMINA, QUEDA SUSPENDIDO COMO SOLICITASTE)
    # ==========================================================================
    # datos_modelo = {col: 0 for col in columnas_modelo}
    # try:
    #     columnas_del_scaler = list(scaler.feature_names_in_)
    # except AttributeError:
    #     columnas_del_scaler = ['OverallQual', 'GrLivArea', 'YearBuilt']
    # ... El pipeline sigue intacto abajo para cuando decidas reactivarlo ...
    # ==========================================================================