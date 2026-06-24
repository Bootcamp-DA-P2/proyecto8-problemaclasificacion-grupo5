# 🕵️ Fake Job Postings — Detección de Ofertas de Empleo Fraudulentas

> Proyecto grupal de Machine Learning | Bootcamp de Data Science  
> **Problema:** Clasificación binaria · **Target:** `fraudulent` (0 = real, 1 = fraude)

---

## 📋 Descripción

Este proyecto aborda la detección automática de ofertas de empleo fraudulentas publicadas en plataformas de reclutamiento. A partir del dataset [**Fake Job Postings**](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), se desarrolla un pipeline completo de Machine Learning que va desde el análisis exploratorio hasta el despliegue de una aplicación interactiva capaz de clasificar una oferta en tiempo real.

El dataset presenta un **desbalanceo severo** (~95% reales / ~5% fraudulentas), lo que convierte este proyecto en un caso real de aprendizaje en condiciones adversas, donde métricas como `Recall` y `F1-Score` son más relevantes que la `Accuracy`.

---

## 🗂️ Estructura del Proyecto

```
├── data/
│   ├── fake_job_postings.csv          # Dataset original
│   └── fake_job_postings_clean.csv    # Dataset tras limpieza de outliers
├── main.ipynb                         # Notebook principal (EDA + Modelado)
├── catboost_info/
│   ├── ...                            # Entrenamiento del modelo
├── View/
│   └── dashboard.py                   # Visualización del dashboard
│
└── README.md
```

---

## 🔍 Fases del Proyecto

### 1. Preprocesamiento y Limpieza

- Eliminación de columnas sin valor predictivo (`job_id`)
- Imputación de nulos en variables categóricas y de texto con `'Unspecified'`
- **Limpieza de texto** de la columna `description`: lowercase, eliminación de caracteres especiales y espacios dobles
- Extracción de `country`, `state` y `city` desde la columna `location`
- **Detección y eliminación de outliers** en la longitud de las descripciones mediante el método IQR (límite inferior mínimo de 10 caracteres)

### 2. Ingeniería de Features

| Feature creada | Descripción |
|---|---|
| `description_clean` | Texto de la descripción normalizado |
| `description_length` | Número de caracteres de la descripción limpia |
| `text_length` | Longitud de la descripción para el modelo |
| `has_requirements` | Binario: si la oferta incluye requisitos o no |
| `has_salary` | Binario: si se especifica rango salarial |
| `has_company_profile` | Binario: si hay perfil de empresa |
| `full_text` | Concatenación de `title` + `company_profile` + `description_clean` para NLP |

### 3. Análisis Exploratorio (EDA)

**Univariado:**
- Distribución del target `fraudulent`: desbalanceo ~95/5%
- Distribución de `required_experience` e `industry`
- Histograma de `description_length`: distribución asimétrica con pico en 0

**Bivariado:**
- Heatmap de correlación con `fraudulent` (variables: `telecommuting`, `has_company_logo`, `has_questions`, `description_length`, `has_salary`, `has_company_profile`)
- Proporción de fraude según presencia de logo y preguntas de filtrado
- Top 10 sectores con mayor tasa de fraude

### 4. Modelado

**Algoritmo:** `CatBoostClassifier`

CatBoost fue seleccionado por su capacidad nativa para manejar:
- Variables categóricas de alta cardinalidad (`country`, `industry`, `function`…) sin necesidad de encoding manual
- Variables de texto (`full_text`) directamente como `text_features`

**Configuración para controlar el overfitting (<5%):**

```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.015,       # Ritmo lento y controlado
    depth=3,                   # Árboles simples
    l2_leaf_reg=25,            # Regularización L2 fuerte
    subsample=0.6,
    scale_pos_weight=ratio_suave,  # Compensación del desbalanceo (√(mayoritaria/minoritaria))
    early_stopping_rounds=40
)
```

**Split:** 80/20 estratificado sobre `fraudulent`

### 5. Evaluación

Las métricas prioritarias dado el desbalanceo son **Recall** y **F1-Score** sobre la clase positiva (fraude).

| Métrica | Descripción |
|---|---|
| Confusion Matrix | Visualización de FP, FN, TP, TN |
| Classification Report | Precision, Recall, F1 por clase |
| ROC-AUC | Capacidad discriminativa global del modelo |
| Overfitting % | `((test_loss - train_loss) / test_loss) × 100` → objetivo: < 5% |

---

## 🛠️ Tecnologías

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.x |
| Manipulación de datos | Pandas, NumPy, SciPy |
| Machine Learning | Scikit-learn, CatBoost, XGBoost |
| Visualización | Matplotlib, Seaborn, Plotly |
| Persistencia | Joblib |
| Productivización | Streamlit / Gradio / Dash |
| Control de versiones | Git + GitHub |
| Contenedores | Docker |

---

## 🚀 Instalación y Uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-DA-P2/proyecto8-problemaclasificacion-grupo5.git


# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el notebook de análisis
jupyter notebook main.ipynb

# 4. Lanzar la aplicación
streamlit run view/dashboard.py
```

---

## 📊 Resultados Principales

- El modelo logra mantener el overfitting **por debajo del 5%** gracias a la regularización L2 fuerte, el learning rate bajo y el `early_stopping`
- La variable más discriminante es `has_company_logo`: las ofertas **sin logo** tienen una tasa de fraude significativamente mayor
- Los sectores con mayor concentración de fraude corresponden a industrias de alta rotación y baja barrera de entrada
- Las descripciones fraudulentas tienden a ser más cortas y con mayor proporción de mayúsculas

---

## 👥 Equipo

Proyecto desarrollado en grupo con Ana Paula Montiel, Felix Gonzalez y Ana Ganfornina.

---

## 📄 Licencia

Este proyecto tiene fines educativos. El dataset original es de uso público en [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).