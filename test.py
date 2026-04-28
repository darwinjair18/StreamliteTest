import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1. Configuración inicial de la página
st.set_page_config(page_title="Forense 360 - PMV", layout="wide")

# 2. Inicializar la variable de sesión
# Si es la primera vez que entra, definimos que NO está autenticado
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

# 3. Datos ficticios de usuarios (Mock Data para el Login)
USUARIOS_MOCK = {
    "auditor": "forense2026",
    "admin": "admin123"
}

# --- FUNCIONES DE INTERFAZ ---

def mostrar_pantalla_login():
    """Renderiza el formulario de inicio de sesión."""
    # Usamos columnas para centrar el formulario de login
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Acceso a Forense 360")
        st.markdown("Por favor, ingresa tus credenciales corporativas.")
        
        # Formulario de Streamlit
        with st.form("login_form"):
            usuario = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Ingresar")
            
            if submit:
                # Lógica de validación
                if usuario in USUARIOS_MOCK and USUARIOS_MOCK[usuario] == password:
                    st.session_state['autenticado'] = True
                    st.session_state['usuario_actual'] = usuario
                    st.success("Acceso concedido. Redirigiendo...")
                    st.rerun() # Fuerza a Streamlit a recargar la página con el nuevo estado
                else:
                    st.error("Usuario o contraseña incorrectos.")

def generar_datos_ml():
    """Genera datos médicos con variables más complejas para el ML"""
    np.random.seed(42)
    n = 1500
    
    # Variables base
    df = pd.DataFrame({
        'ID_Reclamo': [f"REC-{i:04d}" for i in range(n)],
        'Especialidad': np.random.choice(['Traumatología', 'Cardiología', 'Medicina General', 'Odontología'], n),
        'Edad_Paciente': np.random.randint(18, 85, size=n),
        'Frecuencia_Mensual': np.random.poisson(lam=1.5, size=n), # Cuántas veces fue al médico este mes
        'Costo_Atencion': np.random.normal(loc=150, scale=40, size=n),
        'Fraude_Historico': 0 # Inicialmente todo es normal (Etiqueta para XGBoost)
    })
    
    # Inyectamos patrones de Fraude Lógico (Casos para que XGBoost y Isolation Forest detecten)
    # Patrón 1: Costos absurdamente altos en Medicina General
    idx_fraude_1 = np.random.choice(df.index, 30, replace=False)
    df.loc[idx_fraude_1, 'Costo_Atencion'] = df.loc[idx_fraude_1, 'Costo_Atencion'] * 7
    df.loc[idx_fraude_1, 'Especialidad'] = 'Medicina General'
    df.loc[idx_fraude_1, 'Fraude_Historico'] = 1
    
    # Patrón 2: Paciente va demasiadas veces al mes y cobra el máximo posible
    idx_fraude_2 = np.random.choice(df[df['Fraude_Historico'] == 0].index, 20, replace=False)
    df.loc[idx_fraude_2, 'Frecuencia_Mensual'] = np.random.randint(8, 15, size=20)
    df.loc[idx_fraude_2, 'Costo_Atencion'] = 400
    df.loc[idx_fraude_2, 'Fraude_Historico'] = 1

    return df

def mostrar_dashboard():
    # --- BARRA LATERAL ---
    with st.sidebar:
        st.write(f"👤 Bienvenido, **{st.session_state['usuario_actual']}**")
        if st.button("Cerrar Sesión"):
            st.session_state['autenticado'] = False
            st.rerun()

    st.title("Forense 360: Motor de Machine Learning")
    
    # 1. Cargar y preparar datos
    df = generar_datos_ml()
    
    # Preparar variables para ML (Codificamos variables categóricas)
    le = LabelEncoder()
    df['Especialidad_Cod'] = le.fit_transform(df['Especialidad'])
    
    features = ['Edad_Paciente', 'Frecuencia_Mensual', 'Costo_Atencion', 'Especialidad_Cod']
    X = df[features]
    y = df['Fraude_Historico']

    # --- ENTRENAMIENTO DE MODELOS ---
    with st.spinner("Entrenando modelos de detección..."):
        # Modelo 1: Isolation Forest (No Supervisado)
        # Contamination es el % estimado de fraude en la data (asumimos 3%)
        iso_forest = IsolationForest(contamination=0.03, random_state=42)
        df['Anomalia_IF'] = iso_forest.fit_predict(X)
        # Isolation Forest devuelve -1 para anomalías y 1 para normales. Lo pasamos a 1 (Fraude) y 0 (Normal)
        df['Alerta_Isolation'] = df['Anomalia_IF'].apply(lambda x: 1 if x == -1 else 0)

        # Modelo 2: XGBoost (Supervisado)
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X, y)
        # Predecimos la probabilidad de que sea fraude (0 a 100%)
        df['Probabilidad_Fraude_XGB'] = xgb_model.predict_proba(X)[:, 1]

    # --- INTERFAZ VISUAL ---
    tab1, tab2, tab3 = st.tabs(["📊 Resumen Global", "🌲 Isolation Forest (Anomalías)", "🚀 XGBoost (Predicción Supervisada)"])

    with tab1:
        st.subheader("Métricas de Auditoría")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reclamos Analizados", len(df))
        col2.metric("Alertas Generadas (Isolation)", df['Alerta_Isolation'].sum(), help="Casos estadísticamente atípicos")
        col3.metric("Casos Alto Riesgo (XGBoost > 80%)", len(df[df['Probabilidad_Fraude_XGB'] > 0.80]), delta_color="inverse")
        
        st.divider()
        st.dataframe(
            df[['ID_Reclamo', 'Especialidad', 'Costo_Atencion', 'Frecuencia_Mensual', 'Alerta_Isolation', 'Probabilidad_Fraude_XGB']].sort_values(by='Probabilidad_Fraude_XGB', ascending=False).head(50),
            use_container_width=True
        )

    with tab2:
        st.subheader("Detección de Valores Atípicos (Isolation Forest)")
        st.markdown("El modelo aísla automáticamente los reclamos que no siguen el patrón normal de facturación.")
        
        fig_iso = px.scatter(
            df, x="Costo_Atencion", y="Frecuencia_Mensual", color=df['Alerta_Isolation'].astype(str),
            color_discrete_map={"0": "blue", "1": "red"},
            hover_data=["ID_Reclamo", "Especialidad"],
            title="Mapa de Anomalías Aisladas (Rojo = Atípico)"
        )
        st.plotly_chart(fig_iso, use_container_width=True)

    with tab3:
        st.subheader("Predicción de Probabilidad de Fraude (XGBoost)")
        st.markdown("El modelo asigna un % de riesgo basado en el aprendizaje de fraudes históricos.")
        
        fig_xgb = px.histogram(
            df, x="Probabilidad_Fraude_XGB", nbins=50, 
            title="Distribución de Probabilidad de Riesgo",
            color_discrete_sequence=['purple']
        )
        fig_xgb.add_vline(x=0.8, line_dash="dash", line_color="red", annotation_text="Umbral de Auditoría (80%)")
        st.plotly_chart(fig_xgb, use_container_width=True)

# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---

# Esta es la lógica que decide qué mostrar basándose en la sesión
if not st.session_state['autenticado']:
    mostrar_pantalla_login()
else:
    mostrar_dashboard()