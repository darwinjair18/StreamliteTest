import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

def mostrar_dashboard():
    """Renderiza el panel principal de la aplicación (solo si está logueado)."""
    
    # --- BARRA LATERAL ---
    with st.sidebar:
        st.write(f"👤 Bienvenido, **{st.session_state['usuario_actual']}**")
        if st.button("Cerrar Sesión"):
            st.session_state['autenticado'] = False
            st.rerun()

    # --- CONTENIDO PRINCIPAL (El código que armamos antes) ---
    st.title("Forense 360: Detección de Anomalías Médicas")
    st.markdown("Panel de control para auditoría de facturación.")

    # Generación de datos mock
    np.random.seed(42)
    num_registros = 1000
    df = pd.DataFrame({
        'ID_Reclamo': [f"REC-{i:04d}" for i in range(num_registros)],
        'Especialidad': np.random.choice(['Traumatología', 'Cardiología', 'Medicina General', 'Odontología'], num_registros),
        'Costo_Atencion': np.random.normal(loc=150, scale=50, size=num_registros),
        'Edad_Paciente': np.random.randint(18, 85, size=num_registros)
    })
    df.loc[10:15, 'Costo_Atencion'] = df['Costo_Atencion'] * 8

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reclamos", num_registros)
    col2.metric("Costo Promedio", f"${df['Costo_Atencion'].mean():.2f}")
    col3.metric("Reclamos Sospechosos (> $500)", len(df[df['Costo_Atencion'] > 500]), delta_color="inverse")

    st.divider()

    # Gráfico
    st.subheader("Distribución de Costos por Especialidad")
    fig = px.scatter(
        df, x="Edad_Paciente", y="Costo_Atencion", color="Especialidad",
        hover_data=["ID_Reclamo"], title="Identificación de Outliers"
    )
    fig.add_hline(y=500, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---

# Esta es la lógica que decide qué mostrar basándose en la sesión
if not st.session_state['autenticado']:
    mostrar_pantalla_login()
else:
    mostrar_dashboard()