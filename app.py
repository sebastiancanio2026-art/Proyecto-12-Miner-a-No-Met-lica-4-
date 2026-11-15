import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="Dashboard Minería No Metálica", page_icon="⛏️", layout="wide")

@st.cache_data
def load_data():
    path = "datos/procesados/dataset_limpio.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        productos = ['CLORURO DE SODIO', 'CARBONATO DE CALCIO', 'CALIZA', 'COMPUESTOS DE POTASIO', 'RECURSOS SILÍCEOS']
        anos = list(range(1996, 2026))
        data = [{'PRODUCTO': p, 'AÑO': a, 'PRODUCCION_TM': np.random.randint(500000, 3000000)} for p in productos for a in anos]
        return pd.DataFrame(data)

df = load_data()

st.title("⛏️ Dashboard Interactivo: Minería No Metálica")
st.markdown("Herramienta de análisis histórico y proyección de escenarios futuros para la toma de decisiones.")

# --- 1. FILTROS INTERACTIVOS ---
st.sidebar.header("🔍 Filtros de Análisis")

productos = sorted(df['PRODUCTO'].dropna().unique().tolist())
producto_seleccionado = st.sidebar.selectbox("Seleccione Producto:", options=productos)

min_anio = int(df['AÑO'].min())
max_anio = int(df['AÑO'].max())
selected_anos = st.sidebar.slider("Rango de Años (Histórico):", min_value=min_anio, max_value=max_anio, value=(min_anio, max_anio))

modelo_elegido = st.sidebar.selectbox("Modelo de Proyección:", ["Regresión Lineal", "Promedio Móvil"])
anios_proyeccion = st.sidebar.slider("Años a Proyectar:", min_value=1, max_value=10, value=5)

# Filtrar datos
df_filtrado = df[
    (df['PRODUCTO'] == producto_seleccionado) & 
    (df['AÑO'] >= selected_anos[0]) & 
    (df['AÑO'] <= selected_anos[1])
]

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    # --- 2. INDICADORES DINÁMICOS ---
    st.markdown("### 📊 Indicadores Automáticos Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    total_prod = df_filtrado['PRODUCCION_TM'].sum()
    prom_prod = df_filtrado['PRODUCCION_TM'].mean()
    max_prod_val = df_filtrado['PRODUCCION_TM'].max()
    num_registros = len(df_filtrado)
    
    col1.metric("Producción Total Acumulada", f"{total_prod:,.0f} TM")
    col2.metric("Producción Promedio Anual", f"{prom_prod:,.0f} TM")
    col3.metric("Máximo Registrado", f"{max_prod_val:,.0f} TM")
    col4.metric("Registros Analizados", f"{num_registros:,}")
    
    st.divider()

    # --- 3. MODELO Y PROYECCIÓN ---
    X_hist = df_filtrado[['AÑO']].values
    y_hist = df_filtrado['PRODUCCION_TM'].values
    ultimo_anio = int(df_filtrado['AÑO'].max())
    futuros_anios = np.array([ultimo_anio + i for i in range(1, anios_proyeccion + 1)]).reshape(-1, 1)

    if len(X_hist) > 1:
        if modelo_elegido == "Regresión Lineal":
            modelo = LinearRegression()
            modelo.fit(X_hist, y_hist)
            y_pred = modelo.predict(futuros_anios)
        else:
            window = min(3, len(y_hist))
            media_movil_val = pd.Series(y_hist).rolling(window=window).mean().iloc[-1]
            y_pred = np.array([media_movil_val for _ in futuros_anios])

        # Gráfico con Matplotlib
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df_filtrado['AÑO'], df_filtrado['PRODUCCION_TM'], marker='o', color='#0e76a8', linewidth=2.5, label='Histórico Real')

        anios_plot_futuro = np.concatenate(([ultimo_anio], futuros_anios.flatten()))
        y_plot_futuro = np.concatenate(([y_hist[-1]], y_pred))

        ax.plot(anios_plot_futuro, y_plot_futuro, marker='s', linestyle='--', color='#f59e0b', linewidth=2.5, label=f'Proyección ({modelo_elegido})')
        ax.set_title(f'Evolución y Comportamiento Futuro - {producto_seleccionado}', fontsize=13, fontweight='bold', color='#1e293b')
        ax.set_xlabel('Año', fontsize=10)
        ax.set_ylabel('Producción (Toneladas Métricas)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Seleccione un rango histórico mayor (mínimo 2 años) para generar la proyección matemática.")

    st.markdown("---")
    st.markdown("**Proyecto Final Integrador – Programación Avanzada** | *Minería No Metálica*")