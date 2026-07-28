import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import shap
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Control Hub Pro", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")
        return None

st.title("🏭 Centro de Control Metalúrgico: Inteligencia en Tiempo Real")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1️⃣ Gestión de Datos")
    archivo = st.file_uploader("Subir dataset (CSV o XLSX)", type=["csv", "xlsx"])
    modo_datos = st.radio("Filtro de Ruido:", ["Dataset Original", "Sin Outliers (IQR)"])
    
    st.header("2️⃣ Configuración del Modelo")
    tipo_modelo = st.selectbox("Algoritmo de IA:", ["XGBoost", "CatBoost"])
    n_estimators = st.slider("Número de Árboles:", 50, 500, 100, step=50)
    learning_rate = st.slider("Tasa de Aprendizaje (LR):", 0.01, 0.3, 0.05, step=0.01)

if archivo is not None:
    df_raw = cargar_datos(archivo)
    
    if df_raw is not None:
        # Selección de columnas numéricas y eliminación de nulos
        df_num = df_raw.select_dtypes(include=[np.number]).dropna()
        
        # Filtro de Outliers por IQR
        if modo_datos == "Sin Outliers (IQR)":
            Q1 = df_num.quantile(0.25)
            Q3 = df_num.quantile(0.75)
            IQR = Q3 - Q1
            df = df_num[~((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            df = df_num.copy()
            
        columnas = df.columns.tolist()
        
        with st.sidebar:
            st.header("3️⃣ Variables de Proceso")
            target = st.selectbox("Variable Objetivo (Y):", columnas, index=len(columnas)-1)
            features = st.multiselect("Variables Predictoras (X):", [c for c in columnas if c != target], default=[c for c in columnas if c != target])
            
        if features and target:
            X = df[features]
            y = df[target]
            
            # Inicialización del modelo según selección
            if tipo_modelo == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            else:
                model = CatBoostRegressor(iterations=n_estimators, learning_rate=learning_rate, random_state=42, verbose=0)
                
            # Validación Cruzada (K-Fold K=5)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=kf)
            
            # Entrenamiento global
            model.fit(X, y)
            
            # Cálculo de Métricas (Originales + MAPE)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            # --- PESTAÑAS PRINCIPALES ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Tendencias y Correlación", 
                "🎯 Rendimiento del Modelo", 
                "🎛️ Simulador Proactivo", 
                "🚨 Auditoría de Turnos",
                "🧠 Explicabilidad (SHAP)"
            ])
            
            # --- PESTAÑA 1: TENDENCIAS Y CORRELACIÓN ---
            with tab1:
                st.subheader("Análisis Exploratorio de Variables")
                col_scatter, col_corr = st.columns(2)
                
                with col_scatter:
                    var_x = st.selectbox("Variable para Scatter Plot vs " + target, features)
                    fig_disp = px.scatter(df, x=var_x, y=target, trendline="ols", 
                                          title=f"Relación entre {var_x} y {target}")
                    st.plotly_chart(fig_disp, use_container_width=True)
                    
                with col_corr:
                    st.subheader("Matriz de Correlación (Heatmap)")
                    corr_matrix = df[[target] + features].corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                                         title="Correlación entre Variables")
                    st.plotly_chart(fig_corr, use_container_width=True)

            # --- PESTAÑA 2: RENDIMIENTO DEL MODELO ---
            with tab2:
                st.subheader(f"Métricas de Desempeño ({tipo_modelo})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2:.3f}")
                m2.metric("MAE (Error Absoluto)", f"{mae:.3f}")
                m3.metric("RMSE (Riesgo)", f"{rmse:.3f}")
                m4.metric("MAPE (Error %)", f"{mape:.2f}%")
                
                col_real_pred, col_imp = st.columns(2)
                with col_real_pred:
                    fig_rp = px.scatter(x=y, y=y_pred, labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
                                        title="Real vs Predicho")
                    fig_rp.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                                    line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_rp, use_container_width=True)
                    
                with col_imp:
                    if tipo_modelo == "XGBoost":
                        importances = model.feature_importances_
                    else:
                        importances = model.get_feature_importance()
                        
                    df_imp = pd.DataFrame({'Variable': features, 'Importancia': importances}).sort_values('Importancia', ascending=True)
                    fig_imp = px.bar(df_imp, x='Importancia', y='Variable', orientation='h', 
                                     title="Importancia Relativa de Variables")
                    st.plotly_chart(fig_imp, use_container_width=True)

            # --- PESTAÑA 3: SIMULADOR PROACTIVO ---
            with tab3:
                st.subheader("Simulación Operativa en Tiempo Real")
                st.markdown("Ajuste los parámetros operativos para predecir el impacto en la variable objetivo:")
                
                inputs_sim = {}
                cols_sim = st.columns(3)
                for idx, col_name in enumerate(features):
                    min_val = float(df[col_name].min())
                    max_val = float(df[col_name].max())
                    mean_val = float(df[col_name].mean())
                    
                    with cols_sim[idx % 3]:
                        inputs_sim[col_name] = st.slider(f"{col_name}:", min_val, max_val, mean_val)
                        
                df_sim_input = pd.DataFrame([inputs_sim])
                pred_simulada = model.predict(df_sim_input)[0]
                
                st.divider()
                col_res_sim, col_gauge = st.columns([1, 2])
                
                with col_res_sim:
                    st.markdown("### Resultado de la Predicción")
                    st.metric(label=f"{target} Predicho", value=f"{pred_simulada:.2f}")
                    
                    media_target = y.mean()
                    if pred_simulada < (media_target - mae):
                        st.warning("⚠️ **Atención:** La predicción está por debajo del promedio histórico esperado.")
                    elif pred_simulada >= media_target:
                        st.success("✅ **Operación Óptima:** La predicción supera el promedio histórico.")

                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_simulada,
                        title={'text': f"Indicador: {target}"},
                        gauge={
                            'axis': {'range': [y.min(), y.max()]},
                            'bar': {'color': "navy"},
                            'steps': [
                                {'range': [y.min(), media_target - mae], 'color': "#FF4B4B"},
                                {'range': [media_target - mae, media_target + mae], 'color': "#FFA500"},
                                {'range': [media_target + mae, y.max()], 'color': "#00CC96"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': media_target
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)

            # --- PESTAÑA 4: AUDITORÍA DE TURNOS ---
            with tab4:
                st.subheader("Auditoría de Desviaciones por Turno / Muestra")
                
                df_audit = df.copy()
                df_audit['Predicho'] = y_pred
                df_audit['Error_Absoluto'] = np.abs(df_audit[target] - df_audit['Predicho'])
                
                def categorizar_error(err):
                    if err <= mae:
                        return "🟢 Normal"
                    elif err <= 2 * mae:
                        return "🟡 Advertencia"
                    else:
                        return "🔴 Anomalía"
                        
                df_audit['Estado_Semáforo'] = df_audit['Error_Absoluto'].apply(categorizar_error)
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    filtro_estado = st.multiselect("Filtrar por Estado de Auditoría:", 
                                                   ["🟢 Normal", "🟡 Advertencia", "🔴 Anomalía"],
                                                   default=["🟢 Normal", "🟡 Advertencia", "🔴 Anomalía"])
                
                df_audit_filtrado = df_audit[df_audit['Estado_Semáforo'].isin(filtro_estado)]
                
                st.dataframe(df_audit_filtrado[[target, 'Predicho', 'Error_Absoluto', 'Estado_Semáforo'] + features], 
                             use_container_width=True)
                
                fig_err = px.scatter(df_audit, x=df_audit.index, y='Error_Absoluto', color='Estado_Semáforo',
                                     color_discrete_map={"🟢 Normal": "green", "🟡 Advertencia": "orange", "🔴 Anomalía": "red"},
                                     title="Límites de Control de Error (MAE)")
                fig_err.add_hline(y=mae, line_dash="dash", line_color="orange", annotation_text="1x MAE")
                fig_err.add_hline(y=2*mae, line_dash="dash", line_color="red", annotation_text="2x MAE")
                st.plotly_chart(fig_err, use_container_width=True)

            # --- PESTAÑA 5: EXPLICABILIDAD SHAP ---
            with tab5:
                st.subheader("Análisis de Explicabilidad con SHAP")
                st.markdown("Muestra la contribución e impacto directo de cada variable en el comportamiento global del modelo:")
                try:
                    explainer = shap.Explainer(model, X)
                    shap_values = explainer(X)
                    
                    fig_shap, ax = plt.subplots(figsize=(10, 5))
                    shap.summary_plot(shap_values, X, show=False)
                    st.pyplot(fig_shap)
                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico SHAP completo: {e}")

else:
    st.info("👈 Por favor, suba un archivo CSV o XLSX en la barra lateral para comenzar.")
