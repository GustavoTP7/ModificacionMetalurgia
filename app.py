import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import shap
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Importación defensiva de SMOTE para evitar fallos si no está instalado
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Geomet Twin Pro: Integrated Mine-to-Mill DSS", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
        df = df.loc[:, df.columns != '']
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        st.error(f"Error en la ingesta de datos: {e}")
        return None

def generar_features_mina(df):
    df_c = df.copy()
    col_dict = {c.lower(): c for c in df_c.columns}
    
    cus = df_c[col_dict['cus']] if 'cus' in col_dict else (df_c[col_dict['cusac']] if 'cusac' in col_dict else (df_c[col_dict['cusac_pct']] if 'cusac_pct' in col_dict else 0))
    cucn = df_c[col_dict['cucn']] if 'cucn' in col_dict else 0
    cut = df_c[col_dict['cut']] if 'cut' in col_dict else (df_c[col_dict['cu_t']] if 'cu_t' in col_dict else (df_c[col_dict['cu_pct_alim']] if 'cu_pct_alim' in col_dict else 1))
    fet = df_c[col_dict['fe']] if 'fe' in col_dict else (df_c[col_dict['fe_t']] if 'fe_t' in col_dict else (df_c[col_dict['fe_pct_alim']] if 'fe_pct_alim' in col_dict else 1))
    
    if 'factor_k_calc' not in col_dict and 'rsol' not in col_dict and 'r_sol_pct' not in col_dict:
        if isinstance(cus, pd.Series) or isinstance(cucn, pd.Series):
            df_c['Factor_K_calc'] = ((cus + cucn) / (cut + 1e-6)) * 100
    if 'razon_fe_cu_calc' not in col_dict:
        if isinstance(fet, pd.Series) and isinstance(cut, pd.Series):
            df_c['Razon_Fe_Cu_calc'] = fet / (cut + 1e-6)
        
    cao = df_c[col_dict['cao_18']] if 'cao_18' in col_dict else 0
    mont = df_c[col_dict['mont_18']] if 'mont_18' in col_dict else 0
    filo = df_c[col_dict['filo_18']] if 'filo_18' in col_dict else 0
    if (isinstance(cao, pd.Series) or isinstance(mont, pd.Series) or isinstance(filo, pd.Series)) and 'arcillas_totales_calc' not in col_dict:
        df_c['Arcillas_Totales_calc'] = cao + mont + filo
            
    cpy = df_c[col_dict['cpy']] if 'cpy' in col_dict else 0
    cc = df_c[col_dict['cc']] if 'cc' in col_dict else 0
    cv = df_c[col_dict['cv']] if 'cv' in col_dict else 0
    bn = df_c[col_dict['bn']] if 'bn' in col_dict else 0
    if (isinstance(cpy, pd.Series) or isinstance(cc, pd.Series)) and 'sulfuros_cu_calc' not in col_dict:
        df_c['Sulfuros_Cu_calc'] = cpy + cc + cv + bn
    return df_c

def generar_features_planta(df):
    df_c = df.copy()
    col_dict = {c.lower(): c for c in df_c.columns}
    
    cut = df_c[col_dict['cu_pct_alim']] if 'cu_pct_alim' in col_dict else (df_c[col_dict['cut']] if 'cut' in col_dict else 1)
    fet = df_c[col_dict['fe_pct_alim']] if 'fe_pct_alim' in col_dict else (df_c[col_dict['fe']] if 'fe' in col_dict else 1)
    if 'razon_fe_cu_calc' not in col_dict:
        if isinstance(fet, pd.Series) and isinstance(cut, pd.Series):
            df_c['Razon_Fe_Cu_calc'] = fet / (cut + 1e-6)
        
    chalco = df_c[col_dict['chalcopyrite_pct']] if 'chalcopyrite_pct' in col_dict else 0
    sec = df_c[col_dict['min_sec_pct']] if 'min_sec_pct' in col_dict else 0
    pirita = df_c[col_dict['pirita_pct']] if 'pirita_pct' in col_dict else 0
    
    if isinstance(chalco, pd.Series) or isinstance(sec, pd.Series):
        if 'sulfuros_cu_calc' not in col_dict:
            df_c['Sulfuros_Cu_calc'] = chalco + sec
        if 'razon_sulf_pirita_calc' not in col_dict and isinstance(pirita, pd.Series):
            df_c['Razon_Sulf_Pirita_calc'] = (chalco + sec) / (pirita + 1e-3)
            
    tph = df_c[col_dict['tonelaje flotación']] if 'tonelaje flotación' in col_dict else (df_c[col_dict['plttph']] if 'plttph' in col_dict else 0)
    p80 = df_c[col_dict['p80']] if 'p80' in col_dict else 0
    if isinstance(tph, pd.Series) and isinstance(p80, pd.Series) and 'interaccion_tph_p80_calc' not in col_dict:
        df_c['Interaccion_TPH_P80_calc'] = tph * p80
    return df_c

# --- ENCABEZADO ---
st.title("💎 Geomet Twin Pro: Gemelo Digital Integrado Mina-Planta")
st.markdown("""
**Sistema de Soporte a la Decisión (DSS) y Conciliación Geometalúrgica**.
Integración Feed-Forward (Bloques de Mina) & Feedback (Operación de Planta SCADA), aislamiento de causas raíz FDI y simulación de *Blending*.
""")

# --- BARRA LATERAL (CARGA DUAL DE ARCHIVOS) ---
with st.sidebar:
    st.header("⚙️ 1. Ingesta Dual de Datos")
    st.markdown("Cargue las dos fuentes de datos para habilitar la conciliación completa Mina-Planta:")
    
    file_mina = st.file_uploader("📂 Data 1: Mina / Bloques (MineStart)", type=["csv", "xlsx"])
    file_planta = st.file_uploader("📂 Data 2: Planta / Turnos (SCADA)", type=["csv", "xlsx"])
    
    st.header("🧹 2. Tratamiento de Datos")
    modo_ruido = st.radio("Filtro de Outliers:", ["Data Original", "Depuración por IQR", "Isolation Forest (Multivariado)"])
    auto_fe = st.checkbox("Generar Ingeniería de Variables Autónomas (Ratios Fe/Cu, Arcillas, Factor K)", value=True)
    
    st.header("🤖 3. Motor de IA Autónomo")
    tipo_modelo = st.selectbox("Algoritmo de Aprendizaje:", ["XGBoost", "CatBoost"])
    estrategia_model = st.radio("Estrategia de Entrenamiento:", [
        "Modelos Especializados por UGM (Recomendado)",
        "Modelo Global Único"
    ])
    transformar_log = st.checkbox("Aplicar Transformación Logarítmica a distribuciones sesgadas")
    
    if HAS_SMOTE:
        balancear = st.checkbox("Balanceo SMOTE (Casos Críticos)")
    else:
        balancear = False
        st.caption("⚠️ SMOTE desactivado (requiere imbalanced-learn)")

    st.divider()
    ejecutar = st.button("🚀 Iniciar Gemelo Digital Integrado", use_container_width=True, type="primary")

# --- PROCESAMIENTO PRINCIPAL ---
df_m_raw = cargar_datos(file_mina) if file_mina else None
df_p_raw = cargar_datos(file_planta) if file_planta else None

if df_m_raw is not None or df_p_raw is not None:
    if df_m_raw is not None and df_p_raw is not None:
        st.success("✅ **Ambas fuentes de datos cargadas exitosamente (Mina + Planta). Conciliación Feed-Forward / Feedback Habilitada.**")
    elif df_m_raw is not None:
        st.info("ℹ️ **Cargada Data 1 (Mina / MineStart). Habilitado modo de Planificación Geometalúrgica.**")
    else:
        st.info("ℹ️ **Cargada Data 2 (Planta / SCADA Turnos). Habilitado modo de Control Operacional FDI.**")
        
    if ejecutar or ('res_mina' in st.session_state or 'res_planta' in st.session_state):
        if ejecutar:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def entrenar_pipeline(df_raw, tag_nombre, es_mina=True):
                df_proc = df_raw.copy()
                if auto_fe:
                    df_proc = generar_features_mina(df_proc) if es_mina else generar_features_planta(df_proc)
                    
                id_col = None
                for col in df_proc.columns:
                    c_low = str(col).lower()
                    if any(kw in c_low for kw in ['mining block', 'id_turno', 'id', 'fecha', 'turno', 'sample', 'day', 'block']):
                        id_col = col
                        break
                if id_col is None:
                    id_col = df_proc.columns[0]
                    
                col_nums = df_proc.select_dtypes(include=[np.number]).columns.tolist()
                if id_col in col_nums:
                    col_nums.remove(id_col)
                    
                target = col_nums[-1]
                for i, c in enumerate(col_nums):
                    if c.lower() in ['reccu', 'rec ro', 'rec_cu', 'recuperacion', 'recuperación', 'rec_ro']:
                        target = c
                        break
                        
                features = [c for c in col_nums if c != target and not any(kw in c.lower() for kw in ['recag', 'recmo', 'id', 'index'])]
                
                df_num = df_proc[col_nums].dropna().reset_index(drop=True)
                id_series = df_proc.loc[df_num.index, id_col].astype(str).values if id_col else np.array([f"Row_{i+1}" for i in range(len(df_num))])
                
                df_clean = df_num.copy()
                mask = np.ones(len(df_clean), dtype=bool)
                if modo_ruido == "Depuración por IQR":
                    Q1, Q3 = df_clean.quantile(0.25), df_clean.quantile(0.75)
                    IQR = Q3 - Q1
                    mask = ~((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)
                elif modo_ruido == "Isolation Forest (Multivariado)":
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    mask = iso.fit_predict(df_clean[features + [target]]) == 1
                    
                df_clean = df_clean[mask].reset_index(drop=True)
                ids = id_series[mask]
                
                best_k, best_score = 2, -1
                for k in range(2, 6):
                    if len(df_clean) > k:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(df_clean[features + [target]])
                        score = silhouette_score(df_clean[features + [target]], labels)
                        if score > best_score: best_score, best_k = score, k
                km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                df_clean['Dominio_GMD'] = km_final.fit_predict(df_clean[features + [target]])
                
                X = df_clean[features].copy()
                y = df_clean[target].values
                dominios = df_clean['Dominio_GMD'].values
                
                if transformar_log:
                    for c in features:
                        if df_clean[c].min() >= 0 and abs(df_clean[c].skew()) > 1.0:
                            X[c] = np.log1p(X[c])
                            
                if balancear and HAS_SMOTE:
                    y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                    sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                    X_w = X.copy()
                    X_w['__t__'] = y
                    X_w['__d__'] = dominios
                    X_res, _ = sm.fit_resample(X_w, y_disc)
                    y = X_res['__t__'].values
                    dominios = np.round(X_res['__d__'].values).astype(int)
                    X = X_res[features]
                    n_sint = len(X_res) - len(ids)
                    if n_sint > 0:
                        ids = np.concatenate([ids, [f"SMOTE_{i+1}" for i in range(n_sint)]])
                        
                X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
                if tipo_modelo == "XGBoost":
                    m_glob = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
                    m_glob.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    m_glob = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
                    m_glob.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                    
                sub_models = {}
                y_pred_cv = np.zeros_like(y)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                if estrategia_model == "Modelos Especializados por UGM (Recomendado)":
                    for dom in np.unique(dominios):
                        idx_d = np.where(dominios == dom)[0]
                        X_d = X.iloc[idx_d] if isinstance(X, pd.DataFrame) else X[idx_d]
                        y_d = y[idx_d]
                        
                        if len(y_d) >= 25:
                            if tipo_modelo == "XGBoost":
                                m_u = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42)
                            else:
                                m_u = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_state=42, verbose=0)
                            m_u.fit(X_d, y_d, verbose=False if tipo_modelo == "XGBoost" else 0)
                            sub_models[dom] = m_u
                            
                            kf_u = KFold(n_splits=min(5, len(y_d)), shuffle=True, random_state=42)
                            for tr_i, va_i in kf_u.split(X_d):
                                if tipo_modelo == "XGBoost":
                                    m_tmp = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
                                else:
                                    m_tmp = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, random_state=42, verbose=0)
                                m_tmp.fit(X_d.iloc[tr_i], y_d[tr_i], verbose=False if tipo_modelo == "XGBoost" else 0)
                                y_pred_cv[idx_d[va_i]] = m_tmp.predict(X_d.iloc[va_i])
                        else:
                            sub_models[dom] = m_glob
                            y_pred_cv[idx_d] = m_glob.predict(X_d)
                else:
                    y_pred_cv = cross_val_predict(m_glob, X, y, cv=kf)
                    for dom in np.unique(dominios):
                        sub_models[dom] = m_glob
                        
                r2 = r2_score(y, y_pred_cv)
                mae = mean_absolute_error(y, y_pred_cv)
                rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
                mape = mean_absolute_percentage_error(y, y_pred_cv) * 100
                
                return {
                    'm_glob': m_glob, 'sub_models': sub_models, 'km_final': km_final,
                    'df_clean': df_clean, 'features': features, 'target': target,
                    'X': X, 'y': y, 'y_pred': y_pred_cv, 'dominios': dominios,
                    'ids': ids, 'id_col': id_col, 'metrics': (r2, mae, rmse, mape)
                }

            if df_m_raw is not None:
                status_text.text("Entrenando Modelo 1: Planificación Geometalúrgica (Mina / MineStart)...")
                st.session_state.res_mina = entrenar_pipeline(df_m_raw, "Mina", es_mina=True)
                progress_bar.progress(50)
                
            if df_p_raw is not None:
                status_text.text("Entrenando Modelo 2: Control Operacional & SCADA (Planta / Turnos)...")
                st.session_state.res_planta = entrenar_pipeline(df_p_raw, "Planta", es_mina=False)
                progress_bar.progress(100)
                
            status_text.empty(); progress_bar.empty()

        # --- RENDERIZADO DE PESTAÑAS INTEGRADAS ---
        res_m = st.session_state.get('res_mina', None)
        res_p = st.session_state.get('res_planta', None)

        tab_names = ["📈 Fidelidad Mina / Bloques", "🏭 Control Planta / SCADA", "🤝 Conciliación Mina-Planta & FDI", "🎛️ Blending & Optimización", "🧠 IA Explicable (XAI)"]
        t1, t2, t3, t4, t5 = st.tabs(tab_names)

        with t1:
            if res_m:
                st.subheader("🏗️ Modelo 1: Evaluación Geometalúrgica de Bloques de Minado (MineStart)")
                r2, mae, rmse, mape = res_m['metrics']
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad Mina (R²)", f"{r2:.3f}")
                m2.metric("Error MAE", f"{mae:.3f}")
                m3.metric("Riesgo RMSE", f"{rmse:.3f}")
                m4.metric("Error MAPE", f"{mape:.2f}%")
                
                st.divider()
                st.dataframe(res_m['df_clean'].groupby('Dominio_GMD')[res_m['features'] + [res_m['target']]].mean().style.background_gradient(cmap='viridis'), use_container_width=True)
                st.plotly_chart(px.scatter(x=res_m['y'], y=res_m['y_pred'], color=[f"UGM {d}" for d in res_m['dominios']], 
                                           labels={'x': 'Recuperación Esperada Bloque (%)', 'y': 'Recuperación Estimada IA (%)'}, 
                                           title="Predicción de Bloques de Minado por UGM", trendline="ols"), use_container_width=True)
            else:
                st.info("👈 Cargue la Data 1 (Mina / MineStart) en la barra lateral para ver los resultados de bloques.")

        with t2:
            if res_p:
                st.subheader("🏭 Modelo 2: Control Operacional y Balance por Turnos de Planta (SCADA)")
                r2, mae, rmse, mape = res_p['metrics']
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad Planta (R²)", f"{r2:.3f}")
                m2.metric("Error MAE", f"{mae:.3f}")
                m3.metric("Riesgo RMSE", f"{rmse:.3f}")
                m4.metric("Error MAPE", f"{mape:.2f}%")
                
                st.divider()
                st.dataframe(res_p['df_clean'].groupby('Dominio_GMD')[res_p['features'] + [res_p['target']]].mean().style.background_gradient(cmap='magma'), use_container_width=True)
                st.plotly_chart(px.scatter(x=res_p['y'], y=res_p['y_pred'], color=[f"UGM {d}" for d in res_p['dominios']], 
                                           labels={'x': 'Recuperación Real Planta (%)', 'y': 'Recuperación Digital (%)'}, 
                                           title="Desempeño Operacional por Turno de Planta", trendline="ols"), use_container_width=True)
            else:
                st.info("👈 Cargue la Data 2 (Planta / Turnos) en la barra lateral para ver el control operacional.")

        with t3:
            st.subheader("🤝 Módulo de Conciliación Mina-Planta y Aislamiento de Causa Raíz FDI")
            st.markdown("""
            Este módulo compara el **Potencial Geometalúrgico Teórico del Bloque de Minado** (Feed-Forward) con la **Recuperación Real Obtenida en Planta** (Feedback) para aislar si una pérdida responde a la roca o a una falla en las celdas.
            """)
            
            if res_m and res_p:
                st.success("⚡ **Conciliación Automatizada Mina-Planta Activada.**")
                
                n_rows = min(len(res_p['ids']), len(res_m['ids']))
                df_conc = pd.DataFrame({
                    'ID / Turno / Fecha': res_p['ids'][:n_rows],
                    'Potencial Bloque Mina (%)': np.round(res_m['y_pred'][:n_rows], 2),
                    'Recuperación Real Planta (%)': np.round(res_p['y'][:n_rows], 2),
                })
                df_conc['Brecha (Mina - Planta)'] = np.round(df_conc['Potencial Bloque Mina (%)'] - df_conc['Recuperación Real Planta (%)'], 2)
                
                mae_p = res_p['metrics'][1]
                def diagnosticar(row):
                    brecha = row['Brecha (Mina - Planta)']
                    if abs(brecha) <= mae_p:
                        return "🟢 Normal (Planta dentro del potencial de la roca)"
                    elif brecha > mae_p:
                        return "🔴 Anomalía Operativa (Falla de reactivos, aireación o celdas en Planta)"
                    else:
                        return "🟡 Rendimiento Sobre-Esperado"
                        
                df_conc['Diagnóstico FDI / Causa Raíz'] = df_conc.apply(diagnosticar, axis=1)
                
                st.dataframe(df_conc.head(100).style.map(
                    lambda x: "background-color: #90EE90; color: black; font-weight: bold" if "🟢" in str(x)
                    else ("background-color: #F08080; color: black; font-weight: bold" if "🔴" in str(x)
                    else ("background-color: #FFD700; color: black; font-weight: bold" if "🟡" in str(x) else "")),
                    subset=['Diagnóstico FDI / Causa Raíz']
                ), use_container_width=True)
                
                st.plotly_chart(px.bar(df_conc.head(30), x='ID / Turno / Fecha', y=['Potencial Bloque Mina (%)', 'Recuperación Real Planta (%)'], 
                                       barmode='group', title="Comparativa Directa: Promesa del Bloque vs Respuesta Real del Turno"), use_container_width=True)
            else:
                st.warning("⚠️ Carga ambas fuentes de datos (Mina + Planta) en la barra lateral para generar la conciliación automatizada.")

        with t4:
            st.subheader("🎛️ Centro de Blending (Mezclas) y Optimización Prescriptiva")
            ref_res = res_m if res_m else res_p
            if ref_res:
                st.info("💡 **Simulador de Mezcla de Minerales (Cancha de Acopio / Stockpiles)**")
                c_b1, c_b2 = st.columns(2)
                
                with c_b1:
                    st.markdown("##### 🧱 Configuración del Bloque A (Mineral Base)")
                    b_a = {col: st.slider(f"{col} (A)", float(ref_res['df_clean'][col].min()), float(ref_res['df_clean'][col].max()), float(ref_res['df_clean'][col].mean()), key=f"ba_{col}") for col in ref_res['features']}
                    
                with c_b2:
                    st.markdown("##### 🧱 Configuración del Bloque B (Mineral de Alteración/Transición)")
                    b_b = {col: st.slider(f"{col} (B)", float(ref_res['df_clean'][col].min()), float(ref_res['df_clean'][col].max()), float(ref_res['df_clean'][col].quantile(0.25)), key=f"bb_{col}") for col in ref_res['features']}
                    
                st.divider()
                prop_a = st.slider("⚖️ Porcentaje de Bloque A en la Mezcla (%)", 0, 100, 70)
                prop_b = 100 - prop_a
                st.caption(f"Proporción Final de Alimentación: **{prop_a}% Bloque A + {prop_b}% Bloque B**")
                
                vec_blend = {col: (prop_a/100.0)*b_a[col] + (prop_b/100.0)*b_b[col] for col in ref_res['features']}
                df_blend = pd.DataFrame([vec_blend])
                
                pred_a = ref_res['m_glob'].predict(pd.DataFrame([b_a]))[0]
                pred_b = ref_res['m_glob'].predict(pd.DataFrame([b_b]))[0]
                pred_blend = ref_res['m_glob'].predict(df_blend)[0]
                
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Recuperación Bloque A Solo", f"{pred_a:.2f}%")
                mc2.metric("Recuperación Bloque B Solo", f"{pred_b:.2f}%")
                mc3.metric("Recuperación Mezcla (Blending)", f"{pred_blend:.2f}%", delta=f"{pred_blend - pred_b:.2f}% vs B")
                
                st.plotly_chart(go.Figure(data=[
                    go.Bar(name='Bloque A (Puro)', x=['Recuperación Estimada'], y=[pred_a]),
                    go.Bar(name='Bloque B (Puro)', x=['Recuperación Estimada'], y=[pred_b]),
                    go.Bar(name='Mezcla Blending', x=['Recuperación Estimada'], y=[pred_blend])
                ], layout=go.Layout(title="Impacto del Blending en la Recuperación Final")), use_container_width=True)

        with t5:
            st.subheader("🧠 Inteligencia Artificial Explicable (XAI) vía SHAP")
            ref_res = res_m if res_m else res_p
            if ref_res:
                st.markdown("Visualización de las variables que más influyen en la predicción del modelo:")
                X_samp = ref_res['X'].sample(min(100, len(ref_res['X'])))
                explainer = shap.Explainer(ref_res['m_glob'], X_samp)
                shap_v = explainer(X_samp)
                fig_s, _ = plt.subplots(); shap.summary_plot(shap_v, X_samp, show=False)
                st.pyplot(fig_s)
else:
    st.info("👈 Cargue al menos una fuente de datos (Mina / MineStart o Planta / Turnos) en la barra lateral para iniciar.")
