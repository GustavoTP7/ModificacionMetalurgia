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
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Geomet Twin Pro", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        # Eliminar columnas implícitas o sin nombre
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
        df = df.loc[:, df.columns != '']
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        st.error(f"Error en la ingesta de datos: {e}")
        return None

def aplicar_feature_engineering(df):
    df_feat = df.copy()
    cols = df_feat.columns.tolist()

    # --- DATASET TIPO MINASTART / BLOQUES ---
    if any(c in cols for c in ['CuT', 'CuT_pct', 'Cu_pct_Alim']):
        cut_col = 'CuT' if 'CuT' in cols else ('CuT_pct' if 'CuT_pct' in cols else 'Cu_pct_Alim')

        # Factor K / Cobre Soluble
        cus_col = 'CuS' if 'CuS' in cols else ('CuSac_pct' if 'CuSac_pct' in cols else None)
        cucn_col = 'CuCN' if 'CuCN' in cols else None

        if cus_col and cut_col and 'Factor_K_calc' not in cols:
            if cucn_col:
                df_feat['Factor_K_calc'] = ((df_feat[cus_col] + df_feat[cucn_col]) / np.maximum(df_feat[cut_col], 0.001)) * 100
            else:
                df_feat['Factor_K_calc'] = (df_feat[cus_col] / np.maximum(df_feat[cut_col], 0.001)) * 100

        # Razón Fe/Cu
        fe_col = 'Fe' if 'Fe' in cols else ('Fe_pct_Alim' if 'Fe_pct_Alim' in cols else None)
        if fe_col and cut_col and 'Razon_Fe_Cu_calc' not in cols:
            df_feat['Razon_Fe_Cu_calc'] = df_feat[fe_col] / np.maximum(df_feat[cut_col], 0.001)

        # Arcillas Totales
        arcilla_cols = [c for c in ['Cao_18', 'Mont_18', 'Filo_18', 'Arcillas_pct'] if c in cols]
        if arcilla_cols and 'Arcillas_Totales_calc' not in cols and len(arcilla_cols) > 1:
            df_feat['Arcillas_Totales_calc'] = df_feat[arcilla_cols].sum(axis=1)

        # Sulfuros Cu Totales
        sulf_cols = [c for c in ['CPY', 'CC', 'CV', 'BN', 'Chalcopyrite_pct', 'Min_Sec_pct'] if c in cols]
        if sulf_cols and 'Sulfuros_Cu_calc' not in cols and len(sulf_cols) > 1:
            df_feat['Sulfuros_Cu_calc'] = df_feat[sulf_cols].sum(axis=1)

        # Razón Sulfuros / Pirita
        pirita_col = 'PY' if 'PY' in cols else ('Pirita_pct' if 'Pirita_pct' in cols else None)
        if pirita_col and sulf_cols and 'Razon_Sulf_Pirita_calc' not in cols:
            df_feat['Razon_Sulf_Pirita_calc'] = df_feat[sulf_cols].sum(axis=1) / np.maximum(df_feat[pirita_col], 0.001)

        # Interacción TPH - P80
        tph_col = 'pltTph' if 'pltTph' in cols else ('Tonelaje Flotación' if 'Tonelaje Flotación' in cols else None)
        p80_col = 'P80' if 'P80' in cols else None
        if tph_col and p80_col and 'Interaccion_TPH_P80_calc' not in cols:
            df_feat['Interaccion_TPH_P80_calc'] = df_feat[tph_col] * df_feat[p80_col]

    return df_feat

# --- ENCABEZADO PRINCIPAL ---
st.title("💎 Geomet Twin Pro: Inteligencia Operacional Integrada")
st.markdown("""
**Digital Twin de Soporte a la Decisión (DSS)** para Minería y Procesamiento de Minerales.
Soporta arquitecturas de **Planificación por Bloques (MineStart)** y **Control Operacional de Planta (SCADA/Turnos)**.
""")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ 1. Arquitectura de Datos")
    archivo = st.file_uploader("Subir registros históricos (CSV o Excel)", type=["csv", "xlsx"])

    st.header("⚡ 2. Ingeniería de Variables")
    aplicar_fe = st.checkbox("Generar Ratios Geometalúrgicos Automáticos (Factor K, Fe/Cu, Arcillas)", value=True)
    modo_ruido = st.radio("Filtro de Outliers:", ["Data Original", "Depuración por IQR", "Isolation Forest (Multivariado)"])

    st.header("🤖 3. Motor de IA Autónomo")
    tipo_modelo = st.selectbox("Seleccionar Algoritmo:", ["XGBoost", "CatBoost"])
    estrategia_model = st.radio("Estrategia de Entrenamiento:", [
        "Modelos Especializados por UGM (Recomendado)",
        "Modelo Global Único"
    ])
    transformar_log = st.checkbox("Aplicar Transformación Logarítmica a variables sesgadas")
    balancear = st.checkbox("Balanceo SMOTE (Casos Críticos)")

    st.divider()
    ejecutar = st.button("🚀 Iniciar Simulación Digital", use_container_width=True, type="primary")

if archivo is not None:
    df_raw = cargar_datos(archivo)

    if df_raw is not None:
        # Aplicar Feature Engineering si está activado
        if aplicar_fe:
            df_proc = aplicar_feature_engineering(df_raw)
        else:
            df_proc = df_raw.copy()

        # Detectar automáticamente contexto del dataset
        cols_lower = [str(c).lower() for c in df_proc.columns]

        es_bloques = any(kw in cols_lower for kw in ['mining block', 'shovel', 'phase', 'level', 'payload', 'bwi', 'cpy'])
        es_turnos = any(kw in cols_lower for kw in ['id_turno', 'shift', 'cu_pct_alim', 'tonelaje flotación', 'rec ro'])

        if es_bloques:
            contexto_str = "🏗️ Planificación Geometalúrgica (Bloques de Minado / MineStart)"
        elif es_turnos:
            contexto_str = "🏭 Control Operacional &amp; Auditoría (Turnos / Planta SCADA)"
        else:
            contexto_str = "🔍 Data Geometalúrgica General"

        st.info(f"**Contexto Detectado:** {contexto_str}")

        # Detectar columna de Identificador / Metadata (para la Pestaña 5 - Monitor FDI)
        id_col = None
        for col in df_proc.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['mining block', 'id_turno', 'id', 'fecha', 'date', 'turno', 'sample', 'muestra', 'day']):
                id_col = col
                break

        if id_col is None:
            first_col = df_proc.columns[0]
            if not pd.api.types.is_numeric_dtype(df_proc[first_col]) or pd.api.types.is_datetime64_any_dtype(df_proc[first_col]):
                id_col = first_col

        # Separar columnas numéricas para IA
        columnas_num = df_proc.select_dtypes(include=[np.number]).columns.tolist()
        if id_col in columnas_num:
            columnas_num.remove(id_col)

        # Sugerir la mejor variable objetivo (Y)
        target_default_idx = len(columnas_num) - 1
        for i, c in enumerate(columnas_num):
            if c.lower() in ['reccu', 'rec ro', 'rec_cu', 'recuperacion', 'recuperación', 'rec_ro', 'recmo']:
                target_default_idx = i
                break

        with st.sidebar:
            st.header("🎯 4. Selección de Variables")
            target = st.selectbox("Variable Objetivo (Y):", columnas_num, index=target_default_idx)

            # Predictores físicos numéricos
            posibles_features = [c for c in columnas_num if c != target and not any(kw in c.lower() for kw in ['recag', 'recmo', 'id', 'index'])]
            features = st.multiselect("Predictores (X):", [c for c in columnas_num if c != target],
                                     default=posibles_features)

        # --- LÓGICA DE PERSISTENCIA Y ENTRENAMIENTO ---
        if ejecutar or ('sub_models' in st.session_state and 'model' in st.session_state):
            if ejecutar:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # FASE 1: Preparación y Depuración
                status_text.text("Fase 1/5: Refinando datos y filtrando ruido...")
                df_num = df_proc[columnas_num].dropna().reset_index(drop=True)

                if id_col:
                    id_series = df_proc.loc[df_num.index, id_col].astype(str).values
                else:
                    id_series = np.array([f"Registro_{i+1}" for i in range(len(df_num))])

                df = df_num.copy()
                mask = np.ones(len(df), dtype=bool)

                if modo_ruido == "Depuración por IQR":
                    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
                    IQR = Q3 - Q1
                    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
                elif modo_ruido == "Isolation Forest (Multivariado)":
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    mask = iso.fit_predict(df[features + [target]]) == 1

                df = df[mask].reset_index(drop=True)
                ids = id_series[mask]
                progress_bar.progress(20)

                # FASE 2: Dominios Geometalúrgicos (UGM) vía Clustering
                status_text.text("Fase 2/5: Identificando UGM dinámicas...")
                best_k, best_score = 2, -1
                for k in range(2, 6):
                    if len(df) > k:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(df[features + [target]])
                        score = silhouette_score(df[features + [target]], labels)
                        if score > best_score: best_score, best_k = score, k
                kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                df['Dominio_GMD'] = kmeans_final.fit_predict(df[features + [target]])
                progress_bar.progress(40)

                # FASE 3: Transformación Asimétrica y SMOTE
                X = df[features].copy()
                y = df[target].values
                dominios = df['Dominio_GMD'].values
                id_f = ids

                if transformar_log:
                    status_text.text("Fase 3/5: Normalizando distribuciones sesgadas...")
                    for c in features:
                        if df[c].min() >= 0 and abs(df[c].skew()) > 1.0:
                            X[c] = np.log1p(X[c])

                if balancear:
                    status_text.text("Fase 3/5: Aplicando SMOTE para balanceo de casos críticos...")
                    y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                    sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                    X_with_y = X.copy()
                    X_with_y['__t__'] = y
                    X_with_y['__dom__'] = dominios
                    X_res, _ = sm.fit_resample(X_with_y, y_disc)

                    y = X_res['__t__'].values
                    dominios = np.round(X_res['__dom__'].values).astype(int)
                    X = X_res[features]

                    n_sinteticos = len(X_res) - len(ids)
                    if n_sinteticos > 0:
                        id_f = np.concatenate([ids, [f"SMOTE_{i+1}" for i in range(n_sinteticos)]])
                progress_bar.progress(60)

                # FASE 4 &amp; 5: Entrenamiento y Validación Cruzada (K-Fold)
                status_text.text(f"Fase 4/5: Entrenando IA ({estrategia_model})...")

                sub_models = {}
                y_pred_cv = np.zeros_like(y)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                # Modelo global siempre entrenado como respaldo
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                if tipo_modelo == "XGBoost":
                    m_global = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
                    m_global.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    m_global = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
                    m_global.fit(X_train, y_train, eval_set=(X_val, y_val))

                if estrategia_model == "Modelos Especializados por UGM (Recomendado)":
                    status_text.text("Fase 5/5: Evaluando sub-modelos independientes por UGM...")
                    for dom in np.unique(dominios):
                        idx_dom = np.where(dominios == dom)[0]
                        X_dom = X.iloc[idx_dom] if isinstance(X, pd.DataFrame) else X[idx_dom]
                        y_dom = y[idx_dom]

                        y_pred_dom = np.zeros_like(y_dom)
                        # Umbral adaptativo: Si N >= 25 se entrena sub-modelo local; de lo contrario se usa el Modelo Global
                        if len(y_dom) >= 25:
                            n_splits_u = min(5, len(y_dom))
                            kf_u = KFold(n_splits=n_splits_u, shuffle=True, random_state=42)
                            for train_in, val_in in kf_u.split(X_dom):
                                if tipo_modelo == "XGBoost":
                                    m_u = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
                                else:
                                    m_u = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, random_state=42, verbose=0)
                                m_u.fit(X_dom.iloc[train_in], y_dom[train_in], verbose=False if tipo_modelo == "XGBoost" else 0)
                                y_pred_dom[val_in] = m_u.predict(X_dom.iloc[val_in])

                            # Sub-modelo final entrenado en la UGM
                            if tipo_modelo == "XGBoost":
                                m_final_u = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42)
                            else:
                                m_final_u = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_state=42, verbose=0)
                            m_final_u.fit(X_dom, y_dom, verbose=False if tipo_modelo == "XGBoost" else 0)
                            sub_models[dom] = m_final_u
                        else:
                            # Respaldo por modelo global para poblaciones pequeñas
                            sub_models[dom] = m_global
                            y_pred_dom = m_global.predict(X_dom)

                        y_pred_cv[idx_dom] = y_pred_dom
                else:
                    status_text.text("Fase 5/5: Validando modelo global K-Fold...")
                    y_pred_cv = cross_val_predict(m_global, X, y, cv=kf)
                    for dom in np.unique(dominios):
                        sub_models[dom] = m_global

                # Guardado en Estado de Sesión
                st.session_state.model = m_global
                st.session_state.sub_models = sub_models
                st.session_state.kmeans_final = kmeans_final
                st.session_state.df_p = df
                st.session_state.y_pred = y_pred_cv
                st.session_state.metrics = (r2_score(y, y_pred_cv), mean_absolute_error(y, y_pred_cv),
                                           np.sqrt(mean_squared_error(y, y_pred_cv)),
                                           mean_absolute_percentage_error(y, y_pred_cv) * 100)
                st.session_state.X_f = X
                st.session_state.y_f = y
                st.session_state.dominios_f = dominios
                st.session_state.id_f = id_f
                st.session_state.id_col = id_col if id_col else "Fecha / ID Turno / Bloque"
                st.session_state.transformar_log = transformar_log

                progress_bar.progress(100); time.sleep(0.5); status_text.empty(); progress_bar.empty()

            # --- RENDERIZADO DE PESTAÑAS ---
            model = st.session_state.model
            sub_models = st.session_state.sub_models
            kmeans_final = st.session_state.kmeans_final
            df_p = st.session_state.df_p
            y_pred, y_f = st.session_state.y_pred, st.session_state.y_f
            dominios_f = st.session_state.dominios_f
            id_f = st.session_state.id_f
            id_col_nombre = st.session_state.id_col
            r2, mae, rmse, mape = st.session_state.metrics
            X_f = st.session_state.X_f
            transformar_log = st.session_state.transformar_log

            # Helper de predicción inteligente
            def predecir_muestra(df_input):
                X_in = df_input[features].copy()
                if transformar_log:
                    for c in features:
                        if df_p[c].min() >= 0 and abs(df_p[c].skew()) > 1.0:
                            X_in[c] = np.log1p(X_in[c])

                try:
                    dummy_target = pd.DataFrame({target: [df_p[target].mean()]*len(df_input)})
                    X_target_dummy = pd.concat([df_input[features], dummy_target], axis=1)
                    dom_in = kmeans_final.predict(X_target_dummy)
                except:
                    dom_in = np.zeros(len(df_input), dtype=int)

                preds = []
                for i in range(len(df_input)):
                    d = dom_in[i]
                    m = sub_models.get(d, model)
                    row_x = X_in.iloc[[i]]
                    preds.append(m.predict(row_x)[0])
                return np.array(preds)

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Calidad de Datos", "📊 Análisis Multivariante", "🎯 Score de Precisión", "🎛️ Simulador", "🚨 Monitor FDI", "🧠 XAI"
            ])

            with tab1:
                st.subheader("Caracterización de Unidades Geometalúrgicas (UGM)")
                st.dataframe(df_p.groupby('Dominio_GMD')[features + [target]].mean().style.background_gradient(cmap='viridis'))
                c1, c2 = st.columns(2)
                vx = c1.selectbox("Eje X:", df_p.columns, key="v_x")
                vy = c1.selectbox("Eje Y:", df_p.columns, index=columnas_num.index(target) if target in columnas_num else 0, key="v_y")
                if c1.button("🔄 Actualizar Gráfico"):
                    st.session_state.fig_exp = px.scatter(df_p, x=vx, y=vy, color='Dominio_GMD', trendline="ols") if vx != vy else px.histogram(df_p, x=vx, color='Dominio_GMD')
                if 'fig_exp' in st.session_state: c2.plotly_chart(st.session_state.fig_exp, use_container_width=True)

            with tab2:
                ch, ci = st.columns(2)
                ch.write("**Heatmap de Correlación**")
                ch.plotly_chart(px.imshow(df_p[[target] + features].corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)
                ci.write("**Ranking de Importancia de Variables (IA)**")
                imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_feature_importance()
                ci.plotly_chart(px.bar(pd.DataFrame({'V': features, 'I': imp}).sort_values('I'), x='I', y='V', orientation='h'), use_container_width=True)

            with tab3:
                st.subheader("🎯 Fidelidad Predictiva del Gemelo Digital")

                # 1. Métricas Globales
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad Global (R²)", f"{r2:.3f}")
                m2.metric("Error Global (MAE)", f"{mae:.3f}")
                m3.metric("Riesgo Global (RMSE)", f"{rmse:.3f}")
                m4.metric("Error Relativo Global (MAPE)", f"{mape:.2f}%")

                st.divider()

                # 2. Desglose Diferenciado por UGM / Dominio
                st.subheader("📊 Evaluación de Desempeño por Unidad Geometalúrgica (UGM)")
                st.markdown("Cada UGM posee una respuesta metalúrgica distinta. A continuación se evalúa la precisión del modelo dentro de cada dominio:")

                metrics_ugm = []
                for dom in sorted(np.unique(dominios_f)):
                    idx = (dominios_f == dom)
                    y_real_ugm = y_f[idx]
                    y_pred_ugm = y_pred[idx]

                    if len(y_real_ugm) > 1:
                        r2_u = r2_score(y_real_ugm, y_pred_ugm)
                        mae_u = mean_absolute_error(y_real_ugm, y_pred_ugm)
                        rmse_u = np.sqrt(mean_squared_error(y_real_ugm, y_pred_ugm))
                        mape_u = mean_absolute_percentage_error(y_real_ugm, y_pred_ugm) * 100

                        metrics_ugm.append({
                            "UGM / Dominio": f"Dominio {dom}",
                            "N° Muestras": len(y_real_ugm),
                            "R² (Fidelidad)": round(r2_u, 3),
                            "MAE (% Rec)": round(mae_u, 3),
                            "RMSE": round(rmse_u, 3),
                            "MAPE (%)": f"{mape_u:.2f}%"
                        })

                if metrics_ugm:
                    df_metrics_ugm = pd.DataFrame(metrics_ugm)
                    st.dataframe(
                        df_metrics_ugm.style.background_gradient(subset=["R² (Fidelidad)"], cmap="RdYlGn"),
                        use_container_width=True
                    )

                st.plotly_chart(
                    px.scatter(
                        x=y_f, y=y_pred,
                        color=[f"Dominio {d}" for d in dominios_f],
                        labels={'x': 'Recuperación Real (%)', 'y': 'Recuperación Digital (%)', 'color': 'UGM'},
                        title="Comparativa Real vs Digital por Dominio UGM",
                        trendline="ols"
                    ),
                    use_container_width=True
                )

            with tab4:
                st.subheader("🎛️ Centro de Optimización Prescriptiva")
                col_ctrl, col_res = st.columns(2)
                with col_ctrl:
                    st.info("🎮 **Ajuste Manual de Set-Points**")
                    inputs_sim = {col: st.slider(f"{col}", float(df_p[col].min()), float(df_p[col].max()), float(df_p[col].mean()), key=f"s_{col}") for col in features}
                    st.divider()
                    btn_opt = st.button("🚀 ENCONTRAR OPERACIÓN ÓPTIMA", use_container_width=True, type="primary")

                with col_res:
                    pred_manual = predecir_muestra(pd.DataFrame([inputs_sim]))[0]
                    if btn_opt:
                        rand_data = pd.DataFrame({c: np.random.uniform(df_p[c].min(), df_p[c].max(), 1000) for c in features})
                        preds_opt = predecir_muestra(rand_data)
                        top_idx = np.argsort(preds_opt)[-5:][::-1]
                        st.session_state.top_5 = rand_data.iloc[top_idx].copy()
                        st.session_state.top_5['Recuperación_Estimada'] = preds_opt[top_idx]

                    if 'top_5' in st.session_state:
                        mejor_cfg = st.session_state.top_5.head(1).squeeze().to_dict()
                        mejor_val = mejor_cfg.pop('Recuperación_Estimada')
                        ganancia = mejor_val - pred_manual

                        cont = st.container(border=True)
                        mc1, mc2 = cont.columns(2)
                        mc1.metric("Recuperación Actual", f"{pred_manual:.2f}%")
                        mc2.metric("Máximo Técnico", f"{mejor_val:.2f}%", delta=f"{ganancia:.2f}%")

                        st.write("### 🥇 Top 5 Escenarios Recomendados")
                        st.dataframe(st.session_state.top_5.style.background_gradient(subset=['Recuperación_Estimada'], cmap='Blues'), use_container_width=True)

                        # NORMALIZACIÓN VISUAL MIN-MAX
                        y_manual_norm = []
                        y_opt_norm = []

                        for f in features:
                            f_min = float(df_p[f].min())
                            f_max = float(df_p[f].max())
                            rango = f_max - f_min if (f_max - f_min) > 0 else 1

                            val_man_norm = ((inputs_sim[f] - f_min) / rango) * 100
                            val_opt_norm = ((mejor_cfg[f] - f_min) / rango) * 100

                            y_manual_norm.append(val_man_norm)
                            y_opt_norm.append(val_opt_norm)

                        fig_comp = go.Figure()
                        fig_comp.add_trace(go.Bar(
                            name='Manual',
                            x=features,
                            y=y_manual_norm,
                            hovertemplate="%{x}: <b>%{customdata}</b> (rango: %{y:.1f}%)",
                            customdata=[f"{inputs_sim[f]:.2f}" for f in features]
                        ))
                        fig_comp.add_trace(go.Bar(
                            name='Óptimo',
                            x=features,
                            y=y_opt_norm,
                            hovertemplate="%{x}: <b>%{customdata}</b> (rango: %{y:.1f}%)",
                            customdata=[f"{mejor_cfg[f]:.2f}" for f in features]
                        ))

                        rango_escala_y = [0.0, 100.0]

                        fig_comp.update_layout(
                            title="Comparativa de Set-Points (Normalizado: 0% a 100% de su Rango Operativo)",
                            barmode='group',
                            height=380,
                            yaxis_title="Posición en el Rango (%)",
                            yaxis=dict(range=rango_escala_y)
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)

            with tab5:
                st.subheader("🚨 Protocolo FDI: Auditoría de Turnos y Detección de Anomalías")
                df_audit = pd.DataFrame(X_f, columns=features) if isinstance(X_f, np.ndarray) else X_f.copy()

                # Insertar identificador de Metadata/Turno/Bloque y Dominio UGM
                df_audit.insert(0, 'UGM / Dominio', [f"Dominio {d}" for d in dominios_f])
                df_audit.insert(0, id_col_nombre, id_f)

                df_audit['Rec. Real (%)'] = y_f
                df_audit['Rec. Digital (%)'] = y_pred
                df_audit['Error Absoluto'] = np.abs(df_audit['Rec. Real (%)'] - df_audit['Rec. Digital (%)'])

                def evaluar_semaforo(e):
                    return "🟢 Normal" if e <= mae else ("🟡 Advertencia" if e <= 2*mae else "🔴 Anomalía")

                df_audit['Estado FDI'] = df_audit['Error Absoluto'].apply(evaluar_semaforo)

                columnas_mostrar = [id_col_nombre, 'UGM / Dominio', 'Estado FDI', 'Rec. Real (%)', 'Rec. Digital (%)', 'Error Absoluto'] + features

                st.dataframe(
                    df_audit[columnas_mostrar].head(500).style.map(
                        lambda x: "background-color: #90EE90; color: black; font-weight: bold" if x == "🟢 Normal"
                        else ("background-color: #FFD700; color: black; font-weight: bold" if x == "🟡 Advertencia"
                        else ("background-color: #F08080; color: black; font-weight: bold" if x == "🔴 Anomalía" else "")),
                        subset=['Estado FDI']
                    ),
                    use_container_width=True
                )

            with tab6:
                st.subheader("IA Explicable (XAI) via SHAP")
                X_sample = X_f.sample(min(100, len(X_f))) if isinstance(X_f, pd.DataFrame) else pd.DataFrame(X_f, columns=features).sample(min(100, len(X_f)))
                explainer = shap.Explainer(model, X_sample)
                shap_v = explainer(X_sample)
                fig_s, _ = plt.subplots(); shap.summary_plot(shap_v, X_sample, show=False)
                st.pyplot(fig_s)
        else:
            st.info("💡 Configure los parámetros y pulse 'Iniciar Simulación Digital' para procesar los datos.")
else:
    st.info("👈 Cargue el dataset histórico para iniciar el Digital Twin.")

Gemini Notebook puede ofrecer respuestas incorrectas. Compruébalas.

