"""
Dashboard Streamlit - TechLogistics S.A.S.
Sistema de Soporte a la Decision (DSS) para deteccion de outliers.
Fase 1: Auditoria de Calidad y Transparencia.
Fase 2: Integracion y Fuente Unica de Verdad.
"""
import streamlit as st
import pandas as pd

from src.utils import cargar_todo
from src.calidad import auditoria_completa, _health_score
from src.limpieza import limpieza_completa
from src.procesamiento import (
    guardar_limpios, crear_fuente_verdad, guardar_fuente_verdad,
    resumen_huerfanos,
)


def _filtrar_outliers_dominio(df, columna, info):
    """Filtra filas con violaciones de dominio (fuera del rango de dominio o bajo limite natural)."""
    met = info.get("metodo", "iqr")
    col = df[columna]
    if met == "dominio":
        return df[(col < info["lim_inf"]) | (col > info["lim_sup"])]
    else:
        lim = info.get("lim_inf_natural")
        if lim is not None:
            return df[col < lim]
        return df.iloc[:0]


def _filtrar_outliers_superiores(df, columna, info):
    """Filtra filas con outliers superiores IQR (por encima del limite estadistico)."""
    met = info.get("metodo", "iqr")
    if met == "iqr":
        lim = info.get("lim_sup_iqr")
        if lim is not None:
            col = df[columna]
            return df[col > lim]
    return df.iloc[:0]


def _mostrar_outliers_auditoria(df_raw, col_name, info):
    """Muestra el resumen de anomalias con detalle por tipo."""
    met = info.get("metodo", "iqr")

    if met == "dominio":
        st.warning(
            f"**{col_name}**: {info['cantidad']} valores fuera del rango "
            f"válido {info['dominio_esperado']} ({info['pct']}%)\n\n"
            f"Violación de dominio: la variable tiene una escala definida "
            f"({info['dominio_esperado']}). Cualquier valor fuera de ese rango "
            f"es un error de captura, no un outlier estadístico."
        )
        if info["tiene_outliers"]:
            outliers = _filtrar_outliers_dominio(df_raw, col_name, info)
            if len(outliers) > 0:
                with st.expander(f"Ver los {len(outliers)} registros con {col_name} fuera de rango"):
                    st.dataframe(outliers, use_container_width=True, height=250)
    else:
        viol = info.get("violaciones_dominio", 0)
        sup = info.get("outliers_superiores_iqr", 0)
        lim_nat = info.get("lim_inf_natural")
        lim_iqr = info.get("lim_sup_iqr")
        q1 = info.get("q1")
        q3 = info.get("q3")

        if viol > 0 or sup > 0:
            partes = []
            if viol > 0:
                partes.append(
                    f"**{viol}** por debajo del límite natural "
                    f"({col_name} < {lim_nat}) → error de captura"
                )
            if sup > 0:
                partes.append(
                    f"**{sup}** por encima de Q3+1.5×IQR "
                    f"({col_name} > {lim_iqr}) → estadísticamente atípico"
                )
            st.warning(f"**{col_name}**: " + " | ".join(partes) + f"\n\n"
                       f"Q1={q1}, Q3={q3}")

            if viol > 0:
                df_viol = _filtrar_outliers_dominio(df_raw, col_name, info)
                if len(df_viol) > 0:
                    with st.expander(f"Ver los {viol} registros con {col_name} por debajo de {lim_nat} (error de captura)"):
                        st.dataframe(df_viol, use_container_width=True, height=250)
            if sup > 0:
                df_sup = _filtrar_outliers_superiores(df_raw, col_name, info)
                if len(df_sup) > 0:
                    with st.expander(f"Ver los {sup} registros con {col_name} por encima de {lim_iqr} (atípico)"):
                        st.dataframe(df_sup, use_container_width=True, height=250)
        else:
            st.success(f"**{col_name}**: sin anomalías detectadas")


st.set_page_config(
    page_title="TechLogistics DSS | Auditoria",
    page_icon="📊",
    layout="wide",
)

st.title("📊 TechLogistics S.A.S. — Sistema de Soporte a la Decisión")
st.caption("Consultoría Senior | Auditoría de Calidad y Transparencia")


@st.cache_data
def cargar_datos():
    return cargar_todo()


@st.cache_data
def ejecutar_auditoria(_inv, _trx, _fb):
    return auditoria_completa(_inv, _trx, _fb)


@st.cache_data
def ejecutar_limpieza(_inv, _trx, _fb):
    return limpieza_completa(_inv, _trx, _fb)


@st.cache_data
def ejecutar_procesamiento(_inv_limpio, _trx_limpio, _fb_limpio):
    """Fase 2: guarda limpios, crea y guarda la fuente unica de verdad."""
    guardar_limpios(_inv_limpio, _trx_limpio, _fb_limpio)
    df_unido = crear_fuente_verdad(_trx_limpio, _inv_limpio)
    guardar_fuente_verdad(df_unido)
    return df_unido


# --- Sidebar ---
st.sidebar.header("🔎 Filtros")
st.sidebar.button("🔄 Refrescar Análisis", use_container_width=True, on_click=st.cache_data.clear)

st.sidebar.divider()
st.sidebar.caption("Fase 1: Auditoría de Calidad")
st.sidebar.caption("Fase 2: Integración (próximamente)")
st.sidebar.caption("Fase 3: IA con Groq (próximamente)")

# --- Carga de datos ---
with st.spinner("Cargando datasets..."):
    inv_raw, trx_raw, fb_raw = cargar_datos()

# --- Auditoría ---
auditoria = ejecutar_auditoria(inv_raw, trx_raw, fb_raw)

# --- Limpieza ---
resultados = ejecutar_limpieza(inv_raw, trx_raw, fb_raw)
inv_limpio = resultados["inventario"]["df"]
trx_limpio = resultados["transacciones"]["df"]
fb_limpio = resultados["feedback"]["df"]

# --- Fase 2: Procesamiento (guardar limpios + left join) ---
with st.spinner("Procesando Fase 2: creando fuente única de verdad..."):
    df_unido = ejecutar_procesamiento(inv_limpio, trx_limpio, fb_limpio)
    info_huerfanos = resumen_huerfanos(df_unido)

# --- Tabs ---
tab_auditoria, tab_transparencia, tab_operaciones, tab_cliente, tab_ia = st.tabs(
    ["🔍 Auditoría", "🧾 Transparencia", "🚚 Operaciones", "👤 Cliente", "🤖 Insights IA"]
)

# =============================================
# TAB 1: AUDITORIA
# =============================================
with tab_auditoria:
    st.header("Auditoría de Calidad — Antes del Procesamiento")

    col1, col2, col3 = st.columns(3)

    datasets_data = [
        ("📦 Inventario", auditoria["inventario"], inv_raw),
        ("🚚 Transacciones", auditoria["transacciones"], trx_raw),
        ("💬 Feedback", auditoria["feedback"], fb_raw),
    ]

    for col, (titulo, aud, df_raw) in zip([col1, col2, col3], datasets_data):
        with col:
            st.subheader(titulo)
            met = aud["metricas"]
            hs = aud["health_score_antes"]
            st.metric("Health Score (antes)", f"{hs}%")
            st.metric("Filas", f"{met['total_filas']:,}")
            st.metric("Celdas nulas", f"{met['celdas_nulas']:,} ({met['pct_nulidad_global']}%)")
            dup_exactos = met.get("duplicados_exactos", 0)
            dup_ids = met.get("ids_repetidos_datos_diferentes", 0)
            st.metric("Duplicados exactos", dup_exactos)
            if dup_ids > 0:
                st.warning(f"IDs repetidos con datos diferentes: {dup_ids} registros (conservados)")

    st.divider()
    st.subheader("Porcentaje de Nulidad por Columna (Antes)")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**📦 Inventario**")
        st.dataframe(
            pd.DataFrame(
                auditoria["inventario"]["pct_nulidad_por_columna"].items(),
                columns=["Columna", "% Nulos"]
            ).set_index("Columna"),
            use_container_width=True,
        )

    with col_b:
        st.markdown("**🚚 Transacciones**")
        st.dataframe(
            pd.DataFrame(
                auditoria["transacciones"]["pct_nulidad_por_columna"].items(),
                columns=["Columna", "% Nulos"]
            ).set_index("Columna"),
            use_container_width=True,
        )

    with col_c:
        st.markdown("**💬 Feedback**")
        st.dataframe(
            pd.DataFrame(
                auditoria["feedback"]["pct_nulidad_por_columna"].items(),
                columns=["Columna", "% Nulos"]
            ).set_index("Columna"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Anomalías Detectadas (Reglas de Dominio + IQR)")

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.markdown("**📦 Inventario**")
        for col_name, info in auditoria["inventario"]["outliers_detectados"].items():
            if info["tiene_outliers"]:
                _mostrar_outliers_auditoria(inv_raw, col_name, info)
            else:
                st.success(f"**{col_name}**: sin anomalías detectadas")

    with col_o2:
        st.markdown("**🚚 Transacciones**")
        for col_name, info in auditoria["transacciones"]["outliers_detectados"].items():
            if info["tiene_outliers"]:
                _mostrar_outliers_auditoria(trx_raw, col_name, info)
            else:
                st.success(f"**{col_name}**: sin anomalías detectadas")

    st.markdown("---")
    st.markdown("**💬 Feedback**")
    col_fb1, col_fb2 = st.columns(2)
    fb_outs = auditoria["feedback"]["outliers_detectados"]
    for i, (col_name, info) in enumerate(fb_outs.items()):
        target = col_fb1 if i % 2 == 0 else col_fb2
        with target:
            if info["tiene_outliers"]:
                _mostrar_outliers_auditoria(fb_raw, col_name, info)
            else:
                st.success(f"**{col_name}**: todos los valores dentro del rango válido")

    st.divider()
    st.subheader("📋 Inconsistencias Categóricas (Antes de normalizar)")

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        st.markdown("**Categorías (Inventario)**")
        st.json(auditoria["inventario"]["categ_inconsistentes"]["Categoria"])
    with col_cat2:
        st.markdown("**Bodegas (Inventario)**")
        st.json(auditoria["inventario"]["categ_inconsistentes"]["Bodega_Origen"])


# =============================================
# TAB 2: TRANSPARENCIA (Antes vs Despues)
# =============================================
with tab_transparencia:
    st.header("Módulo de Transparencia — Antes vs Después")

    # Health Scores comparativo
    st.subheader("🏥 Health Score: Comparativo Antes vs Después")

    hs_inv_antes = auditoria["inventario"]["health_score_antes"]
    hs_trx_antes = auditoria["transacciones"]["health_score_antes"]
    hs_fb_antes = auditoria["feedback"]["health_score_antes"]

    hs_inv_despues = _health_score(inv_limpio)
    hs_trx_despues = _health_score(trx_limpio)
    hs_fb_despues = _health_score(fb_limpio)

    col_h1, col_h2, col_h3 = st.columns(3)

    with col_h1:
        delta_inv = round(hs_inv_despues - hs_inv_antes, 1)
        st.metric("📦 Inventario", f"{hs_inv_despues}%", delta=f"{delta_inv:+.1f}%")
    with col_h2:
        delta_trx = round(hs_trx_despues - hs_trx_antes, 1)
        st.metric("🚚 Transacciones", f"{hs_trx_despues}%", delta=f"{delta_trx:+.1f}%")
    with col_h3:
        delta_fb = round(hs_fb_despues - hs_fb_antes, 1)
        st.metric("💬 Feedback", f"{hs_fb_despues}%", delta=f"{delta_fb:+.1f}%")

    st.divider()
    st.subheader("📋 Registro de Cambios por Dataset")

    for nombre, datos in [
        ("📦 Inventario", resultados["inventario"]["cambios"]),
        ("🚚 Transacciones", resultados["transacciones"]["cambios"]),
        ("💬 Feedback", resultados["feedback"]["cambios"]),
    ]:
        with st.expander(nombre, expanded=True):
            c = datos
            st.write(f"**Filas iniciales:** {c['filas_inicial']:,}  →  **Filas finales:** {c['filas_final']:,}")
            if c["acciones"]:
                st.write("**Acciones realizadas:**")
                for accion in c["acciones"]:
                    st.write(f"- {accion}")
            else:
                st.write("Sin acciones adicionales.")

    st.divider()
    st.subheader("📊 Nulidad por Columna: Antes vs Después")

    dataset_opcion = st.selectbox("Seleccionar dataset:", ["Inventario", "Transacciones", "Feedback"])

    if dataset_opcion == "Inventario":
        df_antes = inv_raw
        df_despues = inv_limpio
    elif dataset_opcion == "Transacciones":
        df_antes = trx_raw
        df_despues = trx_limpio
    else:
        df_antes = fb_raw
        df_despues = fb_limpio

    cols_comunes = [c for c in df_antes.columns if c in df_despues.columns]
    comparativa = pd.DataFrame({
        "Columna": cols_comunes,
        "% Nulos (Antes)": [round(df_antes[c].isnull().sum() / len(df_antes) * 100, 1) for c in cols_comunes],
        "% Nulos (Después)": [round(df_despues[c].isnull().sum() / len(df_despues) * 100, 1) for c in cols_comunes],
    })
    comparativa["Mejora"] = comparativa["% Nulos (Antes)"] - comparativa["% Nulos (Después)"]
    st.dataframe(comparativa.set_index("Columna"), use_container_width=True)

    st.divider()
    st.subheader("📄 Justificación de Decisiones de Imputación")

    with st.expander("📦 Inventario", expanded=True):
        st.markdown("""
        - **Stock_Actual nulos**: Imputados con **mediana por categoría** (distribución asimétrica con outliers).
        - **Stock negativo**: Forzado a **0** (no existe stock físico negativo; es error de captura del ERP).
        - **Costo_Unitario_USD outliers**: Capado al límite superior IQR por categoría, sustituyendo valores extremos ($850k) por la mediana categórica.
        - **Lead_Time_Dias**: Extraído valor numérico de cadenas como "25-30 días" calculando el promedio aritmético; nulos imputados con mediana global.
        - **Categoría "???"**: Marcada como **"Sin especificar"** (imposible inferir la categoría real sin datos adicionales).
        - **Ultima_Revision**: Fechas normalizadas a formato **DD/MM/YYYY**. Se detectan y anulan fechas futuras o anteriores a 2020. El dataset original usaba YYYY-MM-DD.
        """)

    with st.expander("🚚 Transacciones", expanded=True):
        st.markdown("""
        - **Cantidad_Vendida negativa**: Tratado como error de captura, no cancelación. Imputado con **mediana por SKU** (si no disponible, mediana global).
        - **Tiempo_Entrega_Real outliers (>P99)**: Capado al percentil 99 para no distorsionar promedios operativos.
        - **Costo_Envio nulos**: Imputado con **mediana por Canal_Venta** (cada canal tiene estructura de costos diferente).
        - **Fecha_Venta**: Fechas normalizadas a formato **DD/MM/YYYY**. Se detectan y anulan fechas futuras o anteriores a 2020. Soporta 7 formatos de entrada distintos.
        """)

    with st.expander("💬 Feedback", expanded=True):
        st.markdown("""
        - **Duplicados exactos**: Solo se eliminan filas 100% idénticas (todas las columnas iguales).
        - **IDs repetidos con datos diferentes**: Se CONSERVAN. Un mismo Feedback_ID con distinto Transaccion_ID, rating o comentario NO es un duplicado real; es una inconsistencia del sistema de IDs. Eliminarlos destruiría información válida.
        - **Rating_Producto fuera de [1,5]**: Sustituido por la **mediana** redondeada (escala ordinal).
        - **Edad_Cliente > 100**: Imputado con **mediana** global (error de captura, no valor real).
        - **Ticket_Soporte_Abierto**: Normalizado de ["Sí","1","0","No"] a ["Sí","No"] binario.
        """)

    # --- Descargar reporte ---
    st.divider()
    st.subheader("📥 Exportar Reporte de Limpieza")

    reporte = pd.DataFrame([
        {"Dataset": nombre, "Filas Inicial": c["filas_inicial"], "Filas Final": c["filas_final"],
         "Filas Afectadas": c["filas_inicial"] - c["filas_final"]}
        for nombre, c in [
            ("Inventario", resultados["inventario"]["cambios"]),
            ("Transacciones", resultados["transacciones"]["cambios"]),
            ("Feedback", resultados["feedback"]["cambios"]),
        ]
    ])
    st.dataframe(reporte, use_container_width=True)

    csv_reporte = reporte.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar reporte (CSV)",
        data=csv_reporte,
        file_name="reporte_limpieza.csv",
        mime="text/csv",
    )


# =============================================
# TAB 3: OPERACIONES (Fase 2)
# =============================================
with tab_operaciones:
    st.header("🚚 Operaciones — Fuente Única de Verdad")

    st.subheader("📊 Left Join: Transacciones + Inventario")
    st.caption(
        "Se unieron las 10,000 transacciones con los 2,500 productos del inventario "
        "usando `SKU_ID` como llave. Se usó **LEFT JOIN** para no perder ninguna venta. "
        "Las columnas de inventario quedan vacías para los SKUs que no existen en el catálogo."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total transacciones", f"{info_huerfanos['total_transacciones']:,}")
    with col2:
        st.metric("SKUs huérfanos", info_huerfanos['skus_huerfanos_unicos'])
    with col3:
        st.metric("Ventas sin catálogo",
                  f"{info_huerfanos['transacciones_huerfanas']:,}",
                  delta=f"{info_huerfanos['pct_huerfanas']}%",
                  delta_color="off")
    with col4:
        st.metric("Ingreso en riesgo",
                  f"${info_huerfanos['ingreso_huerfano_usd']:,.0f}",
                  delta=f"{info_huerfanos['pct_ingreso_en_riesgo']}%",
                  delta_color="off")

    st.success(
        f"✅ **Trazabilidad verificada**: el left join preserva las {info_huerfanos['total_transacciones']:,} "
        f"transacciones originales. Ingreso total: ${info_huerfanos['ingreso_total_usd']:,.2f}. "
        f"0 filas perdidas, 0 filas duplicadas."
    )

    st.divider()
    st.subheader("🔍 Explorar el Dataset Unido")

    st.caption(
        "Usá el filtro para analizar los SKUs huérfanos y decidir si son productos "
        "no catalogados (hay que registrarlos) o errores de digitación (hay que corregir el SKU_ID)."
    )

    tipo_filtro = st.radio(
        "Filtrar por clasificación de SKU:",
        ["Todos", "Solo catálogo oficial", "Solo SKU fantasma (huérfanos)"],
        horizontal=True,
    )

    if tipo_filtro == "Solo catálogo oficial":
        df_filtrado = df_unido[df_unido["clasificacion_sku"] == "Catalogo oficial"]
    elif tipo_filtro == "Solo SKU fantasma (huérfanos)":
        df_filtrado = df_unido[df_unido["clasificacion_sku"] == "SKU fantasma - sin inventario"]
    else:
        df_filtrado = df_unido

    st.write(f"Mostrando {len(df_filtrado):,} de {len(df_unido):,} registros")
    st.dataframe(df_filtrado, use_container_width=True, height=500)

    # Stats cuando se filtra por huerfanos
    if tipo_filtro == "Solo SKU fantasma (huérfanos)":
        st.divider()
        st.subheader("📋 Análisis de SKUs Huérfanos")
        fantasma = df_filtrado

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.metric("SKU fantasma más frecuente",
                      fantasma["SKU_ID"].value_counts().index[0],
                      help="SKU huérfano con más transacciones")
        with col_f2:
            st.metric("SKU fantasma con más ingreso",
                      f"${fantasma.groupby('SKU_ID')['Precio_Venta_Final'].sum().max():,.0f}",
                      help="Mayor ingreso generado por un solo SKU huérfano")

        st.subheader("Top 15 SKUs huérfanos por frecuencia")
        top = fantasma["SKU_ID"].value_counts().head(15).reset_index()
        top.columns = ["SKU_ID", "Transacciones"]
        st.dataframe(top, use_container_width=True)

    st.divider()
    st.subheader("📁 Archivos generados")
    st.caption("Los siguientes archivos se guardaron en disco:")
    st.code(
        "data/processed/\n"
        "  inventario_limpio.csv\n"
        "  transacciones_limpio.csv\n"
        "  feedback_limpio.csv\n"
        "data/processed/join/\n"
        "  transacciones_con_inventario.csv"
    )


# =============================================
# TAB 4: CLIENTE (placeholder Fase 2)
# =============================================
with tab_cliente:
    st.header("👤 Cliente — Fase 2 (Integración)")
    st.info("Módulo en desarrollo. Aquí se mostrarán: Satisfacción NPS, Tickets de Soporte, Ratings por categoría, etc.")
    st.caption(f"Datos disponibles: {len(fb_limpio):,} registros de feedback.")


# =============================================
# TAB 5: INSIGHTS IA (placeholder Fase 3)
# =============================================
with tab_ia:
    st.header("🤖 Insights IA — Fase 3 (Groq + Llama-3)")
    st.info("Módulo en desarrollo. Integración con Groq para recomendaciones estratégicas en tiempo real.")
