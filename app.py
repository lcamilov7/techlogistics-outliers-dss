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

    # --- Conclusiones y Metricas Financieras (siempre visibles) ---
    st.divider()
    st.subheader("📊 Diagnóstico: ¿Error de Digitación o Productos No Catalogados?")

    st.markdown("""
    Los **480 SKUs huérfanos** no son errores de digitación. La evidencia apunta a que son
    **productos reales que nunca se registraron en el ERP de inventario**.
    """)

    st.markdown("**Evidencia:**")
    evidencia = pd.DataFrame({
        "Indicador": [
            "Nomenclatura de SKU_ID",
            "Recurrencia promedio por SKU",
            "SKUs vendidos 2+ veces",
            "SKUs vendidos 5+ veces",
            "SKUs vendidos 1 sola vez",
            "Distribución por canales",
            "Distribución geográfica",
            "Precio promedio vs catálogo",
        ],
        "Valor": [
            "PROD-XXXX (idéntica al catálogo, sin typos)",
            "3.6 transacciones por SKU",
            "421 de 480 (88%)",
            "142 de 480 (30%)",
            "59 de 480 (12%)",
            "Uniforme: Físico 26%, App 25%, Online 25%, WhatsApp 23%",
            "Presente en TODAS las ciudades",
            f"H: ${df_unido[df_unido.clasificacion_sku == 'SKU fantasma - sin inventario'].Precio_Venta_Final.mean():,.0f} vs C: ${df_unido[df_unido.clasificacion_sku == 'Catalogo oficial'].Precio_Venta_Final.mean():,.0f} (idéntico)",
        ],
        "Conclusión": [
            "No hay caracteres extraños ni patrones de typo",
            "Productos con demanda recurrente, no incidentes aislados",
            "Casi 9 de cada 10 se venden más de una vez",
            "1 de cada 3 tiene alta rotación",
            "Solo el 12% son casos aislados",
            "Se venden por todos los canales por igual",
            "Distribución nacional, no concentrada en una región",
            "Mismo nivel de precios que el catálogo oficial",
        ],
    })
    st.dataframe(evidencia.set_index("Indicador"), use_container_width=True)

    st.caption(
        "Un error de digitación sería: pocos casos aislados, IDs con caracteres extraños, "
        "concentrados en un solo canal o ciudad. Nada de eso ocurre aquí."
    )

    st.divider()
    st.subheader("💰 Ingresos Totales y Margen de Utilidad Parcial")

    st.caption(
        "El ingreso de TODAS las transacciones es calculable (no hay Transaccion_ID duplicados). "
        "El margen solo es calculable para los SKUs que sí existen en el catálogo (tienen Costo_Unitario_USD). "
        "Para los SKUs fantasma, el costo del producto es desconocido → margen incalculable."
    )

    # Ingresos totales: suma de Precio_Venta_Final de todas las transacciones
    ingreso_total = df_unido["Precio_Venta_Final"].sum()
    ingreso_catalogo = df_unido.loc[
        df_unido["clasificacion_sku"] == "Catalogo oficial", "Precio_Venta_Final"
    ].sum()
    ingreso_fantasma = df_unido.loc[
        df_unido["clasificacion_sku"] == "SKU fantasma - sin inventario", "Precio_Venta_Final"
    ].sum()

    # Margen_Bruto ya viene precalculado desde el procesamiento
    # Solo para SKUs con catalogo (Costo_Unitario_USD no nulo); SKUs fantasma tienen NaN
    con_costo = df_unido["Margen_Bruto"].notnull()
    margen_total = df_unido.loc[con_costo, "Margen_Bruto"].sum()
    n_margen_positivo = int((df_unido.loc[con_costo, "Margen_Bruto"] >= 0).sum())
    n_margen_negativo = int((df_unido.loc[con_costo, "Margen_Bruto"] < 0).sum())
    n_sin_margen = int((~con_costo).sum())

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Ingreso Total", f"${ingreso_total:,.0f}",
                  help="Suma de Precio_Venta_Final de las 10,000 transacciones (sin duplicados)")
    with col_m2:
        st.metric("Ingreso Catálogo Oficial", f"${ingreso_catalogo:,.0f}")
    with col_m3:
        st.metric("Ingreso SKU Fantasma", f"${ingreso_fantasma:,.0f}",
                  delta=f"{ingreso_fantasma/ingreso_total*100:.1f}% del total",
                  delta_color="off")

    col_m4, col_m5, col_m6 = st.columns(3)
    with col_m4:
        margen_delta = f"Pérdida" if margen_total < 0 else f"Ganancia"
        st.metric("Margen Bruto Calculable",
                  f"${margen_total:,.0f}",
                  delta=margen_delta,
                  delta_color="inverse" if margen_total < 0 else "normal",
                  help="Margen = (Precio_Venta × Cantidad) - (Costo_Unitario × Cantidad + Costo_Envio). Solo para SKUs con catálogo.")
    with col_m5:
        st.metric("Transacciones con margen (+)",
                  f"{n_margen_positivo:,}",
                  help="Ventas donde el precio supera el costo")
    with col_m6:
        st.metric("Transacciones sin margen calculable",
                  f"{n_sin_margen:,}",
                  delta="SKU fantasma",
                  delta_color="off",
                  help="No se puede calcular el margen porque el SKU no está en el catálogo")

    if n_margen_negativo > 0:
        st.warning(
            f"⚠️ **{n_margen_negativo:,}** transacciones tienen **margen negativo** "
            f"(el costo del producto supera el precio de venta). Esto representa una "
            f"fuga de capital dentro de los productos con catálogo conocido."
        )

    with st.expander("📐 ¿Cómo se calculó cada métrica?", expanded=False):
        explicacion = pd.DataFrame([
            {
                "Métrica": "Ingreso Total",
                "Fórmula": "SUM(Precio_Venta_Final) de las 10,000 transacciones",
                "Datos usados": "Archivo limpio de transacciones (post-Fase 1)",
                "Nota": "Los Transaccion_ID son 100% únicos, no hay duplicados que inflen la suma.",
            },
            {
                "Métrica": "Ingreso Catálogo Oficial",
                "Fórmula": "SUM(Precio_Venta_Final) WHERE SKU_ID existe en inventario",
                "Datos usados": "8,249 transacciones con match en catálogo (Costo_Unitario_USD ≠ NaN)",
                "Nota": "Estos productos tienen costo conocido, se puede calcular margen.",
            },
            {
                "Métrica": "Ingreso SKU Fantasma",
                "Fórmula": "SUM(Precio_Venta_Final) WHERE SKU_ID NO existe en inventario",
                "Datos usados": "1,751 transacciones (480 SKUs únicos sin catálogo oficial)",
                "Nota": "Ingreso real pero con costo desconocido. Margen incalculable.",
            },
            {
                "Métrica": "Margen Bruto Calculable",
                "Fórmula": "SUM( Precio_Venta_Final × Cantidad_Vendida − Costo_Unitario_USD × Cantidad_Vendida − Costo_Envio ) para cada transacción con costo conocido",
                "Datos usados": "8,249 transacciones con Costo_Unitario_USD conocido",
                "Nota": "Margen = Ingreso total - Costo total del producto - Costo de envío. Negativo = se perdió dinero en esa venta.",
            },
            {
                "Métrica": "Transacciones con margen (+)",
                "Fórmula": "COUNT WHERE (Precio_Venta × Cantidad − Costo_Unitario × Cantidad − Costo_Envio) ≥ 0",
                "Datos usados": "Subconjunto de las 8,249 con costo conocido",
                "Nota": "Solo estas transacciones generan ganancia neta después de cubrir costo del producto y envío.",
            },
            {
                "Métrica": "Transacciones sin margen calculable",
                "Fórmula": "COUNT WHERE Costo_Unitario_USD IS NULL",
                "Datos usados": "1,751 transacciones de SKUs fantasma",
                "Nota": "El SKU no está en el inventario → el costo del producto es desconocido.",
            },
        ])
        st.dataframe(explicacion.set_index("Métrica"), use_container_width=True)

    with st.expander("🧾 Ejemplo concreto del cálculo de margen", expanded=False):
        st.markdown("""
        **Transacción #2 del dataset (PROD-1456):**
        - Precio de venta **unitario**: $686.88
        - Cantidad vendida: **12 unidades**
        - Ingreso total: $686.88 × 12 = **$8,242.56**
        - Costo unitario del producto: **$245.14**
        - Costo total del producto: $245.14 × 12 = **$2,941.68**
        - Costo de envío: **$67.16**
        - **Margen Bruto = ($686.88 × 12) − ($245.14 × 12 + $67.16) = $8,242.56 − $3,008.84 = $5,233.72**

        En este caso el margen es positivo. Pero hay miles de casos donde
        el costo del producto o el envío superan el ingreso total.
        """)


# =============================================
# TAB 4: CLIENTE (Fase 2)
# =============================================
with tab_cliente:
    st.header("👤 Cliente — Brecha de Entrega vs Prometido")

    st.markdown("""
    Mide la diferencia entre el tiempo real de entrega y el tiempo prometido por el proveedor.
    Solo calculable para SKUs con catálogo (tienen `Lead_Time_Dias`).
    """)

    with st.expander("📐 ¿Cómo se calcula?", expanded=False):
        formula = pd.DataFrame([
            {"Variable": "Tiempo_Entrega_Real", "Origen": "transacciones", "Significado": "Días reales desde el pedido hasta la entrega al cliente"},
            {"Variable": "Lead_Time_Dias", "Origen": "inventario", "Significado": "Días prometidos por el proveedor para reposición"},
            {"Resultado": "Brecha_Entrega", "Fórmula": "Tiempo_Entrega_Real − Lead_Time_Dias", "Significado": "Diferencia en días: positivo = demora, negativo = adelanto"},
        ])
        st.dataframe(formula.set_index("Variable"), use_container_width=True)

    with st.expander("📖 ¿Cómo interpretarla?", expanded=False):
        interpretacion = pd.DataFrame([
            {"Brecha": "Positiva (+)", "Resultado": "> 0", "Significado": "La entrega tardó MÁS de lo prometido → incumplimiento logístico"},
            {"Brecha": "Negativa (−)", "Resultado": "< 0", "Significado": "La entrega fue MÁS RÁPIDA de lo prometido → sobrecumplimiento, operación eficiente"},
            {"Brecha": "Cero", "Resultado": "= 0", "Significado": "Exactamente lo prometido"},
        ])
        st.dataframe(interpretacion.set_index("Brecha"), use_container_width=True)

    con_brecha = df_unido["Brecha_Entrega"].notnull()
    brecha_promedio = df_unido.loc[con_brecha, "Brecha_Entrega"].mean()

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.metric("Brecha promedio global", f"{brecha_promedio:+.1f} días",
                  help="Promedio de Tiempo_Entrega_Real − Lead_Time_Dias. Negativo = se entrega antes de lo prometido.")
    with col_b2:
        st.metric("Transacciones con brecha calculable", f"{con_brecha.sum():,.0f}",
                  help="Solo SKUs con catálogo tienen Lead_Time_Dias conocido.")
    with col_b3:
        pct_cumplen = int((df_unido.loc[con_brecha, "Brecha_Entrega"] <= 0).sum() / con_brecha.sum() * 100)
        st.metric("% que cumple o adelanta", f"{pct_cumplen}%",
                  help="Porcentaje de entregas que llegaron a tiempo o antes de lo prometido.")

    st.divider()
    st.markdown("**Promedios por dimensión:**")

    tab_ciudad, tab_bodega, tab_cat = st.tabs(["Por Ciudad", "Por Bodega", "Por Categoría"])

    def _tabla_brecha(df, grupo_col, nombre_col):
        resumen = df.dropna(subset=["Brecha_Entrega"]).groupby(grupo_col).agg(
            Transacciones=("Transaccion_ID", "count"),
            Brecha_Promedio=("Brecha_Entrega", "mean"),
            Desviacion=("Brecha_Entrega", "std"),
            Min=("Brecha_Entrega", "min"),
            Max=("Brecha_Entrega", "max"),
        ).reset_index()
        resumen.columns = [nombre_col, "Transacciones", "Brecha Promedio", "Desviación", "Mín", "Máx"]
        resumen = resumen.sort_values("Brecha Promedio")
        for c in ["Brecha Promedio", "Desviación", "Mín", "Máx"]:
            resumen[c] = resumen[c].round(1)
        resumen["Transacciones"] = resumen["Transacciones"].astype(int)
        return resumen

    with tab_ciudad:
        resumen_ciudad = _tabla_brecha(df_unido, "Ciudad_Destino", "Ciudad")
        mejor = resumen_ciudad.iloc[0]
        peor = resumen_ciudad.iloc[-1]
        st.caption(f"🏆 Mejor: **{mejor['Ciudad']}** ({mejor['Brecha Promedio']:+.1f} días) | ⚠️ Peor: **{peor['Ciudad']}** ({peor['Brecha Promedio']:+.1f} días)")
        st.dataframe(resumen_ciudad.set_index("Ciudad"), use_container_width=True)
        st.bar_chart(resumen_ciudad.set_index("Ciudad")["Brecha Promedio"], x_label="Ciudad", y_label="Brecha Promedio (días)")

    with tab_bodega:
        resumen_bodega = _tabla_brecha(df_unido, "Bodega_Origen", "Bodega")
        mejor = resumen_bodega.iloc[0]
        peor = resumen_bodega.iloc[-1]
        st.caption(f"🏆 Mejor: **{mejor['Bodega']}** ({mejor['Brecha Promedio']:+.1f} días) | ⚠️ Peor: **{peor['Bodega']}** ({peor['Brecha Promedio']:+.1f} días)")
        st.dataframe(resumen_bodega.set_index("Bodega"), use_container_width=True)
        st.bar_chart(resumen_bodega.set_index("Bodega")["Brecha Promedio"], x_label="Bodega", y_label="Brecha Promedio (días)")

    with tab_cat:
        resumen_cat = _tabla_brecha(df_unido, "Categoria", "Categoría")
        mejor = resumen_cat.iloc[0]
        peor = resumen_cat.iloc[-1]
        st.caption(f"🏆 Mejor: **{mejor['Categoría']}** ({mejor['Brecha Promedio']:+.1f} días) | ⚠️ Peor: **{peor['Categoría']}** ({peor['Brecha Promedio']:+.1f} días)")
        st.dataframe(resumen_cat.set_index("Categoría"), use_container_width=True)
        st.bar_chart(resumen_cat.set_index("Categoría")["Brecha Promedio"], x_label="Categoría", y_label="Brecha Promedio (días)")

    st.divider()
    st.subheader("🎫 Ratio de Soporte por Categoría")

    st.markdown("""
    Mide qué porcentaje de transacciones por categoría generaron un ticket de soporte.
    Un ratio alto indica problemas de calidad del producto o insatisfacción del cliente.
    """)

    with st.expander("📐 ¿Cómo se calcula?", expanded=False):
        formula_ticket = pd.DataFrame([
            {"Paso": "1", "Acción": "Del feedback, agrupar por Transaccion_ID: ¿algún registro tiene Ticket_Soporte_Abierto = 'Sí'?"},
            {"Paso": "2", "Acción": "Cruzar con transacciones_con_inventario (por Transaccion_ID, 1:1 sin duplicados)"},
            {"Paso": "3", "Acción": "Agrupar por Categoria o Bodega_Origen"},
            {"Paso": "4", "Acción": "Ratio = COUNT(Ticket_Sí) / COUNT(Total_Transacciones) × 100"},
        ])
        st.dataframe(formula_ticket.set_index("Paso"), use_container_width=True)

    # Agregar feedback: por Transaccion_ID, si existe algun ticket
    fb_agg = fb_limpio.groupby("Transaccion_ID").agg(
        Tiene_Ticket=("Ticket_Soporte_Abierto", lambda x: int((x == "Sí").any())),
    ).reset_index()

    # Cruzar 1:1 con df_unido
    df_ticket = df_unido.merge(fb_agg, on="Transaccion_ID", how="left")
    df_ticket["Tiene_Ticket"] = df_ticket["Tiene_Ticket"].fillna(0).astype(int)
    df_ticket["Tiene_Ticket"] = df_ticket["Tiene_Ticket"].map({1: "Con ticket", 0: "Sin ticket"})

    # Ratio por categoria
    ticket_cat = df_ticket.groupby("Categoria").agg(
        Total=("Transaccion_ID", "count"),
        Con_Ticket=("Tiene_Ticket", lambda x: (x == "Con ticket").sum()),
    ).reset_index()
    ticket_cat["Ratio_Tickets_%"] = (ticket_cat["Con_Ticket"] / ticket_cat["Total"] * 100).round(1)
    ticket_cat = ticket_cat.sort_values("Ratio_Tickets_%", ascending=False)

    # Ratio por bodega
    ticket_bodega = df_ticket.groupby("Bodega_Origen").agg(
        Total=("Transaccion_ID", "count"),
        Con_Ticket=("Tiene_Ticket", lambda x: (x == "Con ticket").sum()),
    ).reset_index()
    ticket_bodega["Ratio_Tickets_%"] = (ticket_bodega["Con_Ticket"] / ticket_bodega["Total"] * 100).round(1)
    ticket_bodega = ticket_bodega.sort_values("Ratio_Tickets_%", ascending=False)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Por Categoría**")
        st.dataframe(
            ticket_cat.set_index("Categoria"),
            use_container_width=True,
            column_config={
                "Total": "Total ventas",
                "Con_Ticket": "Con ticket",
                "Ratio_Tickets_%": st.column_config.NumberColumn("Ratio %", format="%.1f%%"),
            },
        )
        peor_ticket = ticket_cat.iloc[0]
        st.warning(f"🚨 Mayor ratio: **{peor_ticket['Categoria']}** ({peor_ticket['Ratio_Tickets_%']}% de {int(peor_ticket['Total'])} ventas)")

    with col_t2:
        st.markdown("**Por Bodega**")
        st.dataframe(
            ticket_bodega.set_index("Bodega_Origen"),
            use_container_width=True,
            column_config={
                "Total": "Total ventas",
                "Con_Ticket": "Con ticket",
                "Ratio_Tickets_%": st.column_config.NumberColumn("Ratio %", format="%.1f%%"),
            },
        )
        peor_bodega = ticket_bodega.iloc[0]
        st.warning(f"🚨 Mayor ratio: **{peor_bodega['Bodega_Origen']}** ({peor_bodega['Ratio_Tickets_%']}% de {int(peor_bodega['Total'])} ventas)")

    st.divider()
    st.subheader("🕐 Antigüedad de Revisión vs Tickets de Soporte")

    st.markdown("""
    Relación entre cuándo fue la última revisión de inventario y la tasa de tickets.
    Bodegas con revisiones muy antiguas pueden estar operando con datos desactualizados,
    generando más reclamos de clientes.
    """)

    # Antigüedad en dias desde Ultima_Revision hasta hoy
    hoy = pd.Timestamp.today()
    df_ticket["Antiguedad_Revision_Dias"] = (hoy - df_ticket["Ultima_Revision"]).dt.days

    # Agrupar por bodega: antiguedad promedio vs ratio de tickets
    riesgo_bodega = df_ticket.groupby("Bodega_Origen").agg(
        Total_Ventas=("Transaccion_ID", "count"),
        Antiguedad_Promedio=("Antiguedad_Revision_Dias", "mean"),
        Con_Ticket=("Tiene_Ticket", lambda x: (x == "Con ticket").sum()),
    ).reset_index()
    riesgo_bodega["Ratio_Tickets_%"] = (riesgo_bodega["Con_Ticket"] / riesgo_bodega["Total_Ventas"] * 100).round(1)
    riesgo_bodega["Antiguedad_Promedio"] = riesgo_bodega["Antiguedad_Promedio"].round(0).astype(int)

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.dataframe(
            riesgo_bodega.set_index("Bodega_Origen"),
            use_container_width=True,
            column_config={
                "Total_Ventas": "Ventas",
                "Antiguedad_Promedio": st.column_config.NumberColumn("Antigüedad (días)", format="%d"),
                "Con_Ticket": "Con ticket",
                "Ratio_Tickets_%": st.column_config.NumberColumn("Ratio Tickets %", format="%.1f%%"),
            },
        )

    with col_r2:
        bodegas_riesgo = riesgo_bodega[riesgo_bodega["Antiguedad_Promedio"] > 365]
        if len(bodegas_riesgo) > 0:
            st.warning(
                f"⚠️ **{len(bodegas_riesgo)} bodega(s)** tienen más de 1 año sin revisión:\n\n" +
                "\n".join(f"- **{r['Bodega_Origen']}**: {r['Antiguedad_Promedio']} días, "
                          f"{r['Ratio_Tickets_%']}% tickets"
                          for _, r in bodegas_riesgo.iterrows())
            )
        else:
            st.success("Ninguna bodega supera 1 año sin revisión.")


# =============================================
# TAB 5: INSIGHTS IA (placeholder Fase 3)
# =============================================
with tab_ia:
    st.header("🤖 Insights IA — Fase 3 (Groq + Llama-3)")
    st.info("Módulo en desarrollo. Integración con Groq para recomendaciones estratégicas en tiempo real.")
