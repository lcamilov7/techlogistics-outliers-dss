"""
Modulo de limpieza de datos (Fase 1).
Aplica transformaciones, imputaciones y eliminacion de outliers.
Cada funcion devuelve (df_limpio, resumen_cambios) para trazabilidad.
"""
import pandas as pd
import numpy as np
from .utils import (
    normalizar_ciudad, normalizar_categoria, normalizar_bodega,
    normalizar_ticket, extraer_lead_time_numerico, parsear_fecha,
    analizar_duplicados, eliminar_duplicados_exactos,
)


def limpiar_inventario(df: pd.DataFrame) -> tuple:
    cambios = {"filas_inicial": len(df), "filas_final": 0, "acciones": []}

    # 0. Eliminar solo duplicados exactos (toda la fila identica)
    dup_info = analizar_duplicados(df, "SKU_ID")
    n_exactos = dup_info["duplicados_exactos"]
    if n_exactos > 0:
        df = eliminar_duplicados_exactos(df)
        cambios["acciones"].append(f"Duplicados exactos eliminados: {n_exactos} filas")
    if dup_info["ids_repetidos_datos_diferentes"] > 0:
        cambios["acciones"].append(
            f"ALERTA: {dup_info['ids_repetidos_datos_diferentes']} registros con SKU_ID repetido "
            f"pero datos diferentes (CONSERVADOS -- posible inconsistencia de ID)"
        )

    # 1. Normalizar categorias y bodegas
    df["Categoria"] = df["Categoria"].apply(normalizar_categoria)
    cat_fantasma = (df["Categoria"] == "???")
    n_fantasma = cat_fantasma.sum()
    if n_fantasma > 0:
        modal = df.loc[~cat_fantasma, "Categoria"].mode()
        if not modal.empty:
            df.loc[cat_fantasma, "Categoria"] = modal[0]
            cambios["acciones"].append(f"Categoria '???' imputada con moda '{modal[0]}': {n_fantasma} filas")
        else:
            cambios["acciones"].append(f"Categoria '???' sin moda disponible: {n_fantasma} filas")

    df["Bodega_Origen"] = df["Bodega_Origen"].apply(normalizar_bodega)

    # 2. Stock negativo -> 0 (error de sistema, no puede haber stock fisico negativo)
    neg_stock = df["Stock_Actual"] < 0
    n_neg = neg_stock.sum()
    if n_neg > 0:
        df.loc[neg_stock, "Stock_Actual"] = 0
        cambios["acciones"].append(f"Stock negativo imputado a 0: {n_neg} filas")

    # 3. Costos atipicos (outliers) -> marcar con mediana por categoria
    for cat in df["Categoria"].unique():
        mask_cat = df["Categoria"] == cat
        serie = df.loc[mask_cat, "Costo_Unitario_USD"].dropna()
        if len(serie) == 0:
            continue
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        lim_sup = q3 + 1.5 * iqr
        outliers = mask_cat & (df["Costo_Unitario_USD"] > lim_sup)
        n_out = outliers.sum()
        if n_out > 0:
            mediana_cat = serie.median()
            df.loc[outliers, "Costo_Unitario_USD"] = mediana_cat
            cambios["acciones"].append(
                f"Costo atipico '{cat}' capado a mediana (${mediana_cat:,.2f}): {n_out} filas"
            )

    # 4. Lead_Time_Dias: extraer numerico, imputar faltantes con mediana
    df["Lead_Time_Dias"] = df["Lead_Time_Dias"].apply(extraer_lead_time_numerico)
    nulos_lead = df["Lead_Time_Dias"].isnull().sum()
    if nulos_lead > 0:
        mediana_lead = df["Lead_Time_Dias"].median()
        df["Lead_Time_Dias"] = df["Lead_Time_Dias"].fillna(mediana_lead)
        cambios["acciones"].append(f"Lead_Time_Dias nulos imputados con mediana ({mediana_lead}): {nulos_lead} filas")

    # 5. Stock_Actual nulos -> imputar con mediana por categoria
    nulos_stock = df["Stock_Actual"].isnull().sum()
    if nulos_stock > 0:
        df["Stock_Actual"] = df.groupby("Categoria")["Stock_Actual"].transform(
            lambda x: x.fillna(x.median())
        )
        cambios["acciones"].append(f"Stock_Actual nulos imputados con mediana por categoria: {nulos_stock} filas")

    # 6. Ultima_Revision como fecha
    df["Ultima_Revision"] = parsear_fecha(df["Ultima_Revision"])

    cambios["filas_final"] = len(df)
    return df, cambios


def limpiar_transacciones(df: pd.DataFrame) -> tuple:
    cambios = {"filas_inicial": len(df), "filas_final": 0, "acciones": []}

    # 0. Eliminar solo duplicados exactos (toda la fila identica)
    dup_info = analizar_duplicados(df, "Transaccion_ID")
    n_exactos = dup_info["duplicados_exactos"]
    if n_exactos > 0:
        df = eliminar_duplicados_exactos(df)
        cambios["acciones"].append(f"Duplicados exactos eliminados: {n_exactos} filas")
    if dup_info["ids_repetidos_datos_diferentes"] > 0:
        cambios["acciones"].append(
            f"ALERTA: {dup_info['ids_repetidos_datos_diferentes']} registros con Transaccion_ID repetido "
            f"pero datos diferentes (CONSERVADOS -- posible inconsistencia de ID)"
        )

    # 1. Fecha_Venta a datetime
    df["Fecha_Venta"] = parsear_fecha(df["Fecha_Venta"])
    n_fecha_inv = df["Fecha_Venta"].isnull().sum()
    if n_fecha_inv > 0:
        cambios["acciones"].append(f"Fechas invalidas eliminadas: {n_fecha_inv} filas")

    # 2. Cantidad_Vendida negativa: imputar con mediana por SKU (error de captura)
    neg_cant = df["Cantidad_Vendida"] < 0
    n_neg = neg_cant.sum()
    if n_neg > 0:
        df.loc[neg_cant, "Cantidad_Vendida"] = np.nan
        df["Cantidad_Vendida"] = df.groupby("SKU_ID")["Cantidad_Vendida"].transform(
            lambda x: x.fillna(x.median())
        )
        aun_nulos = df["Cantidad_Vendida"].isnull().sum()
        if aun_nulos > 0:
            mediana_global = df["Cantidad_Vendida"].median()
            df["Cantidad_Vendida"] = df["Cantidad_Vendida"].fillna(mediana_global)
        cambios["acciones"].append(
            f"Cantidad_Vendida negativa imputada con mediana: {n_neg} filas"
        )

    # 3. Tiempo_Entrega_Real outliers (> percentil 99): capar
    p99 = df["Tiempo_Entrega_Real"].quantile(0.99)
    out_tiempo = df["Tiempo_Entrega_Real"] > p99
    n_out = out_tiempo.sum()
    if n_out > 0:
        df.loc[out_tiempo, "Tiempo_Entrega_Real"] = int(p99)
        cambios["acciones"].append(
            f"Tiempo_Entrega_Real > P99 ({p99:.0f}) capado: {n_out} filas"
        )

    # 4. Costo_Envio nulos: imputar con mediana por Canal_Venta
    nulos_envio = df["Costo_Envio"].isnull().sum()
    if nulos_envio > 0:
        df["Costo_Envio"] = df.groupby("Canal_Venta")["Costo_Envio"].transform(
            lambda x: x.fillna(x.median())
        )
        aun_nulos = df["Costo_Envio"].isnull().sum()
        if aun_nulos > 0:
            df["Costo_Envio"] = df["Costo_Envio"].fillna(df["Costo_Envio"].median())
        cambios["acciones"].append(f"Costo_Envio nulos imputados con mediana: {nulos_envio} filas")

    # 5. Normalizar ciudades
    df["Ciudad_Destino"] = df["Ciudad_Destino"].apply(normalizar_ciudad)

    cambios["filas_final"] = len(df)
    return df, cambios


def limpiar_feedback(df: pd.DataFrame) -> tuple:
    cambios = {"filas_inicial": len(df), "filas_final": 0, "acciones": []}

    # 0. Analisis de duplicados: solo eliminar exactos, conservar IDs repetidos con datos diferentes
    dup_info = analizar_duplicados(df, "Feedback_ID")
    n_exactos = dup_info["duplicados_exactos"]
    if n_exactos > 0:
        df = eliminar_duplicados_exactos(df)
        cambios["acciones"].append(f"Duplicados exactos eliminados: {n_exactos} filas")
    if dup_info["ids_repetidos_datos_diferentes"] > 0:
        cambios["acciones"].append(
            f"ALERTA: {dup_info['ids_repetidos_datos_diferentes']} registros con Feedback_ID repetido "
            f"pero datos diferentes (CONSERVADOS -- posible inconsistencia de ID en el sistema de feedback)"
        )

    # 1. Rating_Producto fuera de 1-5 -> marcar con NaN e imputar con mediana
    mask_rp = (df["Rating_Producto"] < 1) | (df["Rating_Producto"] > 5)
    n_rp = mask_rp.sum()
    if n_rp > 0:
        df.loc[mask_rp, "Rating_Producto"] = np.nan
        df["Rating_Producto"] = df["Rating_Producto"].fillna(round(df["Rating_Producto"].median()))
        cambios["acciones"].append(f"Rating_Producto fuera de rango corregido: {n_rp} filas")

    # 3. Edad_Cliente > 100 -> imputar con mediana (error de captura)
    mask_edad = df["Edad_Cliente"] > 100
    n_edad = mask_edad.sum()
    if n_edad > 0:
        df.loc[mask_edad, "Edad_Cliente"] = np.nan
        df["Edad_Cliente"] = df["Edad_Cliente"].fillna(round(df["Edad_Cliente"].median()))
        cambios["acciones"].append(f"Edad_Cliente > 100 imputada con mediana: {n_edad} filas")

    # 4. Normalizar Ticket_Soporte_Abierto
    df["Ticket_Soporte_Abierto"] = df["Ticket_Soporte_Abierto"].apply(normalizar_ticket)

    # 5. Comentario_Texto: los que son '---', 'N/A', 'nan' -> None
    mask_vacio = df["Comentario_Texto"].isin(["---", "N/A", "nan", "NaN", ""])
    df.loc[mask_vacio, "Comentario_Texto"] = np.nan

    # 6. Recomienda_Marca nulos: rellenar con 'No especifica'
    n_rec = df["Recomienda_Marca"].isnull().sum()
    if n_rec > 0:
        df["Recomienda_Marca"] = df["Recomienda_Marca"].fillna("No especifica")
        cambios["acciones"].append(f"Recomienda_Marca nulos imputados con 'No especifica': {n_rec} filas")

    # 7. Comentario_Texto nulos: mantener como NaN (no se imputa texto)
    cambios["filas_final"] = len(df)
    return df, cambios


def limpieza_completa(inv_raw, trx_raw, fb_raw) -> dict:
    inv_limpio, cambios_inv = limpiar_inventario(inv_raw.copy())
    trx_limpio, cambios_trx = limpiar_transacciones(trx_raw.copy())
    fb_limpio, cambios_fb = limpiar_feedback(fb_raw.copy())

    return {
        "inventario": {"df": inv_limpio, "cambios": cambios_inv},
        "transacciones": {"df": trx_limpio, "cambios": cambios_trx},
        "feedback": {"df": fb_limpio, "cambios": cambios_fb},
    }
