"""
Modulo de auditoria de calidad (Fase 1).
Calcula Health Score, metricas de nulidad, duplicados y outliers por dataset.
"""
import pandas as pd
import numpy as np
from .utils import analizar_duplicados


def _metricas_base(df: pd.DataFrame, nombre: str) -> dict:
    total_filas = len(df)
    total_celdas = df.size
    celdas_nulas = int(df.isnull().sum().sum())

    pct_nulidad = (celdas_nulas / total_celdas * 100) if total_celdas else 0

    return {
        "dataset": nombre,
        "total_filas": total_filas,
        "total_celdas": total_celdas,
        "celdas_nulas": celdas_nulas,
        "pct_nulidad_global": round(pct_nulidad, 2),
    }


def _pct_nulidad_por_columna(df: pd.DataFrame) -> dict:
    nulos = df.isnull().sum()
    total = len(df)
    pct = (nulos / total * 100).round(2) if total else pd.Series(dtype=float)
    return pct.to_dict()


def _health_score(df: pd.DataFrame) -> float:
    """Score 0-100 donde 100 = datos perfectos."""
    total_celdas = df.size
    if total_celdas == 0:
        return 0.0
    celdas_nulas = int(df.isnull().sum().sum())
    filas_dup_exactas = int(df.duplicated().sum())
    penalizacion_nulos = (celdas_nulas / total_celdas) * 100
    penalizacion_dup = (filas_dup_exactas / max(len(df), 1)) * 10
    score = 100 - penalizacion_nulos - penalizacion_dup
    return max(0.0, round(score, 2))


def _detectar_outliers_dominio(df: pd.DataFrame, columna: str, lim_inf: float, lim_sup: float) -> dict:
    """
    Detecta valores fuera de un rango de dominio conocido (ej: rating 1-5, edad 0-120).
    No usa IQR; usa los limites semanticos de la variable.
    """
    serie = df[columna].dropna()
    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]
    cantidad = len(outliers)

    return {
        "columna": columna,
        "metodo": "dominio",
        "tiene_outliers": cantidad > 0,
        "cantidad": cantidad,
        "pct": round(cantidad / max(len(df), 1) * 100, 2),
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "dominio_esperado": f"[{lim_inf}, {lim_sup}]",
    }


def _detectar_outliers_iqr(df: pd.DataFrame, columna: str, factor_iqr: float = 1.5,
                           lim_inf_natural: float = None) -> dict:
    """
    Detecta dos tipos de anomalias por separado:
      1. Violaciones de dominio: valores por debajo del limite natural
         (ej: cantidad negativa, stock negativo, costo $0).
      2. Outliers superiores IQR: valores por encima de Q3 + factor*IQR.
    Las violaciones de dominio son errores de captura; los outliers superiores
    son valores estadisticamente atipicos pero potencialmente legitimos.
    """
    serie = df[columna].dropna()
    vacio = {"columna": columna, "metodo": "iqr", "tiene_outliers": False,
             "violaciones_dominio": 0, "outliers_superiores_iqr": 0,
             "cantidad": 0, "pct": 0.0, "q1": None, "q3": None,
             "lim_inf_natural": lim_inf_natural, "lim_sup_iqr": None}

    if len(serie) == 0:
        return vacio

    col = df[columna]

    # 1. Violaciones de dominio (por debajo del limite natural)
    n_violaciones = 0
    if lim_inf_natural is not None:
        violaciones = col < lim_inf_natural
        n_violaciones = int(violaciones.sum())

    # 2. Outliers superiores via IQR
    q1 = float(serie.quantile(0.25))
    q3 = float(serie.quantile(0.75))
    iqr = q3 - q1
    lim_sup_iqr = q3 + factor_iqr * iqr
    outliers_sup = col > lim_sup_iqr
    n_outliers_sup = int(outliers_sup.sum())

    total = n_violaciones + n_outliers_sup
    pct = round(total / max(len(df), 1) * 100, 2)

    return {
        "columna": columna,
        "metodo": "iqr",
        "tiene_outliers": total > 0,
        "violaciones_dominio": n_violaciones,
        "outliers_superiores_iqr": n_outliers_sup,
        "cantidad": total,
        "pct": pct,
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "lim_inf_natural": lim_inf_natural,
        "lim_sup_iqr": round(lim_sup_iqr, 2),
    }


def auditoria_completa(inv_raw, trx_raw, fb_raw) -> dict:
    """Auditoria de calidad antes de cualquier limpieza."""

    # --- Inventario ---
    inv_dup = analizar_duplicados(inv_raw, "SKU_ID")
    inv_metricas = _metricas_base(inv_raw, "Inventario")
    inv_metricas["duplicados_exactos"] = inv_dup["duplicados_exactos"]
    inv_metricas["ids_repetidos_datos_diferentes"] = inv_dup["ids_repetidos_datos_diferentes"]
    inv_nulos = _pct_nulidad_por_columna(inv_raw)
    inv_health = _health_score(inv_raw)
    inv_outliers = {
        "Stock_Actual": _detectar_outliers_iqr(inv_raw, "Stock_Actual", lim_inf_natural=0),
        "Costo_Unitario_USD": _detectar_outliers_iqr(inv_raw, "Costo_Unitario_USD", lim_inf_natural=0.01),
    }
    inv_categ_inconsistentes = {
        "Categoria": inv_raw["Categoria"].value_counts().to_dict(),
        "Bodega_Origen": inv_raw["Bodega_Origen"].value_counts().to_dict(),
    }

    auditoria_inv = {
        "metricas": inv_metricas,
        "health_score_antes": inv_health,
        "pct_nulidad_por_columna": inv_nulos,
        "outliers_detectados": inv_outliers,
        "categ_inconsistentes": inv_categ_inconsistentes,
        "analisis_duplicados": inv_dup,
    }

    # --- Transacciones ---
    trx_dup = analizar_duplicados(trx_raw, "Transaccion_ID")
    trx_metricas = _metricas_base(trx_raw, "Transacciones")
    trx_metricas["duplicados_exactos"] = trx_dup["duplicados_exactos"]
    trx_metricas["ids_repetidos_datos_diferentes"] = trx_dup["ids_repetidos_datos_diferentes"]
    trx_nulos = _pct_nulidad_por_columna(trx_raw)
    trx_health = _health_score(trx_raw)
    trx_outliers = {
        "Cantidad_Vendida": _detectar_outliers_iqr(trx_raw, "Cantidad_Vendida", lim_inf_natural=0),
        "Precio_Venta_Final": _detectar_outliers_iqr(trx_raw, "Precio_Venta_Final", lim_inf_natural=0),
        "Costo_Envio": _detectar_outliers_iqr(trx_raw, "Costo_Envio", lim_inf_natural=0),
        "Tiempo_Entrega_Real": _detectar_outliers_dominio(trx_raw, "Tiempo_Entrega_Real", 0, 365),
    }
    trx_categ_inconsistentes = {
        "Ciudad_Destino": trx_raw["Ciudad_Destino"].value_counts().to_dict(),
        "Estado_Envio": trx_raw["Estado_Envio"].value_counts().to_dict(),
    }

    auditoria_trx = {
        "metricas": trx_metricas,
        "health_score_antes": trx_health,
        "pct_nulidad_por_columna": trx_nulos,
        "outliers_detectados": trx_outliers,
        "categ_inconsistentes": trx_categ_inconsistentes,
        "analisis_duplicados": trx_dup,
    }

    # --- Feedback ---
    fb_dup = analizar_duplicados(fb_raw, "Feedback_ID")
    fb_metricas = _metricas_base(fb_raw, "Feedback")
    fb_metricas["duplicados_exactos"] = fb_dup["duplicados_exactos"]
    fb_metricas["ids_repetidos_datos_diferentes"] = fb_dup["ids_repetidos_datos_diferentes"]
    fb_nulos = _pct_nulidad_por_columna(fb_raw)
    fb_health = _health_score(fb_raw)
    fb_outliers = {
        "Rating_Producto": _detectar_outliers_dominio(fb_raw, "Rating_Producto", 1, 5),
        "Rating_Logistica": _detectar_outliers_dominio(fb_raw, "Rating_Logistica", 1, 5),
        "Edad_Cliente": _detectar_outliers_dominio(fb_raw, "Edad_Cliente", 0, 120),
        "Satisfaccion_NPS": _detectar_outliers_dominio(fb_raw, "Satisfaccion_NPS", -100, 100),
    }

    auditoria_fb = {
        "metricas": fb_metricas,
        "health_score_antes": fb_health,
        "pct_nulidad_por_columna": fb_nulos,
        "outliers_detectados": fb_outliers,
        "analisis_duplicados": fb_dup,
    }

    return {
        "inventario": auditoria_inv,
        "transacciones": auditoria_trx,
        "feedback": auditoria_fb,
    }
