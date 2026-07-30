"""
Modulo de procesamiento (Fase 2).
Guarda los DataFrames limpios en data/processed/ y genera
la fuente unica de verdad (left join transacciones + inventario).
"""
import os
import pandas as pd
import numpy as np


PROCESSED_DIR = "data/processed"
JOIN_DIR = os.path.join(PROCESSED_DIR, "join")


def guardar_limpios(inv_limpio, trx_limpio, fb_limpio):
    """Guarda los 3 DataFrames limpios (post-Fase 1) en data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    inv_limpio.to_csv(os.path.join(PROCESSED_DIR, "inventario_limpio.csv"), index=False, encoding="utf-8")
    trx_limpio.to_csv(os.path.join(PROCESSED_DIR, "transacciones_limpio.csv"), index=False, encoding="utf-8")
    fb_limpio.to_csv(os.path.join(PROCESSED_DIR, "feedback_limpio.csv"), index=False, encoding="utf-8")


def crear_fuente_verdad(trx_limpio, inv_limpio):
    """
    Left join estrategico: transacciones + inventario sobre SKU_ID.
    - Se usa left para NO perder ninguna venta.
    - Columnas de inventario quedan NaN para SKUs sin catalogo.
    - Se agrega columna 'clasificacion_sku' para identificar huerfanos.
    Retorna el DataFrame unido.
    """
    df = trx_limpio.merge(
        inv_limpio,
        on="SKU_ID",
        how="left",
        suffixes=("", "_inv"),
    )

    # Eliminar columna duplicada de merge indicator si existe
    cols_dup = [c for c in df.columns if c.endswith("_inv")]
    # Ya usamos suffixes, no hay _merge

    df["clasificacion_sku"] = np.where(
        df["Categoria"].isnull(),
        "SKU fantasma - sin inventario",
        "Catalogo oficial",
    )

    # Margen_Bruto: solo calculable para SKUs con catalogo
    # Margen = (Precio_Venta × Cantidad) - (Costo_Unitario × Cantidad + Costo_Envio)
    tiene_costo = df["Costo_Unitario_USD"].notnull()
    df["Margen_Bruto"] = np.nan
    df.loc[tiene_costo, "Margen_Bruto"] = (
        df.loc[tiene_costo, "Precio_Venta_Final"] * df.loc[tiene_costo, "Cantidad_Vendida"]
        - (df.loc[tiene_costo, "Costo_Unitario_USD"] * df.loc[tiene_costo, "Cantidad_Vendida"]
           + df.loc[tiene_costo, "Costo_Envio"])
    )

    return df


def guardar_fuente_verdad(df_unido):
    """Guarda el left join en data/processed/join/."""
    os.makedirs(JOIN_DIR, exist_ok=True)
    df_unido.to_csv(os.path.join(JOIN_DIR, "transacciones_con_inventario.csv"), index=False, encoding="utf-8")


def cargar_fuente_verdad():
    """Carga el left join desde data/processed/join/ si existe."""
    ruta = os.path.join(JOIN_DIR, "transacciones_con_inventario.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta, encoding="utf-8")
    return None


def resumen_huerfanos(df_unido):
    """Resumen de SKUs huerfanos."""
    fantasma = df_unido["clasificacion_sku"] == "SKU fantasma - sin inventario"
    total = len(df_unido)
    n_huerf = int(fantasma.sum())
    skus_unicos = int(df_unido.loc[fantasma, "SKU_ID"].nunique())
    ingreso_total = df_unido["Precio_Venta_Final"].sum()
    ingreso_huerf = df_unido.loc[fantasma, "Precio_Venta_Final"].sum()

    return {
        "total_transacciones": total,
        "transacciones_huerfanas": n_huerf,
        "skus_huerfanos_unicos": skus_unicos,
        "pct_huerfanas": round(n_huerf / total * 100, 1),
        "ingreso_total_usd": round(float(ingreso_total), 2),
        "ingreso_huerfano_usd": round(float(ingreso_huerf), 2),
        "pct_ingreso_en_riesgo": round(ingreso_huerf / ingreso_total * 100, 1),
    }
