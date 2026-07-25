"""
Utilidades compartidas: carga de datos, normalización de texto, manejo de fechas.
"""
import pandas as pd
import re
from datetime import datetime


DATA_PATH = "data"
ENCODING = "latin-1"


def cargar_inventario() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/inventario_central_v2.csv", encoding=ENCODING)


def cargar_transacciones() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/transacciones_logistica_v2.csv", encoding=ENCODING)


def cargar_feedback() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/feedback_clientes_v2.csv", encoding=ENCODING)


def cargar_todo():
    return cargar_inventario(), cargar_transacciones(), cargar_feedback()


def parsear_fecha(serie: pd.Series) -> pd.Series:
    return pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")


# --- Normalización de ciudad ---
CIUDAD_MAPEO = {
    "Bogotá": "Bogotá",
    "Bogota": "Bogotá",
    "bogotá": "Bogotá",
    "bogota": "Bogotá",
    "BOG": "Bogotá",
    "bog": "Bogotá",
    "Medellín": "Medellín",
    "Medellin": "Medellín",
    "medellín": "Medellín",
    "medellin": "Medellín",
    "MED": "Medellín",
    "med": "Medellín",
    "Cali": "Cali",
    "cali": "Cali",
    "CAL": "Cali",
    "Barranquilla": "Barranquilla",
    "barranquilla": "Barranquilla",
    "BAQ": "Barranquilla",
    "Cartagena": "Cartagena",
    "cartagena": "Cartagena",
    "CTG": "Cartagena",
    "Bucaramanga": "Bucaramanga",
    "bucaramanga": "Bucaramanga",
    "BGA": "Bucaramanga",
    "Ventas_Web": "Ventas_Web",
}


def normalizar_ciudad(valor):
    if pd.isna(valor):
        return valor
    val = str(valor).strip()
    return CIUDAD_MAPEO.get(val, val)


# --- Normalización de categoría ---
CATEGORIA_MAPEO = {
    "smart-phone": "Smartphones",
    "Smartphones": "Smartphones",
    "smartphones": "Smartphones",
    "LAPTOP": "Laptops",
    "Laptops": "Laptops",
    "laptops": "Laptops",
    "Accesorios": "Accesorios",
    "accesorios": "Accesorios",
    "Monitores": "Monitores",
    "monitores": "Monitores",
    "Tablets": "Tablets",
    "tablets": "Tablets",
}


def normalizar_categoria(valor):
    if pd.isna(valor):
        return valor
    val = str(valor).strip()
    return CATEGORIA_MAPEO.get(val, val)


# --- Normalización de bodega ---
BODEGA_MAPEO = {
    "Norte": "Norte",
    "norte": "Norte",
    "Sur": "Sur",
    "sur": "Sur",
    "Occidente": "Occidente",
    "occidente": "Occidente",
    "ZONA_FRANCA": "Zona_Franca",
    "zona_franca": "Zona_Franca",
    "BOD-EXT-99": "BOD-EXT-99",
}


def normalizar_bodega(valor):
    if pd.isna(valor):
        return valor
    val = str(valor).strip()
    return BODEGA_MAPEO.get(val, val)


# --- Normalización de Ticket_Soporte_Abierto ---
def normalizar_ticket(valor):
    if pd.isna(valor):
        return valor
    val = str(valor).strip().lower()
    if val in ("sí", "si", "1"):
        return "Sí"
    elif val in ("no", "0"):
        return "No"
    return valor


# --- Duplicados inteligentes ---
def analizar_duplicados(df, columna_id=None):
    """
    Analiza duplicados en un DataFrame distinguiendo dos tipos:
      - Duplicados exactos: toda la fila es idéntica (error de sistema, se eliminan).
      - IDs repetidos con datos diferentes: mismo ID pero distinta informacion
        (inconsistencia de ID, los registros se CONSERVAN pero se reportan).

    Retorna dict con conteos y mascara de filas a eliminar (exactos).
    """
    duplicados_exactos = df.duplicated(keep=False)
    total_exactos = int(duplicados_exactos.sum())
    filas_a_eliminar = df.duplicated(keep="first")

    resultado = {
        "duplicados_exactos": total_exactos,
        "filas_a_eliminar": filas_a_eliminar,
        "ids_repetidos_datos_diferentes": 0,
        "columna_id": columna_id,
    }

    if columna_id and columna_id in df.columns:
        dups_id = df[df.duplicated(subset=columna_id, keep=False)]
        if len(dups_id) > 0:
            diff = 0
            for _, group in dups_id.groupby(columna_id):
                if len(group) > 1:
                    cols = [c for c in group.columns if c != columna_id]
                    if group[cols].nunique().max() > 1:
                        diff += len(group)
            resultado["ids_repetidos_datos_diferentes"] = diff

    return resultado


def eliminar_duplicados_exactos(df):
    """Elimina solo las filas completamente identicas (duplicados exactos)."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


# --- Normalización de Lead_Time_Dias ---
def extraer_lead_time_numerico(valor):
    """Extrae el valor numérico de Lead_Time_Dias (ej: '25-30 días' -> 27.5, 'Inmediato' -> 0)."""
    if pd.isna(valor):
        return None
    texto = str(valor).strip().lower()
    if texto in ("inmediato", "inmediata", "inmediatamente"):
        return 0.0
    numeros = re.findall(r"\d+\.?\d*", texto)
    if not numeros:
        return None
    nums = [float(n) for n in numeros]
    return sum(nums) / len(nums)
