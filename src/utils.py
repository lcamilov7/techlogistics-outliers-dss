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
