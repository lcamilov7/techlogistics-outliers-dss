"""
Utilidades compartidas: carga de datos, normalización de texto, manejo de fechas.
"""
import pandas as pd
import re
import difflib
from datetime import datetime


DATA_PATH = "data"
ENCODING = "latin-1"


def _limpiar_texto(texto: str) -> str:
    """Lowercase, quitar tildes basicas, eliminar caracteres no alfanumericos."""
    t = texto.lower().strip()
    t = t.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


def _fuzzy_match(clave, candidatos, umbral=0.80):
    """
    Busca el mejor match entre 'clave' y las claves de 'candidatos' usando
    similitud de secuencia (difflib). Retorna el valor canónico si supera el umbral.
    """
    mejor = None
    mejor_score = 0.0
    for canon_key, canon_val in candidatos.items():
        score = difflib.SequenceMatcher(None, clave, canon_key).ratio()
        if score > mejor_score:
            mejor_score = score
            mejor = canon_val
    return mejor if mejor_score >= umbral else None


def cargar_inventario() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/inventario_central_v2.csv", encoding=ENCODING)


def cargar_transacciones() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/transacciones_logistica_v2.csv", encoding=ENCODING)


def cargar_feedback() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/feedback_clientes_v2.csv", encoding=ENCODING)


def cargar_todo():
    return cargar_inventario(), cargar_transacciones(), cargar_feedback()


def parsear_fecha(serie: pd.Series) -> pd.Series:
    """
    Parsea fechas en multiple formatos de manera robusta.
    Formatos soportados: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, YYYY/MM/DD,
    DD.MM.YYYY, y variaciones con o sin ceros a la izquierda.
    Retorna datetime64; valores no parseables quedan como NaT.
    """
    FORMATOS = [
        "%d/%m/%Y",   # 25/04/2025
        "%Y-%m-%d",   # 2025-11-17
        "%d-%m-%Y",   # 25-04-2025
        "%Y/%m/%d",   # 2025/11/17
        "%d.%m.%Y",   # 25.04.2025
        "%m/%d/%Y",   # 04/25/2025 (US)
        "%Y%m%d",     # 20251117
    ]

    serie_str = serie.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")

    for fmt in FORMATOS:
        mask = result.isnull() & serie.notnull()
        if not mask.any():
            break
        parsed = pd.to_datetime(serie_str[mask], format=fmt, errors="coerce")
        valid = parsed.notnull()
        result[mask & valid] = parsed[valid]

    return result


def validar_fechas_futuras(df: pd.DataFrame, columna: str, fecha_limite=None) -> pd.Series:
    """
    Retorna mascara booleana de filas con fecha posterior a fecha_limite.
    Por defecto usa la fecha actual.
    """
    if fecha_limite is None:
        fecha_limite = pd.Timestamp.today()
    col = df[columna]
    if not pd.api.types.is_datetime64_any_dtype(col):
        return pd.Series(False, index=df.index)
    return col > fecha_limite


def normalizar_fechas_dataset(df: pd.DataFrame, columnas_fecha: list, fecha_limite=None) -> dict:
    """
    Normaliza todas las columnas de fecha de un dataset:
    - Parsea con parsear_fecha (multiformato)
    - Convierte a DD/MM/YYYY como string para visualizacion
    - Detecta y reporta fechas futuras, invalidas y anomalas
    Retorna (df_modificado, reporte).
    """
    if fecha_limite is None:
        fecha_limite = pd.Timestamp.today()

    reporte = {
        "fechas_parseadas": 0,
        "fechas_no_parseables": 0,
        "fechas_futuras": 0,
        "fechas_muy_antiguas": 0,
        "limite_usado": fecha_limite.strftime("%d/%m/%Y"),
        "columnas_procesadas": [],
    }

    fecha_minima = pd.Timestamp("2020-01-01")

    for col in columnas_fecha:
        if col not in df.columns:
            continue

        reporte["columnas_procesadas"].append(col)
        n_original = df[col].notnull().sum()

        parsed = parsear_fecha(df[col])
        n_parsed = parsed.notnull().sum()
        reporte["fechas_parseadas"] += n_parsed
        reporte["fechas_no_parseables"] += (n_original - n_parsed)

        futuras = parsed > fecha_limite
        n_fut = futuras.sum()
        reporte["fechas_futuras"] += n_fut
        if n_fut > 0:
            parsed[futuras] = pd.NaT

        muy_antiguas = parsed.notnull() & (parsed < fecha_minima)
        n_ant = muy_antiguas.sum()
        reporte["fechas_muy_antiguas"] += n_ant
        if n_ant > 0:
            parsed[muy_antiguas] = pd.NaT

        df[col] = parsed

    return df, reporte


def formatear_fecha_ddmmyyyy(serie) -> pd.Series:
    """Convierte una serie datetime a string DD/MM/YYYY para visualizacion."""
    if not pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    return serie.dt.strftime("%d/%m/%Y").where(serie.notnull(), None)


# --- Normalización de ciudad ---
CIUDAD_CANONICAS = {
    "bogota": "Bogotá",
    "bog": "Bogotá",
    "medellin": "Medellín",
    "med": "Medellín",
    "cali": "Cali",
    "cal": "Cali",
    "barranquilla": "Barranquilla",
    "baq": "Barranquilla",
    "cartagena": "Cartagena",
    "ctg": "Cartagena",
    "bucaramanga": "Bucaramanga",
    "bga": "Bucaramanga",
    "ventasweb": "Ventas_Web",
}


def normalizar_ciudad(valor):
    if pd.isna(valor):
        return valor
    clave = _limpiar_texto(str(valor))
    if not clave or clave in ("nan", "none", "null"):
        return valor
    if clave in CIUDAD_CANONICAS:
        return CIUDAD_CANONICAS[clave]
    match = _fuzzy_match(clave, CIUDAD_CANONICAS, umbral=0.80)
    if match:
        return match
    return str(valor).strip()


# --- Normalización de categoría ---
CATEGORIA_CANONICAS = {
    "smartphone": "Smartphones",
    "smartphones": "Smartphones",
    "accesorios": "Accesorios",
    "laptop": "Laptops",
    "laptops": "Laptops",
    "monitor": "Monitores",
    "monitores": "Monitores",
    "tablet": "Tablets",
    "tablets": "Tablets",
}


def normalizar_categoria(valor):
    if pd.isna(valor):
        return valor
    clave = _limpiar_texto(str(valor))
    if not clave or clave in ("nan", "none", "null"):
        return valor
    if clave in CATEGORIA_CANONICAS:
        return CATEGORIA_CANONICAS[clave]
    match = _fuzzy_match(clave, CATEGORIA_CANONICAS, umbral=0.75)
    if match:
        return match
    return str(valor).strip()


# --- Normalización de bodega ---
BODEGA_CANONICAS = {
    "norte": "Norte",
    "sur": "Sur",
    "occidente": "Occidente",
    "zonafranca": "Zona_Franca",
    "bodext99": "BOD-EXT-99",
    "centro": "Centro",
    "oriental": "Oriental",
}


def normalizar_bodega(valor):
    if pd.isna(valor):
        return valor
    clave = _limpiar_texto(str(valor))
    if not clave or clave in ("nan", "none", "null"):
        return valor
    if clave in BODEGA_CANONICAS:
        return BODEGA_CANONICAS[clave]
    match = _fuzzy_match(clave, BODEGA_CANONICAS, umbral=0.80)
    if match:
        return match
    return str(valor).strip()


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
