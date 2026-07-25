# AGENTS.md

## Contexto del proyecto
Taller de análisis de datos / detección de outliers para datos de logística y e-commerce (TechLogistics S.A.S.). No hay código existente; todos los scripts se construyen desde cero.

## Entorno
- Windows (PowerShell)
- Python 3 con `pandas` (ver `requirements.txt`)

## Datos
Tres archivos CSV en `data/`:

| Archivo | Filas | ID Clave |
|---|---|---|
| `feedback_clientes_v2.csv` | ~4,500 | `Feedback_ID`, `Transaccion_ID` |
| `inventario_central_v2.csv` | ~2,500 | `SKU_ID` |
| `transacciones_logistica_v2.csv` | ~10,000 | `Transaccion_ID`, `SKU_ID` |

**Cruces**: `Transaccion_ID` relaciona feedback ↔ transacciones; `SKU_ID` relaciona transacciones ↔ inventario.

### Codificación
Los CSV están en **Latin-1 (ISO-8859-1)**, no en UTF-8. Leer siempre con:
```python
pd.read_csv("data/...csv", encoding="latin-1")
```
Leer como UTF-8 corrompe silenciosamente los caracteres acentuados (Sí → S�, Físico → F�sico, días → d�as).

### Calidad de datos (intencional)
Estos datasets contienen datos sucios para practicar detección de outliers:
- Valores faltantes (`NaN`) en todos los archivos
- `Cantidad_Vendida` negativa (ej. -5)
- Valores fuera de rango: `Rating_Producto` de 99, `Tiempo_Entrega_Real` de 999, `Satisfaccion_NPS` de -17.5 (escala NPS: -100 a 100)
- Tipos mixtos en campos como `Ciudad_Destino` y `Ticket_Soporte_Abierto` (contiene "Sí")
- **IDs duplicados con datos diferentes**: Hay registros con mismo `Feedback_ID` pero distinta información (Transaccion_ID, ratings, comentarios diferentes). **NO son duplicados reales**; son inconsistencias del sistema de IDs. Se conservan todos los registros y solo se eliminan filas 100% idénticas (duplicados exactos).

### Política de deduplicación
- **Duplicado exacto** (toda la fila idéntica columna por columna) → **se elimina** (error de sistema).
- **ID repetido con datos diferentes** → **se CONSERVA** y se reporta como alerta. Eliminar por ID destruiría información válida.
- Esta política aplica a los 3 datasets.

### Formato de fechas
`DD/MM/AAAA` (ej. `25/04/2025`).

## Comandos
- Instalar dependencias: `pip install -r requirements.txt`
- Ejecutar dashboard: `streamlit run app.py`
- Sin suite de pruebas, linter ni type checker configurados aún.

## Entregables (25% de la nota)

### Dashboard Streamlit
- **Barra lateral**: selectores de fecha, filtros de categoría y bodega, botón "Refrescar Análisis".
- **Módulo de Transparencia**: pestaña con Antes vs Después (filas eliminadas, duplicados, % salud de datos).
- **st.tabs**: Auditoría, Operaciones, Cliente, Insights de IA.
- **Integración IA (Groq)**: botón que dispara análisis de Llama-3 basado en filtros del usuario.

### Documento de Hallazgos (PDF)
- Narrativa de negocio (no explicar código; explicar por qué la empresa pierde dinero).
- Al menos 4 capturas del dashboard.
- 3 recomendaciones tácticas numeradas y priorizadas (Baja, Media, Alta complejidad).

### Repositorio (GitHub)
- README profesional: descripción del problema, guía de instalación, enlace a app en la nube.
- Código modularizado (funciones de limpieza separadas de la UI).
- **API Key de Groq nunca en el código**: usar `st.secrets` o variables de entorno.

## Fases del Pipeline

### Fase 1: Auditoría de Calidad y Transparencia
- Health Score por dataset (antes y después del procesamiento).
- Métricas: % nulidad por columna, duplicados eliminados, magnitud de outliers.
- Justificar decisiones de imputación (media, mediana, moda) según distribución.

### Fase 2: Integración y Feature Engineering
- Unión estratégica (merge) para crear una sola fuente de verdad.
- Tratar SKU fantasma (ventas sin SKU en inventario).
- Variables derivadas: Margen de Utilidad, Brecha de Entrega, Ratio de Soporte por Categoría.

### Fase 3: IA con Groq (Llama-3)
- Análisis del resumen estadístico según filtros del usuario.
- Generar 3 párrafos de recomendación estratégica en tiempo real.

## Preguntas de Alta Gerencia (obligatorio responder)
1. Fuga de Capital: SKUs con margen negativo. ¿Pérdida por volumen o falla de precios?
2. Crisis Logística: Ciudades/bodegas con mayor correlación Tiempo de Entrega vs NPS bajo.
3. Venta Invisible: Impacto financiero de ventas sin SKU en inventario.
4. Diagnóstico de Fidelidad: Categorías con alto stock pero sentimiento negativo.
5. Riesgo Operativo: Relación entre antigüedad de revisión de stock y tasa de tickets de soporte.
