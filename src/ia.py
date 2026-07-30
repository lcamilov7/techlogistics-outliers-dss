"""
Modulo de Inteligencia Artificial (Fase 3).
Genera un resumen estadistico completo y consulta a Groq (Llama-3)
para obtener recomendaciones estrategicas de negocio.
"""
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def construir_resumen_estadistico(auditoria, info_huerfanos, margen_total, n_margen_positivo,
                                   n_margen_negativo, n_sin_margen, brecha_promedio,
                                   pct_cumplen, brecha_por_ciudad, ticket_cat,
                                   ticket_bodega, riesgo_bodega) -> str:
    """
    Construye un prompt con datos estadisticos objetivos para que Llama-3
    genere las conclusiones de negocio. NO debe incluir opiniones ni
    recomendaciones - solo hechos numericos.
    """

    prompt = f"""Eres un consultor senior de estrategia para TechLogistics S.A.S., 
una empresa de retail tecnologico (smartphones, laptops, tablets, accesorios, monitores).

Se realizo una auditoria de calidad de datos y un analisis cruzado de 3 fuentes:
- inventario_central (2,500 productos)
- transacciones_logistica (10,000 ventas)  
- feedback_clientes (4,500 registros)

A continuacion te presento los HALLAZGOS ESTADISTICOS OBJETIVOS.
Tu tarea es interpretarlos y generar 3 parrafos de RECOMENDACION ESTRATEGICA
que la junta directiva pueda ejecutar.

---

📊 FASE 1 — AUDITORIA DE CALIDAD DE DATOS

Health Score (0-100, donde 100 = datos perfectos):
- Inventario: {auditoria['inventario']['health_score_antes']} → 100.0 (despues de limpieza)
- Transacciones: {auditoria['transacciones']['health_score_antes']} → 98.3 (despues de limpieza)
- Feedback: {auditoria['feedback']['health_score_antes']} → 96.8 (despues de limpieza)

Acciones de limpieza realizadas:
- Inventario: {auditoria['inventario']['metricas']['celdas_nulas']} celdas nulas imputadas (stock con mediana por categoria, lead time con mediana, costos outliers capados)
- Transacciones: {auditoria['transacciones']['metricas']['celdas_nulas']} celdas nulas imputadas (cantidades negativas → mediana, costo envio → mediana por canal)
- Feedback: {auditoria['feedback']['metricas']['celdas_nulas']} celdas nulas. 969 registros con Feedback_ID repetido con datos diferentes CONSERVADOS (no son duplicados reales)

Anomalias de dominio detectadas (valores fuera de rangos logicos):
- Rating_Producto fuera de [1,5]: 30 registros (0.67%) → error de captura
- Edad_Cliente > 120: 23 registros (0.51%) → error de captura
- Tiempo_Entrega_Real > 365 dias: 50 registros (0.5%) → logisticamente imposible
- Cantidad_Vendida negativa: 100 registros (1.0%) → error de captura
- Stock_Actual negativo: 60 registros (2.4%) → error de sistema ERP

---

🚚 FASE 2 — INTEGRACION Y FUENTE UNICA DE VERDAD

Left Join transacciones + inventario sobre SKU_ID (10,000 filas, 0 perdidas):
- SKUs huerfanos (sin catalogo de inventario): {info_huerfanos['skus_huerfanos_unicos']} SKUs unicos
- Ventas sin catalogo: {info_huerfanos['transacciones_huerfanas']} transacciones ({info_huerfanos['pct_huerfanas']}% del total)
- Ingreso generado por SKUs fantasma: ${info_huerfanos['ingreso_huerfano_usd']:,.2f} ({info_huerfanos['pct_ingreso_en_riesgo']}% del total)
- Estos productos se venden en TODOS los canales (Fisico, App, Online, WhatsApp) y en TODAS las ciudades.
- No son errores de digitacion: tienen nomenclatura PROD-XXXX identica al catalogo, se venden 3.6 veces en promedio, y su precio promedio (${info_huerfanos['ingreso_huerfano_usd']/info_huerfanos['transacciones_huerfanas']:,.0f}) es identico al de productos con catalogo (${(info_huerfanos['ingreso_total_usd']-info_huerfanos['ingreso_huerfano_usd'])/(info_huerfanos['total_transacciones']-info_huerfanos['transacciones_huerfanas']):,.0f}).

---

💰 ANALISIS DE MARGEN BRUTO

Formula: Margen = (Precio_Venta × Cantidad) - (Costo_Unitario × Cantidad + Costo_Envio)
Calculable solo para {info_huerfanos['total_transacciones'] - n_sin_margen} transacciones con costo conocido.

- Margen bruto total: ${margen_total:,.0f}
- Transacciones con margen POSITIVO: {n_margen_positivo} 
- Transacciones con margen NEGATIVO (perdida): {n_margen_negativo}
- Transacciones SIN margen calculable (SKU fantasma): {n_sin_margen}

---

⏱️ BRECHA DE ENTREGA VS PROMETIDO

Brecha = Tiempo_Entrega_Real - Lead_Time_Dias (Lead_Time viene del inventario).
- Promedio global: {brecha_promedio:+.1f} dias (positivo = se entrega TARDE)
- % de entregas que cumplen o adelantan: {pct_cumplen}%
"""

    # Agregar brecha por ciudad
    prompt += "\n\nBRECHA POR CIUDAD (Tiempo_Entrega_Real - Lead_Time_Dias):\n"
    for _, r in brecha_por_ciudad.iterrows():
        prompt += f"- {r.iloc[0]}: {r['Brecha_Promedio']:+.1f} dias ({int(r['Transacciones'])} ventas)\n"

    prompt += f"""
---

🎫 RATIO DE SOPORTE POR CATEGORIA (Ticket_Soporte_Abierto = 'Si')
"""
    for _, r in ticket_cat.iterrows():
        prompt += f"- {r['Categoria']}: {r['Ratio_Tickets_%']}% de {int(r['Total'])} ventas\n"

    prompt += "\nRATIO DE SOPORTE POR BODEGA:\n"
    for _, r in ticket_bodega.iterrows():
        prompt += f"- {r['Bodega_Origen']}: {r['Ratio_Tickets_%']}% de {int(r['Total'])} ventas\n"

    prompt += f"""
---

🕐 ANTIGUEDAD DE REVISION VS TICKETS
"""
    for _, r in riesgo_bodega.iterrows():
        prompt += f"- {r['Bodega_Origen']}: {r['Antiguedad_Promedio']} dias desde ultima revision, {r['Ratio_Tickets_%']}% tickets\n"

    prompt += f"""

---

🎯 PREGUNTAS QUE DEBE RESPONDER LA JUNTA DIRECTIVA

1. Fuga de Capital: Hay {n_margen_negativo} transacciones con margen negativo. 
   ¿Es por volumen (se vende mucho con poco margen) o por falla de precios?

2. Crisis Logistica: La brecha promedio de entrega es {brecha_promedio:+.1f} dias.
   ¿Que ciudades y bodegas requieren un cambio inmediato de operador logistico?

3. Venta Invisible: ${info_huerfanos['ingreso_huerfano_usd']:,.0f} en ingresos ({info_huerfanos['pct_ingreso_en_riesgo']}%) 
   provienen de {info_huerfanos['skus_huerfanos_unicos']} SKUs sin registrar en inventario. 
   ¿Que porcentaje del ingreso total esta en riesgo por falta de control de inventario?

4. Diagnostico de Fidelidad: ¿Existen categorias con alto stock pero 
   con sentimiento de cliente negativo (NPS bajo, alta tasa de tickets)?

5. Riesgo Operativo: ¿Que bodegas estan operando a ciegas (sin revision reciente) 
   y como impacta esto en la tasa de tickets de soporte?

---

INSTRUCCION: Genera EXACTAMENTE 3 parrafos de recomendacion estrategica.
Cada parrafo debe:
- Tener un titulo en negrita que resuma la accion (ej: **1. Registrar los 480 SKUs huerfanos en el ERP**)
- Incluir datos numericos concretos del analisis para justificar la recomendacion
- Ser accionable por la junta directiva (que, como, cuando, impacto esperado)
- NO uses codigo, NO uses Markdown de tablas, SOLO texto narrativo
"""
    return prompt


def consultar_llama(prompt: str) -> str:
    """Envia el prompt a Groq con Llama-3 y retorna la respuesta.
    Soporta .env local y st.secrets para Streamlit Cloud."""
    api_key = os.getenv("GROQ_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_KEY")
        except Exception:
            pass

    if not api_key:
        return "Error: No se encontro GROQ_KEY en variables de entorno ni en st.secrets."

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un consultor senior de estrategia de negocio. "
                        "Respondes unicamente con recomendaciones estrategicas "
                        "basadas en datos. NO escribes codigo. NO usas tablas. "
                        "Escribes en español profesional."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] Groq: {str(e)}"
