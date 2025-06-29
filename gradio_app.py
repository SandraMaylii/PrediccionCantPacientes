import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

# 1. Cargar modelo y label encoder actualizados
modelo = joblib.load("modelo_rf.pkl")
le_ruc = joblib.load("label_encoder_ruc.pkl")

# 2. Conexión a la base de datos
engine = create_engine("postgresql+psycopg2://pierola:pierola@148.251.179.86:5432/medsoft")

# 3. Obtener empresas en tiempo real
def obtener_empresas():
    query = """
        SELECT DISTINCT razon_empresa, ruc_empresa
        FROM empresas
        WHERE razon_empresa IS NOT NULL AND ruc_empresa IS NOT NULL
    """
    df = pd.read_sql_query(query, engine)
    df['razon_empresa'] = df['razon_empresa'].str.strip().str.upper()
    df['ruc_empresa'] = df['ruc_empresa'].astype(str).str.replace(r'\D', '', regex=True).str.lstrip('0')
    return df

empresas_df = obtener_empresas()

# 4. Convertir fecha a semana y año
def obtener_semana_anio(fecha):
    fecha_dt = pd.to_datetime(fecha)
    semana = fecha_dt.isocalendar().week
    anio = fecha_dt.year
    return semana, anio

# 5. Clasificación operativa
def clasificar_demanda(pred):
    if pred < 5:
        return "Baja", 1, 1, 0
    elif 5 <= pred <= 10:
        return "Media", 1, 2, 1
    else:
        return "Alta", 2, 3, 1

# 6. Función principal para predecir
def predecir_examenes(razon_empresa, fecha):
    empresa_row = empresas_df[empresas_df['razon_empresa'] == razon_empresa]
    if empresa_row.empty:
        return "Empresa no encontrada", "", "", "", "", "", ""

    ruc = empresa_row.iloc[0]['ruc_empresa']
    ruc_limpio = ruc.strip().replace(".", "").replace("-", "").lstrip("0")

    try:
        ruc_cod = le_ruc.transform([ruc_limpio])[0]
    except ValueError:
        return f"⚠️ El RUC {ruc} no está registrado en el modelo.", "", "", "", "", "", ""

    semana, anio = obtener_semana_anio(fecha)
    X = pd.DataFrame([[ruc_cod, anio, semana]], columns=['ruc_cod', 'año', 'semana'])

    pred_log = modelo.predict(X)[0]
    pred_real = np.expm1(np.clip(pred_log, 0, 15))
    pred = round(pred_real)

    nivel, medicos, enfermeros, apoyo = clasificar_demanda(pred)

    return razon_empresa, fecha, str(pred), nivel, str(medicos), str(enfermeros), str(apoyo)

# 7. Interfaz Gradio
iface = gr.Interface(
    fn=predecir_examenes,
    inputs=[
        gr.Dropdown(label="Empresa", choices=empresas_df['razon_empresa'].tolist()),
        gr.Textbox(label="Fecha de evaluación (YYYY-MM-DD)", placeholder="Ej: 2025-07-01")
    ],
    outputs=[
        gr.Textbox(label="Empresa"),
        gr.Textbox(label="Fecha seleccionada"),
        gr.Textbox(label="Exámenes estimados"),
        gr.Textbox(label="Nivel de demanda"),
        gr.Textbox(label="Médicos recomendados"),
        gr.Textbox(label="Enfermeros"),
        gr.Textbox(label="Personal de apoyo"),
    ],
    title="🩺 Asistente de Predicción de Exámenes Ocupacionales",
    description="Selecciona una empresa y una fecha para estimar la demanda semanal de exámenes pre-ocupacionales."
)

if __name__ == "__main__":
    iface.launch()
