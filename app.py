from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Configuración de base de datos y modelo
engine = create_engine("postgresql+psycopg2://pierola:pierola@148.251.179.86:5432/medsoft")
modelo = joblib.load("modelo_rf.pkl")
le_ruc = joblib.load("label_encoder_ruc.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/empresas", methods=["GET"])
def get_empresas():
    query = """
        SELECT DISTINCT razon_empresa, ruc_empresa
        FROM empresas
        WHERE razon_empresa IS NOT NULL AND ruc_empresa IS NOT NULL
    """
    df = pd.read_sql_query(query, engine)
    df['razon_empresa'] = df['razon_empresa'].str.strip().str.upper()
    df['ruc_empresa'] = df['ruc_empresa'].astype(str).str.replace(r'\D', '', regex=True).str.lstrip('0')
    return jsonify(df.to_dict(orient="records"))

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.json
    razon_empresa = data.get("empresa")
    fecha = data.get("fecha")  # formato YYYY-MM-DD

    df_empresas = pd.read_sql_query("SELECT razon_empresa, ruc_empresa FROM empresas", engine)
    df_empresas['razon_empresa'] = df_empresas['razon_empresa'].str.strip().str.upper()
    df_empresas['ruc_empresa'] = df_empresas['ruc_empresa'].astype(str).str.replace(r'\D', '', regex=True).str.lstrip('0')

    row = df_empresas[df_empresas['razon_empresa'] == razon_empresa]
    if row.empty:
        return jsonify({"error": "Empresa no encontrada"}), 404

    ruc = row.iloc[0]['ruc_empresa']
    try:
        ruc_cod = le_ruc.transform([ruc])[0]
    except:
        return jsonify({"error": f"El RUC {ruc} no está en el modelo"}), 400

    try:
        fecha_dt = pd.to_datetime(fecha)
        semana = fecha_dt.isocalendar().week
        anio = fecha_dt.year
    except Exception:
        return jsonify({"error": "Fecha inválida"}), 400

    X = pd.DataFrame([[ruc_cod, anio, semana]], columns=['ruc_cod', 'año', 'semana'])
    pred_log = modelo.predict(X)[0]
    pred_real = np.expm1(np.clip(pred_log, 0, 15))
    pred = round(pred_real)

    # Clasificación del nivel de demanda
    if pred < 5:
        nivel, med, enf, ap = "Baja", 1, 1, 0
    elif 5 <= pred <= 10:
        nivel, med, enf, ap = "Media", 1, 2, 1
    else:
        nivel, med, enf, ap = "Alta", 2, 3, 1

    return jsonify({
        "empresa": razon_empresa,
        "fecha": fecha,
        "examenes_estimados": pred,
        "nivel": nivel,
        "medicos": med,
        "enfermeros": enf,
        "apoyo": ap
    })
@app.route("/graficos", methods=["POST"])
def graficos():
    data = request.json
    razon_empresa = data.get("empresa")

    # Consulta sin convertir fechas aún
    query = """
        SELECT 
            o.fecha_apertura_po::text AS fecha_apertura_po,
            e.razon_empresa
        FROM n_orden_ocupacional o
        JOIN empresas e ON o.razon_empresa = e.razon_empresa
        WHERE o.fecha_apertura_po IS NOT NULL
    """
    df = pd.read_sql_query(query, engine)

    # Limpieza
    df['razon_empresa'] = df['razon_empresa'].str.strip().str.upper()
    df = df[df['razon_empresa'] == razon_empresa]

    # Conversión segura de fechas
    df['fecha'] = pd.to_datetime(df['fecha_apertura_po'], errors='coerce')
    df = df[df['fecha'].notnull()]  # Solo fechas válidas

    # Crear columnas de agrupación
    df['semana'] = df['fecha'].dt.isocalendar().week
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.to_period('M').astype(str)

    # Agrupación semanal
    semanal = df.groupby(['anio', 'semana']).size().reset_index(name='examenes')
    semanal['etiqueta'] = 'S' + semanal['semana'].astype(str) + '-' + semanal['anio'].astype(str)

    # Agrupación mensual
    mensual = df.groupby(['mes']).size().reset_index(name='examenes')

    return jsonify({
        "semanal": semanal[['etiqueta', 'examenes']].to_dict(orient="records"),
        "mensual": mensual.to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)
