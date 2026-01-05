import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CardioCheck AI Backend")

# Habilitar CORS para que el HTML se comunique sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # Decodificación para archivos CSV de Excel (comunes en IPS)
        try:
            decoded_content = content.decode("utf-8")
        except:
            decoded_content = content.decode("latin-1")
            
        df = pd.read_csv(io.StringIO(decoded_content), sep=";")
        df.columns = [str(c).upper().strip() for c in df.columns]

        # 1. LIMPIEZA Y CONVERSIÓN DE COLUMNAS CLÍNICAS
        df['TAS'] = pd.to_numeric(df.get('TENSIÓN ARTERIAL SISTÓLICA AL INGRESO A BASE'), errors='coerce')
        df['TAD'] = pd.to_numeric(df.get('TENSIÓN ARTERIAL DIASTÓLICA AL INGRESO A BASE'), errors='coerce')
        df['HBA1C'] = pd.to_numeric(df.get('HEMOGLOBINA GLICOSILADA (HBA1C)'), errors='coerce')
        df['TFG'] = pd.to_numeric(df.get('TFG fOrmula Cockcroft and Gault Actual'), errors='coerce')
        df['LDL'] = pd.to_numeric(df.get('LDL'), errors='coerce')
        df['EDAD'] = pd.to_numeric(df.get('EDAD'), errors='coerce')
        df['IMC'] = pd.to_numeric(df.get('IMC'), errors='coerce')

        # 2. LÓGICA DE CLASIFICACIÓN (Para el Donut Chart y la Tabla)
        def evaluar_riesgo(row):
            alertas = []
            score = 0
            
            # Criterios de Riesgo / Alertas
            if row['TAS'] >= 140 or row['TAD'] >= 90:
                alertas.append("HTA DESCONTROLADA")
                score += 2
            if row['HBA1C'] >= 7.0:
                alertas.append("DM DESCOMPENSADA")
                score += 2
            if row['TFG'] < 60:
                alertas.append("FALLA RENAL E3+")
                score += 3
            if row['LDL'] > 100:
                alertas.append("DISLIPIDEMIA")
                score += 1

            # Clasificación Final
            if score >= 4 or row['EDAD'] >= 70: riesgo = "ALTO"
            elif score >= 2: riesgo = "MODERADO"
            else: riesgo = "BAJO"
            
            return pd.Series([riesgo, alertas])

        df[['RIESGO_CAT', 'ALERTAS_LIST']] = df.apply(evaluar_riesgo, axis=1)

        # 3. CONSTRUCCIÓN DE RESPUESTA PARA EL FRONTEND
        total_pob = len(df)
        counts = df['RIESGO_CAT'].value_counts()
        
        # Detalle para la tabla (Top 20 pacientes con más alertas)
        detalle_clinico = []
        df_criticos = df[df['RIESGO_CAT'] == "ALTO"].head(20)
        
        for _, r in df_criticos.iterrows():
            detalle_clinico.append({
                "nombre": f"{r.get('PRI NOMBRE', '')} {r.get('PRI APELLIDO', '')}",
                "edad": int(r['EDAD']) if not pd.isna(r['EDAD']) else 0,
                "riesgo": r['RIESGO_CAT'],
                "alertas": r['ALERTAS_LIST'],
                "sugerencia": "Ajustar terapia farmacológica y control en 15 días."
            })

        # 4. LLAMADA A IA PARA INFORME EJECUTIVO
        prompt_ia = (
            f"Resultados Cohorte RCV: Total {total_pob} pacientes. "
            f"Riesgo Alto: {counts.get('ALTO', 0)}, Moderado: {counts.get('MODERADO', 0)}. "
            f"Promedio LDL: {df['LDL'].mean():.1f}. HbA1c promedio: {df['HBA1C'].mean():.1f}."
        )
        
        res_ia = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un Director Médico. Resume brevemente el estado de la cohorte en 2 párrafos técnicos."},
                {"role": "user", "content": prompt_ia}
            ],
            max_tokens=300
        )

        # Retornamos los nombres exactos que usa tu JavaScript
        return {
            "status": "ok",
            "registros": total_pob,
            "riesgo_alto": int(counts.get("ALTO", 0)),
            "riesgo_moderado": int(counts.get("MODERADO", 0)),
            "riesgo_bajo": int(counts.get("BAJO", 0)),
            "detalle_clinico": detalle_clinico,
            "analisis_ia": res_ia.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}
