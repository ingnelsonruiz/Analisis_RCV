import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Sistema de Inteligencia Epidemiológica RCV")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        decoded_content = content.decode("latin-1") # Formato común en reportes de salud
        df = pd.read_csv(io.StringIO(decoded_content), sep=";")
        df.columns = [str(c).upper().strip() for c in df.columns]

        # --- 1. PROCESAMIENTO DE VARIABLES CLÍNICAS ---
        df['TAS'] = pd.to_numeric(df.get('TENSIÓN ARTERIAL SISTÓLICA AL INGRESO A BASE'), errors='coerce')
        df['TAD'] = pd.to_numeric(df.get('TENSIÓN ARTERIAL DIASTÓLICA AL INGRESO A BASE'), errors='coerce')
        df['HBA1C'] = pd.to_numeric(df.get('HEMOGLOBINA GLICOSILADA (HBA1C)'), errors='coerce')
        df['TFG'] = pd.to_numeric(df.get('TFG fOrmula Cockcroft and Gault Actual'), errors='coerce')
        df['LDL'] = pd.to_numeric(df.get('LDL'), errors='coerce')
        df['IMC'] = pd.to_numeric(df.get('IMC'), errors='coerce')

        total = len(df)
        hipertensos = df[df['DX CONFIRMADO HTA'].str.upper() == "SI"]
        diabeticos = df[df['DX CONFIRMADO DM'].str.upper() == "SI"]

        # --- 2. GENERACIÓN DE INFORMACIÓN ESTRATÉGICA (ESTADÍSTICAS) ---
        # Cruce de metas: Diabéticos con LDL fuera de meta (>70)
        dm_mal_control_ldl = (diabeticos['LDL'] > 70).sum() if not diabeticos.empty else 0
        
        # Riesgo Renal Avanzado (Estadios 3b, 4 y 5)
        falla_renal_critica = (df['TFG'] < 45).sum()

        # Obesidad de Riesgo (IMC > 35)
        obesidad_morbida = (df['IMC'] >= 35).sum()

        # Inercia Clínica: Pacientes que fuman o beben y son HTA/DM
        riesgo_estilo_vida = df[(df['FUMA'] == "SI") | (df['CONSUMO DE ALCOHOL'] == "SI")].shape[0]

        stats_profundas = {
            "poblacion": {
                "total": total,
                "hta_porcentaje": f"{(len(hipertensos)/total*100):.1f}%",
                "dm_porcentaje": f"{(len(diabeticos)/total*100):.1f}%"
            },
            "metas_clinicas": {
                "hta_controlada": f"{( ((hipertensos['TAS'] < 140) & (hipertensos['TAD'] < 90)).sum() / len(hipertensos) * 100):.1f}%" if not hipertensos.empty else "0%",
                "dm_controlada_hba1c": f"{( (diabeticos['HBA1C'] < 7.0).sum() / len(diabeticos) * 100):.1f}%" if not diabeticos.empty else "0%",
                "diabeticos_riesgo_ldl": f"{(dm_mal_control_ldl / len(diabeticos) * 100):.1f}% de diabéticos NO están en meta de LDL < 70" if not diabeticos.empty else "N/A"
            },
            "alertas_criticas": {
                "con_falla_renal_grave": int(falla_renal_critica),
                "obesidad_grado_2_3": int(obesidad_morbida),
                "pacientes_con_habitos_riesgo": riesgo_estilo_vida
            }
        }

        # --- 3. ANALISIS DE IA CON ENFOQUE EN GESTIÓN ---
        # Enviamos TODA esta info a la IA para que el reporte sea denso
        prompt = f"Analiza esta cohorte médica de {total} pacientes con estos KPIs: {stats_profundas}."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un Consultor Senior en Riesgo Cardiovascular. Tu informe debe ser detallado, identificar fallas en la atención y proponer 3 intervenciones de salud pública para mejorar los indicadores de la Resolución 0256."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )

        return {
            "status": "ok",
            "dashboard": stats_profundas,
            "informe_ia": response.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}
