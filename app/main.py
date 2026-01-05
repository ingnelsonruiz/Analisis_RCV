import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="API de Análisis Poblacional RCV")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def cargar_conocimiento_txt():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_txt = os.path.join(BASE_DIR, "base_conocimiento.txt")
    try:
        if not os.path.exists(ruta_txt):
            return "Priorizar metas de LDL < 70 en diabéticos y control de HTA según Res. 0256."
        with open(ruta_txt, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Guía técnica básica de riesgo cardiovascular."

CONOCIMIENTO_MEDICO = cargar_conocimiento_txt()

@app.get("/")
async def root():
    return {"mensaje": "Backend Poblacional RCV Activo"}

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded_content = content.decode("latin-1")
            
        df = pd.read_csv(io.StringIO(decoded_content), sep=";")
        df.columns = [str(c).upper().strip() for c in df.columns]

        # --- 1. PROCESAMIENTO MATEMÁTICO RÁPIDO (PANDAS) ---
        df['EDAD'] = pd.to_numeric(df.get('EDAD'), errors='coerce')
        df['LDL'] = pd.to_numeric(df.get('LDL'), errors='coerce')
        df['IMC'] = pd.to_numeric(df.get('IMC'), errors='coerce')
        
        total_pob = len(df)
        
        # Conteo de patologías
        con_hta = (df['DX CONFIRMADO HTA'].str.upper() == "SI").sum()
        con_dm = (df['DX CONFIRMADO DM'].str.upper() == "SI").sum()
        con_obesidad = (df['IMC'] >= 30).sum()
        
        # Lógica de Riesgo Masiva (Sin IA en el bucle)
        # Definimos condiciones para clasificar rápido
        condiciones_alto = (
            (df['DX CONFIRMADO DM'].str.upper() == "SI") | 
            (df['EDAD'] >= 60) & (df['DX CONFIRMADO HTA'].str.upper() == "SI") & (df['LDL'] > 130)
        )
        riesgo_alto = condiciones_alto.sum()
        
        # --- 2. CONSTRUCCIÓN DE INDICADORES GENERALES ---
        stats = {
            "total_pacientes": int(total_pob),
            "prevalencia_hta": f"{(con_hta/total_pob*100):.1f}%",
            "prevalencia_diabetes": f"{(con_dm/total_pob*100):.1f}%",
            "prevalencia_obesidad": f"{(con_obesidad/total_pob*100):.1f}%",
            "riesgo_alto_poblacional": f"{(riesgo_alto/total_pob*100):.1f}%",
            "promedio_ldl_poblacion": round(df['LDL'].mean(), 1) if not df['LDL'].empty else 0
        }

        # --- 3. UNA SOLA LLAMADA A LA IA PARA ANÁLISIS GERENCIAL ---
        analisis_ia = ""
        try:
            prompt_global = (
                f"Analiza los indicadores de esta cohorte de salud: {stats}. "
                f"Basado en esta guía técnica: {CONOCIMIENTO_MEDICO[:800]}"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un auditor médico experto. Resume el estado de la población y da 3 recomendaciones estratégicas para la IPS."
                    },
                    {"role": "user", "content": prompt_global}
                ],
                max_tokens=300
            )
            analisis_ia = response.choices[0].message.content
        except Exception:
            analisis_ia = "Análisis automático no disponible. Se sugiere intervención en pacientes con LDL fuera de meta."

        return {
            "status": "ok",
            "indicadores_generales": stats,
            "analisis_ejecutivo_ia": analisis_ia,
            "mensaje": "Análisis poblacional completado con éxito"
        }

    except Exception as e:
        return {"status": "error", "mensaje": str(e)}
