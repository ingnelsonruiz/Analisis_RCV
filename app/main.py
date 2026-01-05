import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

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
        try:
            decoded = content.decode("utf-8")
        except:
            decoded = content.decode("latin-1")
            
        # Leer el CSV con separador punto y coma
        df = pd.read_csv(io.StringIO(decoded), sep=";")
        df.columns = [c.strip() for c in df.columns]

        # --- PROCESAMIENTO DE DATOS CLÍNICOS ---
        
        # Función para limpiar números (ej: '130,5' -> 130.5)
        def clean_num(col):
            if col in df.columns:
                return pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            return pd.Series([0.0] * len(df))

        # Variables clave según tu estructura
        tas_final = clean_num('ÚLTIMA TENSIÓN ARTERIAL SISTOLICA')
        tad_final = clean_num('ÚLTIMA TENSIÓN ARTERIAL DIASTÓLICA')
        hba1c = clean_num('REPORTE DE HEMOGLOBINA GLICOSILADA (SOLO PARA USUARIOS CON DX DE DM)')
        tfg = clean_num('TFG fórmula Cockcroft and Gault Actual')
        ldl = clean_num('LDL')
        
        # Conteos de Diagnóstico
        tiene_hta = df['DX CONFIRMADO HTA'].str.upper().str.contains('SI', na=False).sum()
        tiene_dm = df['DX CONFIRMADO DM'].str.upper().str.contains('SI', na=False).sum()

        # --- LÓGICA DE RIESGO ---
        riesgo_alto = 0
        riesgo_mod = 0
        riesgo_bajo = 0
        pacientes_prioritarios = []

        for idx, row in df.iterrows():
            alertas = []
            
            # Regla 1: Descontrol Tensional (Basado en la última toma)
            if tas_final[idx] >= 140 or tad_final[idx] >= 90:
                alertas.append("HTA DESCONTROLADA")
            
            # Regla 2: Descontrol Glucémico
            if row['DX CONFIRMADO DM'] == 'SI' and hba1c[idx] >= 7.0:
                alertas.append("DM NO CONTROLADA")
            
            # Regla 3: Función Renal Comprometida
            if 0 < tfg[idx] < 60:
                alertas.append(f"ERC ESTADIO 3+ ({tfg[idx]})")

            # Clasificación de Riesgo para el Dashboard
            if len(alertas) >= 2 or tfg[idx] < 30 or row['CLASIFICACION DEL RCV ACTUAL'] == 'RIESGO ALTO':
                riesgo = "ALTO"; riesgo_alto += 1
            elif len(alertas) == 1 or row['CLASIFICACION DEL RCV ACTUAL'] == 'RIESGO MODERADO':
                riesgo = "MODERADO"; riesgo_mod += 1
            else:
                riesgo = "BAJO"; riesgo_bajo += 1

            # Agregar a tabla de intervención (Top 20 críticos)
            if (riesgo == "ALTO" or riesgo == "MODERADO") and len(pacientes_prioritarios) < 20:
                nombre_completo = f"{row['PRI NOMBRE']} {row['PRI APELLIDO']}"
                pacientes_prioritarios.append({
                    "nombre": nombre_completo,
                    "edad": str(row['EDAD']),
                    "riesgo": riesgo,
                    "alertas": alertas,
                    "sugerencia": "Revisión de terapia farmacológica urgente."
                })

        # --- RESUMEN PARA LA IA ---
        # Calculamos los promedios antes para que la IA tenga datos reales
        hba1c_media = hba1c[hba1c > 0].mean()
        ldl_media = ldl[ldl > 0].mean()

        informe_para_ia = f"""
        Resultados Cohorte IPSI KANKUAMA:
        - Total Pacientes: {len(df)}
        - Diagnóstico HTA: {tiene_hta} pacientes.
        - Diagnóstico DM: {tiene_dm} pacientes.
        - Promedio HbA1c: {hba1c_media:.2f}%
        - Promedio LDL: {ldl_media:.2f} mg/dL
        - Pacientes con TFG < 60: {(tfg < 60).sum()}
        
        Analiza estos datos frente a la Resolución 0256 y da recomendaciones.
        """

        response_ia = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un Director Médico experto en auditoría de la Resolución 0256 en Colombia."},
                {"role": "user", "content": informe_para_ia}
            ]
        )

        return {
            "registros": len(df),
            "riesgo_alto": riesgo_alto,
            "riesgo_moderado": riesgo_mod,
            "riesgo_bajo": riesgo_bajo,
            "detalle_clinico": pacientes_prioritarios,
            "analisis_ia": response_ia.choices[0].message.content
        }

    except Exception as e:
        return {"error": f"Error procesando el CSV: {str(e)}"}
