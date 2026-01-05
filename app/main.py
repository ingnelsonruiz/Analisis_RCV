import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CardioCheck AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- NUEVA FUNCIÓN: CARGAR CONOCIMIENTO ---
def obtener_contexto_normativo():
    ruta = "app/base_conocimiento.txt"
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    return "No hay guías adicionales cargadas."

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            decoded_content = content.decode("utf-8")
        except:
            decoded_content = content.decode("latin-1")
            
        df = pd.read_csv(io.StringIO(decoded_content), sep=";")
        df.columns = [str(c).upper().strip() for c in df.columns]

        # 1. LIMPIEZA DE DATOS (Se mantiene igual)
        df['TAS'] = pd.to_numeric(df.get('TENSIÓN ARTERIAL SISTÓLICA AL INGRESO A BASE'), errors='coerce')
        df['TAD'] = pd.to_numeric(df.get('TENSIÓN ARTERIAL DIASTÓLICA AL INGRESO A BASE'), errors='coerce')
        df['HBA1C'] = pd.to_numeric(df.get('HEMOGLOBINA GLICOSILADA (HBA1C)'), errors='coerce')
        df['TFG'] = pd.to_numeric(df.get('TFG fOrmula Cockcroft and Gault Actual'), errors='coerce')
        df['LDL'] = pd.to_numeric(df.get('LDL'), errors='coerce')
        df['EDAD'] = pd.to_numeric(df.get('EDAD'), errors='coerce')

        # 2. LÓGICA DE CLASIFICACIÓN
        def evaluar_riesgo(row):
            alertas = []
            score = 0
            if row['TAS'] >= 140 or row['TAD'] >= 90:
                alertas.append("HTA DESCONTROLADA"); score += 2
            if row['HBA1C'] >= 7.0:
                alertas.append("DM DESCOMPENSADA"); score += 2
            if row['TFG'] < 60:
                alertas.append("FALLA RENAL E3+"); score += 3
            if row['LDL'] > 100:
                alertas.append("DISLIPIDEMIA"); score += 1

            riesgo = "ALTO" if (score >= 4 or row['EDAD'] >= 70) else "MODERADO" if score >= 2 else "BAJO"
            return pd.Series([riesgo, alertas])

        df[['RIESGO_CAT', 'ALERTAS_LIST']] = df.apply(evaluar_riesgo, axis=1)

        # 3. CONSTRUCCIÓN DE ESTADÍSTICAS PARA EL PROMPT
        total_pob = len(df)
        counts = df['RIESGO_CAT'].value_counts()
        
        # --- NUEVO: CARGA DE BASE DE CONOCIMIENTO ---
        conocimiento_medico = obtener_contexto_normativo()

        # 4. LLAMADA A IA CON RAG (Retrieval Augmented Generation) BÁSICO
        prompt_ia = (
            f"DATOS DE LA COHORTE:\n"
            f"- Total: {total_pob} pacientes.\n"
            f"- Riesgo Alto: {counts.get('ALTO', 0)}\n"
            f"- Riesgo Moderado: {counts.get('MODERADO', 0)}\n"
            f"- Promedio HbA1c: {df['HBA1C'].mean():.2f}\n"
            f"- Pacientes con Falla Renal detectada: {len(df[df['TFG'] < 60])}\n"
        )
        
        res_ia = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Eres un Director Médico experto en Riesgo Cardiovascular y auditoría de la Resolución 0256 de Colombia. "
                        "Utiliza la siguiente BASE DE CONOCIMIENTO para contrastar los resultados de la cohorte y dar recomendaciones legales y clínicas:\n\n"
                        f"### BASE DE CONOCIMIENTO NORMARTIVA:\n{conocimiento_medico}"
                    )
                },
                {"role": "user", "content": prompt_ia}
            ],
            max_tokens=500
        )

        # 5. ESTRUCTURA DE RETORNO (Sincronizada con tu HTML)
        detalle_clinico = []
        df_criticos = df[df['RIESGO_CAT'] == "ALTO"].head(20)
        for _, r in df_criticos.iterrows():
            detalle_clinico.append({
                "nombre": f"{r.get('PRI NOMBRE', '')} {r.get('PRI APELLIDO', '')}",
                "edad": int(r['EDAD']) if not pd.isna(r['EDAD']) else 0,
                "riesgo": r['RIESGO_CAT'],
                "alertas": r['ALERTAS_LIST'],
                "sugerencia": "Priorizar cita por Medicina Interna según protocolo."
            })

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
