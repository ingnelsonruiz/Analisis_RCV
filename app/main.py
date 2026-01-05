import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="API de Análisis de Riesgo Cardiovascular")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEJORA: RUTA ABSOLUTA PARA EL ARCHIVO ---
def cargar_conocimiento_txt():
    # Obtiene la carpeta donde está este archivo (main.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_txt = os.path.join(BASE_DIR, "base_conocimiento.txt")
    
    try:
        if not os.path.exists(ruta_txt):
            print(f"⚠️ Alerta: No se encontró el archivo en {ruta_txt}")
            return "Guía básica: Priorizar control de HTA y Diabetes según protocolos estándar."
        
        with open(ruta_txt, "r", encoding="utf-8") as f:
            print(f"✅ Base de conocimiento cargada exitosamente desde {ruta_txt}")
            return f.read()
    except Exception as e:
        return f"Error leyendo base de conocimiento: {str(e)}"

CONOCIMIENTO_MEDICO = cargar_conocimiento_txt()

# --- NUEVO: RUTA DE BIENVENIDA (Para evitar el 404 en la raíz) ---
@app.get("/")
async def root():
    return {
        "mensaje": "Backend de Análisis RCV funcionando",
        "base_conocimiento_cargada": "No se encontró el archivo" not in CONOCIMIENTO_MEDICO[:50]
    }

def normalizar_columna(col):
    return str(col).upper().strip()

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded_content = content.decode("latin-1")
            
        df = pd.read_csv(io.StringIO(decoded_content), sep=";")
        df.columns = [normalizar_columna(c) for c in df.columns]

        resultados = []
        
        for _, row in df.iterrows():
            nombre = f"{row.get('PRI NOMBRE', '')} {row.get('PRI APELLIDO', '')}".strip()
            edad = pd.to_numeric(row.get('EDAD'), errors='coerce')
            ldl = pd.to_numeric(row.get('LDL'), errors='coerce')
            imc = pd.to_numeric(row.get('IMC'), errors='coerce')
            hta = str(row.get('DX CONFIRMADO HTA')).upper().strip()
            dm = str(row.get('DX CONFIRMADO DM')).upper().strip()
            
            puntos = 0
            alertas = []
            
            if pd.notna(edad) and edad >= 60:
                puntos += 2
                alertas.append("Edad ≥ 60 años")

            if hta == "SI":
                puntos += 2
                alertas.append("Diagnóstico HTA")
                
            es_diabetico = (dm == "SI")
            if es_diabetico:
                puntos += 3
                alertas.append("Paciente Diabético")
                
            if pd.notna(ldl):
                if es_diabetico and ldl > 70:
                    puntos += 2
                    alertas.append(f"LDL fuera de meta DM ({ldl} mg/dl)")
                elif ldl > 130:
                    puntos += 2
                    alertas.append(f"LDL Elevado ({ldl} mg/dl)")
                
            if pd.notna(imc) and imc >= 30:
                puntos += 1
                alertas.append(f"Obesidad (IMC: {imc})")

            nivel = "BAJO"
            if puntos >= 6 or es_diabetico:
                nivel = "ALTO"
            elif puntos >= 3:
                nivel = "MODERADO"

            resumen_ia = ""
            if nivel == "ALTO":
                prompt_ia = (
                    f"Paciente: {nombre}, Edad: {edad}, HTA: {hta}, DM: {dm}, LDL: {ldl}, IMC: {imc}. "
                    f"Contexto técnico de indicadores RCV: {CONOCIMIENTO_MEDICO}"
                )
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Eres un cardiólogo experto. Da una recomendación médica brevísima (máx 3 frases)."},
                            {"role": "user", "content": prompt_ia}
                        ],
                        max_tokens=150
                    )
                    resumen_ia = response.choices[0].message.content
                except Exception:
                    resumen_ia = "Priorizar Medicina Interna."

            resultados.append({
                "nombre": nombre,
                "riesgo": nivel,
                "alertas": alertas,
                "sugerencia_ia": resumen_ia
            })

        return {
            "status": "ok",
            "registros_procesados": len(df),
            "total_riesgo_alto": len([p for p in resultados if p['riesgo'] == "ALTO"]),
            "detalle_clinico": resultados[:50]
        }

    except Exception as e:
        return {"status": "error", "mensaje": str(e)}
