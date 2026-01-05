import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# Cargar variables de entorno (API Key)
load_dotenv()

app = FastAPI()

# Configuración de OpenAI
# Asegúrate de tener OPENAI_API_KEY en tus variables de entorno o archivo .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FUNCIÓN PARA BASE DE CONOCIMIENTO (PDF) ---
def extraer_conocimiento_pdf(ruta_pdf="base_conocimiento.pdf"):
    """Extrae el texto del PDF para usarlo como contexto en GPT"""
    try:
        if not os.path.exists(ruta_pdf):
            return "No se encontró el archivo de referencia médica."
        
        reader = PdfReader(ruta_pdf)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text()
        return texto
    except Exception as e:
        return f"Error leyendo PDF: {str(e)}"

# Cargamos el PDF en memoria al iniciar la API
CONOCIMIENTO_MEDICO = extraer_conocimiento_pdf()

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
            nombre = f"{row.get('PRI NOMBRE', '')} {row.get('PRI APELLIDO', '')}"
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

            # --- INTEGRACIÓN OPCIONAL CON GPT POR PACIENTE ---
            # Si el riesgo es alto, podemos pedir una sugerencia personalizada a la IA
            resumen_ia = ""
            if nivel == "ALTO":
                prompt_ia = f"Paciente: {nombre}, Edad: {edad}, HTA: {hta}, DM: {dm}, LDL: {ldl}. Guía médica: {CONOCIMIENTO_MEDICO[:2000]}"
                # Limitamos el texto del PDF a los primeros 2000 caracteres para no gastar tokens excesivos
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Eres un cardiólogo experto. Da una recomendación breve basada en la guía proporcionada."},
                            {"role": "user", "content": prompt_ia}
                        ],
                        max_tokens=150
                    )
                    resumen_ia = response.choices[0].message.content
                except:
                    resumen_ia = "Consulta IA no disponible"

            resultados.append({
                "nombre": nombre,
                "edad": int(edad) if pd.notna(edad) else 0,
                "riesgo": nivel,
                "alertas": alertas,
                "sugerencia_ia": resumen_ia if resumen_ia else ("Priorizar Medicina Interna" if nivel == "ALTO" else "Seguimiento Preventivo")
            })

        return {
            "status": "ok",
            "registros": len(df),
            "riesgo_alto": len([p for p in resultados if p['riesgo'] == "ALTO"]),
            "detalle_clinico": resultados[:50] # Reducido a 50 para optimizar respuesta
        }

    except Exception as e:
        return {"error": "Error crítico al procesar", "detalle": str(e)}
