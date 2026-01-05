import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# 1. Cargar variables de entorno
load_dotenv()

app = FastAPI(title="API de Análisis de Riesgo Cardiovascular")

# 2. Configuración de OpenAI
# Recuerda configurar OPENAI_API_KEY en el dashboard de Render -> Environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. FUNCIÓN PARA CARGAR BASE DE CONOCIMIENTO OPTIMIZADA (.txt) ---
def cargar_conocimiento_txt(ruta_txt="base_conocimiento.txt"):
    """
    Carga los indicadores técnicos de RCV desde un archivo de texto.
    Esto es más eficiente que procesar un PDF completo en cada ejecución.
    """
    try:
        if not os.path.exists(ruta_txt):
            print(f"⚠️ Alerta: No se encontró {ruta_txt}. La IA usará conocimiento general.")
            return "Guía básica: Priorizar control de HTA y Diabetes según protocolos estándar."
        
        with open(ruta_txt, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error leyendo base de conocimiento: {str(e)}"

# Cargamos el conocimiento filtrado en memoria al iniciar la API
CONOCIMIENTO_MEDICO = cargar_conocimiento_txt()

def normalizar_columna(col):
    return str(col).upper().strip()

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # Manejo de codificación del CSV
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded_content = content.decode("latin-1")
            
        # Lectura del DataFrame (usando punto y coma como separador)
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
            
            # --- Lógica de Riesgo (Score RCV) ---
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

            # Clasificación de Nivel
            nivel = "BAJO"
            if puntos >= 6 or es_diabetico:
                nivel = "ALTO"
            elif puntos >= 3:
                nivel = "MODERADO"

            # --- 4. CONSULTA A LA IA (Solo para Riesgo ALTO) ---
            resumen_ia = ""
            if nivel == "ALTO":
                # Creamos el contexto para la IA usando el TXT cargado
                prompt_ia = (
                    f"Paciente: {nombre}, Edad: {edad}, HTA: {hta}, DM: {dm}, LDL: {ldl}, IMC: {imc}. "
                    f"Contexto técnico de indicadores RCV: {CONOCIMIENTO_MEDICO}"
                )
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system", 
                                "content": "Eres un cardiólogo experto. Da una recomendación médica brevísima (máx 3 frases) "
                                           "basada estrictamente en los indicadores de calidad RCV proporcionados."
                            },
                            {"role": "user", "content": prompt_ia}
                        ],
                        max_tokens=150,
                        temperature=0.5
                    )
                    resumen_ia = response.choices[0].message.content
                except Exception:
                    resumen_ia = "Sugerencia: Priorizar Medicina Interna y control estricto de metas."

            resultados.append({
                "nombre": nombre,
                "edad": int(edad) if pd.notna(edad) else 0,
                "riesgo": nivel,
                "alertas": alertas,
                "sugerencia_ia": resumen_ia if resumen_ia else ("Seguimiento preventivo" if nivel != "ALTO" else "Priorizar Medicina Interna")
            })

        return {
            "status": "ok",
            "registros_procesados": len(df),
            "total_riesgo_alto": len([p for p in resultados if p['riesgo'] == "ALTO"]),
            "detalle_clinico": resultados[:50]  # Limite para evitar saturar la respuesta JSON
        }

    except Exception as e:
        return {"status": "error", "mensaje": "Error crítico al procesar el archivo", "detalle": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
