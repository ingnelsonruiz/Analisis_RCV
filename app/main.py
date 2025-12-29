from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI()

# Habilitar CORS para que tu frontend en Vercel pueda comunicarse sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalizar_columna(col):
    """Limpia encabezados para que coincidan sin importar espacios o tildes"""
    return str(col).upper().strip()

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # SOLUCIÓN AL ERROR 0xd3: Intentar Latin-1 para archivos de Excel/Spanish
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded_content = content.decode("latin-1")
            
        # Lectura con separador punto y coma (;) como viene en tu ejemplo
        df = pd.read_csv(io.StringIO(decoded_content), sep=";")
        
        # Estandarizar nombres de columnas
        df.columns = [normalizar_columna(c) for c in df.columns]

        resultados = []
        
        for _, row in df.iterrows():
            # Extracción de datos según tu estructura exacta
            nombre = f"{row.get('PRI NOMBRE', '')} {row.get('PRI APELLIDO', '')}"
            edad = pd.to_numeric(row.get('EDAD'), errors='coerce')
            ldl = pd.to_numeric(row.get('LDL'), errors='coerce')
            imc = pd.to_numeric(row.get('IMC'), errors='coerce')
            hta = str(row.get('DX CONFIRMADO HTA')).upper().strip()
            dm = str(row.get('DX CONFIRMADO DM')).upper().strip()
            
            # --- Lógica Médica Avanzada ---
            puntos = 0
            alertas = []
            
            # 1. Criterio Edad (Adulto Mayor)
            if pd.notna(edad) and edad >= 60:
                puntos += 2
                alertas.append("Edad ≥ 60 años")

            # 2. Comorbilidad HTA
            if hta == "SI":
                puntos += 2
                alertas.append("Diagnóstico HTA")
                
            # 3. Comorbilidad Diabetes (Criterio de Alto Riesgo Automático)
            es_diabetico = (dm == "SI")
            if es_diabetico:
                puntos += 3
                alertas.append("Paciente Diabético")
                
            # 4. Dislipidemia (LDL fuera de metas)
            if pd.notna(ldl):
                # Meta estricta para diabéticos (< 70) o general (> 130)
                if es_diabetico and ldl > 70:
                    puntos += 2
                    alertas.append(f"LDL fuera de meta DM ({ldl} mg/dl)")
                elif ldl > 130:
                    puntos += 2
                    alertas.append(f"LDL Elevado ({ldl} mg/dl)")
                
            # 5. Estado Nutricional (IMC)
            # Manejamos "SINDATO" convirtiéndolo a nulo
            if pd.notna(imc) and imc >= 30:
                puntos += 1
                alertas.append(f"Obesidad (IMC: {imc})")

            # Clasificación Final de Riesgo
            nivel = "BAJO"
            if puntos >= 6 or es_diabetico:
                nivel = "ALTO"
            elif puntos >= 3:
                nivel = "MODERADO"

            resultados.append({
                "nombre": nombre,
                "edad": int(edad) if pd.notna(edad) else 0,
                "riesgo": nivel,
                "alertas": alertas,
                "sugerencia": "Priorizar Medicina Interna / Estatinas" if nivel == "ALTO" else "Seguimiento Preventivo"
            })

        # Resumen para el Frontend
        return {
            "status": "ok",
            "registros": len(df),
            "riesgo_alto": len([p for p in resultados if p['riesgo'] == "ALTO"]),
            "riesgo_moderado": len([p for p in resultados if p['riesgo'] == "MODERADO"]),
            "riesgo_bajo": len([p for p in resultados if p['riesgo'] == "BAJO"]),
            "detalle_clinico": resultados[:100] # Enviamos los primeros 100 pacientes
        }

    except Exception as e:
        return {"error": "Error crítico al procesar", "detalle": str(e)}
