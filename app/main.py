from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalizar_col(col):
    return col.upper().strip()

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # Tu archivo usa ";" como separador según el ejemplo
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=";")
        
        # Limpiar nombres de columnas
        df.columns = [normalizar_col(c) for c in df.columns]

        analisis_pacientes = []

        for _, row in df.iterrows():
            # Extracción de variables clave según tu estructura
            edad = pd.to_numeric(row.get('EDAD'), errors='coerce')
            ldl = pd.to_numeric(row.get('LDL'), errors='coerce')
            imc = pd.to_numeric(row.get('IMC'), errors='coerce')
            hta = str(row.get('DX CONFIRMADO HTA')).upper()
            dm = str(row.get('DX CONFIRMADO DM')).upper()
            tfg = pd.to_numeric(row.get('CREATININA SANGRE (MG/DL)'), errors='coerce') # Simplificado para el ejemplo
            
            # --- Lógica Médica de RCV ---
            score = 0
            alertas = []

            # 1. Evaluación de Comorbilidades
            if hta == "SI": 
                score += 2
                alertas.append("Diagnóstico de Hipertensión")
            if dm == "SI": 
                score += 3 # La diabetes es un multiplicador de riesgo mayor
                alertas.append("Paciente Diabético (Alto Riesgo Metabólico)")
            
            # 2. Evaluación de Metas (LDL)
            if ldl > 130:
                score += 2
                alertas.append(f"LDL fuera de metas ({ldl} mg/dl)")
            elif ldl > 70 and dm == "SI":
                alertas.append("LDL > 70 en diabético: Requiere ajuste de estatinas")

            # 3. Obesidad e IMC
            if imc >= 30:
                score += 1
                alertas.append(f"Obesidad Grado I+ (IMC: {imc})")

            # 4. Clasificación Final
            nivel = "BAJO"
            if score >= 6 or dm == "SI": nivel = "ALTO"
            elif score >= 3: nivel = "MODERADO"

            analisis_pacientes.append({
                "nombre": f"{row.get('PRI NOMBRE')} {row.get('PRI APELLIDO')}",
                "edad": edad,
                "riesgo": nivel,
                "alertas": alertas,
                "sugerencia": "Iniciar estatinas de alta intensidad" if nivel == "ALTO" else "Seguimiento anual"
            })

        # Resumen estadístico para el Dashboard
        return {
            "status": "ok",
            "registros": len(df),
            "riesgo_alto": len([p for p in analisis_pacientes if p['riesgo'] == "ALTO"]),
            "riesgo_moderado": len([p for p in analisis_pacientes if p['riesgo'] == "MODERADO"]),
            "riesgo_bajo": len([p for p in analisis_pacientes if p['riesgo'] == "BAJO"]),
            "detalle_clinico": analisis_pacientes[:10] # Enviamos los primeros 10 con lujo de detalle
        }

    except Exception as e:
        return {"error": str(e)}
