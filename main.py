from fastapi import FastAPI, UploadFile, File
import pandas as pd
from app.analyzer import analizar_pacientes

app = FastAPI(
    title="API Análisis RCV",
    description="API para análisis clínico de Riesgo Cardiovascular",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "status": "API RCV activa",
        "mensaje": "Lista para análisis clínico"
    }

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, sep=";")
    resultado = analizar_pacientes(df)
    return resultado
