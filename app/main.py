from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import unicodedata

app = FastAPI()

def normalizar(texto):
    return (
        texto.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=";")

        # Normalizar encabezados
        df.columns = [normalizar(c) for c in df.columns]

        # Mapeo real según tu archivo
        columnas = {
            "edad": "edad",
            "dx_confirmado_hta": "dx_confirmado_hta",
            "dx_confirmado_dm": "dx_confirmado_dm",
            "ldl": "ldl",
            "imc": "imc"
        }

        for col in columnas.values():
            if col not in df.columns:
                return {
                    "error": f"❌ Falta la columna requerida: {col}",
                    "columnas_detectadas": list(df.columns)
                }

        # Limpieza de datos
        df.replace(
            ["SINDATO", "NO APLICA", "", " "],
            pd.NA,
            inplace=True
        )

        df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
        df["ldl"] = pd.to_numeric(df["ldl"], errors="coerce")
        df["imc"] = pd.to_numeric(df["imc"], errors="coerce")

        def calcular_riesgo(row):
            score = 0

            if row["edad"] and row["edad"] >= 60:
                score += 2
            if row["dx_confirmado_hta"] == "SI":
                score += 2
            if row["dx_confirmado_dm"] == "SI":
                score += 2
            if row["ldl"] and row["ldl"] > 160:
                score += 2
            if row["imc"] and row["imc"] >= 30:
                score += 1

            if score >= 6:
                return "RIESGO ALTO"
            elif score >= 3:
                return "RIESGO MODERADO"
            else:
                return "RIESGO BAJO"

        df["riesgo"] = df.apply(calcular_riesgo, axis=1)

        return {
            "status": "ok",
            "registros": len(df),
            "riesgo_alto": int((df["riesgo"] == "RIESGO ALTO").sum()),
            "riesgo_moderado": int((df["riesgo"] == "RIESGO MODERADO").sum()),
            "riesgo_bajo": int((df["riesgo"] == "RIESGO BAJO").sum()),
            "ejemplo": df[["edad", "dx_confirmado_hta", "dx_confirmado_dm", "ldl", "imc", "riesgo"]]
                        .head(3)
                        .to_dict(orient="records")
        }

    except Exception as e:
        return {
            "error": "Error al procesar archivo",
            "detalle": str(e)
        }
