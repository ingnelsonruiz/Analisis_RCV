from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

app = FastAPI(title="Analizador RCV Clínico")

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    try:
        # Leer archivo CSV separado por ;
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=";")

        # Normalizar columnas
        df.columns = df.columns.str.strip().str.upper()

        # Limpieza básica
        df.replace(["SINDATO", "NO APLICA", "NA", ""], pd.NA, inplace=True)

        # Variables clínicas clave
        columnas = [
            "EDAD",
            "SEXO",
            "DX CONFIRMADO HTA",
            "DX CONFIRMADO DM",
            "TENSIÓN ARTERIAL SISTÓLICA AL INGRESO A BASE",
            "TENSIÓN ARTERIAL DIASTÓLICA AL INGRESO A BASE",
            "COLESTEROL TOTAL",
            "LDL",
            "IMC"
        ]

        df = df[columnas]

        # Conversión numérica segura
        for col in ["EDAD", "COLESTEROL TOTAL", "LDL", "IMC"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clasificación de riesgo
        def clasificar_riesgo(row):
            riesgo = 0

            if row["EDAD"] >= 60:
                riesgo += 2
            if row["DX CONFIRMADO HTA"] == "SI":
                riesgo += 2
            if row["DX CONFIRMADO DM"] == "SI":
                riesgo += 2
            if row["LDL"] and row["LDL"] > 160:
                riesgo += 2
            if row["IMC"] and row["IMC"] >= 30:
                riesgo += 1

            if riesgo >= 6:
                return "RIESGO ALTO"
            elif riesgo >= 3:
                return "RIESGO MODERADO"
            else:
                return "RIESGO BAJO"

        df["RIESGO_CARDIOVASCULAR"] = df.apply(clasificar_riesgo, axis=1)

        resumen = {
            "total_pacientes": len(df),
            "riesgo_alto": int((df["RIESGO_CARDIOVASCULAR"] == "RIESGO ALTO").sum()),
            "riesgo_moderado": int((df["RIESGO_CARDIOVASCULAR"] == "RIESGO MODERADO").sum()),
            "riesgo_bajo": int((df["RIESGO_CARDIOVASCULAR"] == "RIESGO BAJO").sum()),
        }

        return {
            "status": "ok",
            "resumen": resumen,
            "muestra_resultados": df.head(5).to_dict(orient="records")
        }

    except Exception as e:
        return {
            "status": "error",
            "detalle": str(e)
        }
