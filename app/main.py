from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

app = FastAPI(title="Motor Clínico RCV")

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):

    try:
        # =========================
        # LECTURA DEL ARCHIVO
        # =========================
        content = await file.read()
        df = pd.read_csv(
            io.StringIO(content.decode("utf-8")),
            sep=";",
            dtype=str
        )

        # =========================
        # LIMPIEZA DE DATOS
        # =========================
        df.replace(["SINDATO", "NO APLICA", "", " "], pd.NA, inplace=True)

        # Conversión de campos numéricos
        campos_numericos = [
            "EDAD",
            "IMC",
            "LDL",
            "COLESTEROL TOTAL",
            "TFG fórmula Cockcroft and Gault Actual"
        ]

        for col in campos_numericos:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # =========================
        # INDICADORES CLÍNICOS
        # =========================
        total = len(df)

        hta = (df["DX CONFIRMADO HTA"] == "SI").sum()
        dm = (df["DX CONFIRMADO DM"] == "SI").sum()
        obesos = (df["IMC"] >= 30).sum()
        dislipidemia = (df["LDL"] >= 160).sum()
        erc = (df["TFG fórmula Cockcroft and Gault Actual"] < 60).sum()

        # =========================
        # CÁLCULO DE RIESGO CV
        # =========================
        def calcular_riesgo(row):
            score = 0
            if row.get("EDAD", 0) >= 60: score += 2
            if row.get("DX CONFIRMADO HTA") == "SI": score += 2
            if row.get("DX CONFIRMADO DM") == "SI": score += 2
            if row.get("LDL", 0) and row["LDL"] >= 160: score += 2
            if row.get("IMC", 0) and row["IMC"] >= 30: score += 1

            if score >= 6:
                return "RIESGO ALTO"
            elif score >= 3:
                return "RIESGO MODERADO"
            else:
                return "RIESGO BAJO"

        df["RIESGO_CV"] = df.apply(calcular_riesgo, axis=1)

        # =========================
        # RESULTADOS
        # =========================
        resultado = {
            "poblacion_total": int(total),
            "hipertensos": int(hta),
            "diabeticos": int(dm),
            "obesidad": int(obesos),
            "dislipidemia": int(dislipidemia),
            "enfermedad_renal": int(erc),
            "riesgo_alto": int((df["RIESGO_CV"] == "RIESGO ALTO").sum()),
            "riesgo_moderado": int((df["RIESGO_CV"] == "RIESGO MODERADO").sum()),
            "riesgo_bajo": int((df["RIESGO_CV"] == "RIESGO BAJO").sum()),
            "interpretacion_clinica": [
                "Alta carga de enfermedad cardiovascular",
                "Presencia significativa de HTA y Diabetes",
                "Riesgo renal relevante en la población",
                "Necesidad de control metabólico estricto",
                "Seguimiento prioritario a pacientes de alto riesgo"
            ]
        }

        return resultado

    except Exception as e:
        return {
            "error": "Error al procesar el archivo",
            "detalle": str(e)
        }
