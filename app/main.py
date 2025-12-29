from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

app = FastAPI(title="Analizador Clínico RCV")

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):

    content = await file.read()

    df = pd.read_csv(
        io.BytesIO(content),
        sep=";",
        encoding="latin1",
        dtype=str
    )

    # -------------------------------
    # ANÁLISIS CLÍNICO
    # -------------------------------

    total = len(df)

    hta = (df["DX CONFIRMADO HTA"] == "SI").sum()
    dm = (df["DX CONFIRMADO DM"] == "SI").sum()

    riesgo_alto = (df["CLASIFICACION DEL RCV ACTUAL"] == "RIESGO ALTO").sum()

    # TFG < 60 → ERC
    df["TFG_NUM"] = pd.to_numeric(
        df["TFG fOrmula Cockcroft and Gault Actual"],
        errors="coerce"
    )
    erc = (df["TFG_NUM"] < 60).sum()

    # IMC elevado
    df["IMC_NUM"] = pd.to_numeric(df["IMC"], errors="coerce")
    obesidad = (df["IMC_NUM"] >= 30).sum()

    # -------------------------------
    # RESULTADO CLÍNICO
    # -------------------------------

    resultado = {
        "poblacion_analizada": total,
        "hipertensos": int(hta),
        "diabeticos": int(dm),
        "pacientes_con_erc": int(erc),
        "obesidad": int(obesidad),
        "riesgo_cardiovascular_alto": int(riesgo_alto),

        "interpretacion_clinica": [
            "Alta carga de enfermedad cardiovascular",
            "Presencia significativa de HTA y DM",
            "Pacientes con deterioro de función renal",
            "Riesgo elevado de eventos cardiovasculares",
            "Requiere seguimiento médico continuo"
        ],

        "recomendaciones": [
            "Priorizar control de pacientes con HTA + DM",
            "Seguimiento estrecho de TFG",
            "Optimizar tratamiento antihipertensivo",
            "Refuerzo en adherencia farmacológica",
            "Evaluar control metabólico periódico"
        ]
    }

    return resultado
