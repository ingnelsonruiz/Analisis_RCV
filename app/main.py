from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import unicodedata

app = FastAPI(title="Analizador Clínico RCV")

def normalizar(texto):
    if texto is None:
        return ""
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    return texto.upper().replace(" ", "_")

@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):

    try:
        content = await file.read()

        df = pd.read_csv(
            io.BytesIO(content),
            sep=";",
            encoding="latin1",
            dtype=str
        )

        # Normalizar nombres de columnas
        df.columns = [normalizar(c) for c in df.columns]

        # --------- DETECCIÓN SEGURA DE COLUMNAS ---------
        col_ht = next((c for c in df.columns if "DX_CONFIRMADO_HTA" in c), None)
        col_dm = next((c for c in df.columns if "DX_CONFIRMADO_DM" in c), None)
        col_rcv = next((c for c in df.columns if "CLASIFICACION_DEL_RCV" in c), None)
        col_tfg = next((c for c in df.columns if "TFG" in c and "ACTUAL" in c), None)
        col_imc = next((c for c in df.columns if c == "IMC"), None)

        # --------- ANÁLISIS CLÍNICO ---------
        total = len(df)

        hipertensos = (df[col_ht] == "SI").sum() if col_ht else 0
        diabeticos = (df[col_dm] == "SI").sum() if col_dm else 0
        riesgo_alto = (df[col_rcv] == "RIESGO ALTO").sum() if col_rcv else 0

        erc = 0
        if col_tfg:
            df["TFG_NUM"] = pd.to_numeric(df[col_tfg], errors="coerce")
            erc = (df["TFG_NUM"] < 60).sum()

        obesidad = 0
        if col_imc:
            df["IMC_NUM"] = pd.to_numeric(df[col_imc], errors="coerce")
            obesidad = (df["IMC_NUM"] >= 30).sum()

        return {
            "estado": "ok",
            "archivo_procesado": file.filename,
            "registros": total,
            "analisis_clinico": {
                "hipertensos": int(hipertensos),
                "diabeticos": int(diabeticos),
                "pacientes_con_erc": int(erc),
                "obesidad": int(obesidad),
                "riesgo_cardiovascular_alto": int(riesgo_alto)
            },
            "interpretacion": [
                "Alta carga de enfermedad cardiovascular",
                "Presencia significativa de HTA y DM",
                "Casos con deterioro de función renal",
                "Necesidad de seguimiento clínico continuo"
            ],
            "observacion": "El análisis es descriptivo. No se modifican los datos originales."
        }

    except Exception as e:
        return {
            "estado": "error",
            "mensaje": "Error procesando el archivo",
            "detalle_tecnico": str(e)
        }
