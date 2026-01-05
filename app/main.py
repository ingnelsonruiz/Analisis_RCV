import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
load_dotenv()

app = FastAPI(
    title="Atlas – Analítica Clínica Multi-Prestador",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===============================
# UTILIDADES
# ===============================
def clean_num(df, col):
    if col not in df.columns:
        return pd.Series([0.0] * len(df))
    return pd.to_numeric(
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .replace("SINDATO", None)
        .replace("NO APLICA", None),
        errors="coerce"
    )

def clean_text(df, col):
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str).str.upper().str.strip()

# ===============================
# ENDPOINT PRINCIPAL
# ===============================
@app.post("/analizar")
async def analizar_archivo(
    file: UploadFile = File(...),
    prestador: str | None = Query(default=None)
):
    try:
        content = await file.read()

        try:
            decoded = content.decode("utf-8")
        except:
            decoded = content.decode("latin-1")

        df = pd.read_csv(io.StringIO(decoded), sep=";")
        df.columns = [c.strip().upper() for c in df.columns]

        # ===============================
        # IDENTIFICACIÓN DEL PRESTADOR
        # ===============================
        if prestador:
            nombre_prestador = prestador
        elif "NOMBRE DE LA IPS QUE HACE SEGUIMIENTO" in df.columns:
            nombre_prestador = df["NOMBRE DE LA IPS QUE HACE SEGUIMIENTO"].dropna().unique()[0]
        else:
            nombre_prestador = "PRESTADOR_DESCONOCIDO"

        # ===============================
        # VARIABLES CLÍNICAS CLAVE
        # ===============================
        tas = clean_num(df, "ÚLTIMA TENSIÓN ARTERIAL SISTOLICA")
        tad = clean_num(df, "ÚLTIMA TENSIÓN ARTERIAL DIASTÓLICA")
        hba1c = clean_num(df, "REPORTE DE HEMOGLOBINA GLICOSILADA (SOLO PARA USUARIOS CON DX DE DM)")
        tfg = clean_num(df, "TFG FÓRMULA COCKCROFT AND GAULT ACTUAL")
        ldl = clean_num(df, "LDL")

        dx_hta = clean_text(df, "DX CONFIRMADO HTA")
        dx_dm = clean_text(df, "DX CONFIRMADO DM")
        rcv = clean_text(df, "CLASIFICACION DEL RCV ACTUAL")

        # ===============================
        # CONTADORES GLOBALES
        # ===============================
        total = len(df)
        hta_total = (dx_hta == "SI").sum()
        dm_total = (dx_dm == "SI").sum()
        tfg_menor_60 = (tfg < 60).sum()

        riesgo_alto = 0
        riesgo_moderado = 0
        riesgo_bajo = 0

        pacientes_prioritarios = []

        # ===============================
        # ANÁLISIS PACIENTE A PACIENTE
        # ===============================
        for idx, row in df.iterrows():
            alertas = []

            if tas[idx] >= 140 or tad[idx] >= 90:
                alertas.append("HTA DESCONTROLADA")

            if dx_dm[idx] == "SI" and hba1c[idx] >= 7:
                alertas.append("DM NO CONTROLADA")

            if 0 < tfg[idx] < 60:
                alertas.append(f"TFG BAJA ({round(tfg[idx],1)})")

            if (
                len(alertas) >= 2
                or tfg[idx] < 30
                or rcv[idx] == "RIESGO ALTO"
            ):
                riesgo = "ALTO"
                riesgo_alto += 1

            elif len(alertas) == 1 or rcv[idx] == "RIESGO MODERADO":
                riesgo = "MODERADO"
                riesgo_moderado += 1

            else:
                riesgo = "BAJO"
                riesgo_bajo += 1

            if riesgo in ["ALTO", "MODERADO"] and len(pacientes_prioritarios) < 20:
                pacientes_prioritarios.append({
                    "identificacion": row.get("NÚMERO DE IDENTIFICACIÓN", ""),
                    "nombre": f"{row.get('PRI NOMBRE','')} {row.get('PRI APELLIDO','')}",
                    "edad": row.get("EDAD", ""),
                    "sexo": row.get("SEXO", ""),
                    "riesgo": riesgo,
                    "alertas": alertas
                })

        # ===============================
        # RESUMEN PARA IA
        # ===============================
        resumen_ia = f"""
        Prestador analizado: {nombre_prestador}
        Total pacientes: {total}
        HTA confirmada: {hta_total}
        DM confirmada: {dm_total}
        Pacientes con TFG < 60: {tfg_menor_60}
        Riesgo Alto: {riesgo_alto}
        Riesgo Moderado: {riesgo_moderado}
        Riesgo Bajo: {riesgo_bajo}
        Promedio HbA1c: {round(hba1c[hba1c>0].mean(),2)}
        Promedio LDL: {round(ldl[ldl>0].mean(),2)}
        """

        response_ia = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en auditoría clínica y RCV en Colombia según Resolución 0256."
                },
                {
                    "role": "user",
                    "content": resumen_ia
                }
            ]
        )

        # ===============================
        # RESPUESTA FINAL
        # ===============================
        return {
            "prestador": nombre_prestador,
            "total_registros": total,
            "riesgo_alto": riesgo_alto,
            "riesgo_moderado": riesgo_moderado,
            "riesgo_bajo": riesgo_bajo,
            "pacientes_prioritarios": pacientes_prioritarios,
            "analisis_ia": response_ia.choices[0].message.content
        }

    except Exception as e:
        return {
            "error": "Error procesando el archivo",
            "detalle": str(e)
        }
