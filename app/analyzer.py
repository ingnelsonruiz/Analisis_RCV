def calcular_riesgo(row):
    riesgo = "BAJO"

    if row.get("DX CONFIRMADO HTA") == "SI":
        riesgo = "MODERADO"

    if row.get("DX CONFIRMADO DM") == "SI":
        riesgo = "ALTO"

    if row.get("IMC", 0) >= 30:
        riesgo = "ALTO"

    if row.get("TFG fOrmula Cockcroft and Gault Actual", 100) < 60:
        riesgo = "MUY ALTO"

    return riesgo


def analizar_pacientes(df):
    resultados = []

    for _, row in df.iterrows():
        resultados.append({
            "nombre": f"{row.get('PRI NOMBRE','')} {row.get('PRI APELLIDO','')}",
            "edad": row.get("EDAD"),
            "hta": row.get("DX CONFIRMADO HTA"),
            "diabetes": row.get("DX CONFIRMADO DM"),
            "imc": row.get("IMC"),
            "tfg": row.get("TFG fOrmula Cockcroft and Gault Actual"),
            "riesgo": calcular_riesgo(row)
        })

    return {
        "total_pacientes": len(resultados),
        "pacientes": resultados
    }
