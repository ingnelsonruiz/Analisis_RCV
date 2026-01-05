FROM python:3.10

WORKDIR /app

# Copia los requerimientos e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido de tu carpeta local al WORKDIR
COPY . .

# Si main.py está dentro de una carpeta llamada 'app', usa esta línea:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
