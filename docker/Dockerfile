# Bio-Edge AMD • Entorno de Evaluación Reproducible
# Diseñado por Mario (Frente C)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    BIOEDGE_DEVICE=cpu \
    BIOEDGE_SOURCE=0

# Instalar dependencias del sistema y OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requerimientos e instalar paquetes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del proyecto
COPY src/ src/
COPY frontend/ frontend/
COPY models/ models/
COPY run_edge.py mock_stream.py SETUP_AMD.md README.md ./

# Exponer puerto HTTP y WebSockets
EXPOSE 8000

# Comando de inicio con fallback seguro a CPU
CMD ["python", "run_edge.py", "--device", "cpu", "--host", "0.0.0.0", "--port", "8000"]
