# Bio-Edge AMD • Monitoreo Ecológico & Conteo Inteligente en el Borde

[![AMD Ryzen AI](https://img.shields.io/badge/AMD-Ryzen_AI_NPU-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/)
[![AMD Radeon](https://img.shields.io/badge/AMD-Radeon_GPU_(DirectML)-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Vitis_AI_EP-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSockets_30+_FPS-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL_Edge_Persistence-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org/)
[![Status](https://img.shields.io/badge/Status-Hackathon_Acto_3_Ready-success?style=for-the-badge)]()

> **Hackathon AMD Acto 3 • Ventana de Ejecución: 12 al 18 de Octubre**  
> Solución embebida de borde que realiza inferencia y telemetría in-situ sobre flujos de video localmente con ultra-baja latencia y máxima eficiencia energética por vatio, sin depender de conexión a internet.

---

## 1. Propuesta de Valor y Problema Resuelto

Los sistemas tradicionales de foto-trampeo en reservas ecológicas acumulan gigabytes de video que deben extraerse físicamente y procesarse semanas después en la nube, perdiendo la capacidad de alertar incursiones furtivas o censar dinámicamente especies en peligro.

**Bio-Edge AMD** revoluciona el monitoreo biológico mediante:
* **Inferencia Neuronal in-situ Acelerada por Silicio AMD:** Aprovecha directamente la NPU Ryzen AI (con Vitis AI Execution Provider) y GPUs Radeon (mediante DirectML / ROCm) con cuantización INT8 y FP16.
* **Tracking & Conteo Direccional Autónomo:** Seguimiento espacio-temporal por centroides y cruce de líneas virtuales (`ENTRANDO` / `SALIENDO`), discriminando fauna protegida de intrusos humanos o vehículos furtivos.
* **Persistencia Transaccional Protegida en SQLite (WAL):** Almacenamiento local tolerante a cortes de energía para despliegues remotos.
* **Telemetría y Streaming de Ultra-Baja Latencia:** Servidor FastAPI transmitiendo video anotado por WebSockets a más de 30 FPS y métricas de aceleración en tiempo real a un dashboard interactivo.

---

## 2. Métricas Cuantitativas de Éxito (KPIs del MVP para Jurado AMD)

| Métrica Clave | CPU Estándar (Baseline) | Silicio AMD (Bio-Edge) | Diferencial / Impacto |
| :--- | :---: | :---: | :--- |
| **Latencia por Frame** | $> 110\text{ ms}$ | **$< 25\text{ ms}$** | **$4.5\times - 5.6\times$ más rápido**, procesamiento en tiempo real. |
| **Cadencia Throughput** | $8 - 12\text{ FPS}$ | **$> 45\text{ FPS}$** (sostenidos 1080p) | Flujo fluido sin pérdida de fotogramas críticos. |
| **Retención de Precisión** | $100\%$ (FP32) | **$> 94\%$ mAP@50** (INT8) | Cuantización con mínima pérdida de discernimiento biológico. |
| **Autonomía y Eficiencia** | Saturación CPU | **Descarga de CPU $> 60\%$** | Mayor duración de batería en estaciones de borde autónomas. |

---

## 3. Guía de Inicio Rápido en 3 Pasos

### Paso 1: Clonar y Preparar el Entorno
```bash
# Clonar repositorio
git clone <URL_DEL_REPOSITORIO>
cd "Bio-Edge AMD"

# Crear entorno virtual e instalar dependencias
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Paso 2: Verificar Aceleradores AMD
```bash
python run_edge.py --check-hardware
```

### Paso 3: Iniciar Bio-Edge AMD en el Borde
```bash
# Iniciar con acelerador DirectML y webcam/fuente predeterminada:
python run_edge.py --device directml --port 8000
```
Abrir el navegador en **[http://localhost:8000](http://localhost:8000)** para ver el stream en vivo, el censo por especie y el panel de telemetría.

> **Política de Cero Bloqueos:** Si tu equipo no dispone de GPU o NPU AMD en desarrollo, corre con fallback automático:  
> `python run_edge.py --device cpu` o ejecuta el simulador rápido con `python mock_stream.py`.

---

## 4. Estructura del Proyecto

```text
├── docs/
│   ├── SDD.md                 # System Design Document (Arquitectura, 4 Pilares, Diagramas)
│   ├── TDD.md                 # Technical Design Document & Testing Specification (APIs, Schemas)
│   └── TASKS.md               # Asignación de Tareas por Participante, Calendario y Pairing
├── src/
│   ├── engine/
│   │   ├── base_engine.py     # Clase abstracta BaseInferenceEngine y Detection contract
│   │   ├── inference_amd.py   # Wrapper ONNX Runtime (DirectML, Vitis AI, ROCm, CPU Fallback)
│   │   └── benchmark.py       # Suite de benchmarking automatizado (P50/P95/P99, FPS)
│   ├── pipeline/
│   │   ├── video_capture.py   # Captura multihilo OpenCV con buffer FIFO zero-lag
│   │   ├── tracker.py         # Centroid Tracker liviano con IDs persistentes
│   │   ├── counting.py        # Conteo direccional por cruce de líneas virtuales y ROIs
│   │   └── pipeline.py        # Coordinador central de procesamiento
│   ├── storage/
│   │   └── storage.py         # Persistencia SQLite local en modo WAL
│   └── api/
│       └── main_server.py     # API FastAPI con streaming WebSocket a 30+ FPS y REST
├── frontend/                  # Dashboard reactivo web (HTML, CSS y JS modular)
├── docker/
│   ├── Dockerfile             # Entorno de evaluación limpio y reproducible
│   └── docker-compose.yml     # Despliegue en un comando
├── models/                    # Directorio de modelos cuantizados ONNX (INT8 / FP16)
├── tests/                     # Suite de pruebas unitarias y de integración
├── mock_stream.py             # Simulador desacoplado para desarrollo independiente
├── run_edge.py                # Script CLI de ejecución maestro
├── SETUP_AMD.md               # Guía técnica de drivers e instalación en silicio AMD
└── requirements.txt           # Dependencias de producción y evaluación
```

---

## 5. Roles y Asignación de Frentes de Trabajo

* **Frente A • Core IA & Silicio:**
  * **Emilio (Titular):** Integración con runtimes AMD (DirectML, Vitis AI EP, ROCm), profiling de hardware y script maestro `run_edge.py`.
  * **Angela (Co-Lead):** Preparación de modelos de fauna, pipeline de cuantización INT8/FP16, validación de mAP@50 y suite de benchmarking `benchmark.py`.
* **Frente B • Pipeline & Software:**
  * **Cisneros (Titular):** Arquitectura desacoplada `BaseInferenceEngine`, captura multihilo de video `video_capture.py`, orquestador `pipeline.py` y servidor `main_server.py`.
  * **Angel (Backup Cisneros):** Tracking de objetos `tracker.py`, lógica de conteo direccional `counting.py` y capa SQLite `storage.py`.
* **Frente C • Frontend & Despliegue:**
  * **Elida (Titular):** Dashboard interactivo en `frontend/`, visualizador de bounding boxes y panel en vivo de métricas para la demo del jurado.
  * **Mario (Backup Elida):** Docker reproducible, redacción del `README.md`, badges y producción del video demo de 3 minutos.

---

## 6. Comandos Útiles

```bash
# Ejecutar suite de pruebas unitarias
pytest -v

# Ejecutar benchmark de latencia y throughput
python run_edge.py --benchmark --device directml

# Ejecutar comparativa en vivo CPU vs. AMD por consola
python -m src.engine.benchmark --compare --runs 50

# Desplegar con Docker
docker compose -f docker/docker-compose.yml up --build
```
