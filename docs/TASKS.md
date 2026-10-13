# Plan de Asignación de Tareas, Roles y Cronograma Operativo
## Bio-Edge AMD • Monitoreo Ecológico & Conteo Inteligente en el Borde

**Hackathon AMD Acto 3**  
**Ventana de Ejecución:** 12 al 18 de Octubre  
**Estrategia Operativa:** Pairing Asíncrono, Desacoplamiento Estricto y Cero Bloqueos  

---

## 1. Estructura de Frentes y Responsabilidades Principales

El equipo está organizado en tres frentes de trabajo paralelos y desacoplados para evitar dependencias bloqueantes:

| Frente | Nombre del Frente | Líder Titular | Co-Lead / Backup | Alcance Principal |
| :---: | :--- | :--- | :--- | :--- |
| **A** | **Core IA & Silicio** | **Emilio** | **Angela** | Optimización y cuantización de modelos, aceleración en silicio AMD (DirectML, NPU Ryzen AI con Vitis AI EP, ROCm) y suite de benchmarking. |
| **B** | **Pipeline & Software** | **Cisneros** | **Angel** | Arquitectura modular desacoplada, captura multihilo de video, tracking por centroides/ByteTrack, conteo direccional, persistencia SQLite y API FastAPI. |
| **C** | **Frontend & Despliegue** | **Elida** | **Mario** | Dashboard reactivo web para streaming y telemetría, panel interactivo comparativo CPU vs. AMD, Docker reproducible, documentación y video demo. |

---

## 2. Asignación Exhaustiva de Tareas por Participante

### 2.1 Emilio — Líder Técnico & Hardware Acceleration Lead (Frente A - Titular)
* **Rol:** Responsable de la integración y optimización con el hardware y silicio de AMD.
* **Tareas Técnicas Directas:**
  1. Configurar y validar el entorno de drivers de hardware AMD (ROCm en Linux o NPU Ryzen AI con Vitis AI EP / DirectML en Windows).
  2. Implementar el wrapper de inferencia en ONNX Runtime (`src/engine/inference_amd.py`) para inicializar proveedores de aceleración de AMD (`DmlExecutionProvider`, `VitisAIExecutionProvider`, `ROCMExecutionProvider`) con fallback transparente a CPU.
  3. Realizar profiling de hardware: registrar porcentaje de uso de NPU/GPU, consumo de VRAM y temperatura operativa durante inferencia continua.
  4. Construir y mantener el script de ejecución maestro `run_edge.py` con banderas dinámicas (`--device`, `--source`, `--conf`, `--mock`).
* **Entregables:**
  - Módulo `src/engine/inference_amd.py` completamente funcional y reproducible.
  - Matriz de compatibilidad y guía de configuración técnica documentada en `SETUP_AMD.md`.
* **Protocolo de Cobertura:** Si Emilio no está disponible, Angela asume la ejecución de inferencia y profiling en la máquina con silicio AMD.

---

### 2.2 Angela — Especialista en IA (Co-Lead Modelos & Benchmarks) (Frente A - Co-Lead)
* **Rol:** Preparación, calibración, cuantización y validación métrica de los modelos neuronales.
* **Tareas Técnicas Directas:**
  1. Seleccionar y preparar el modelo base de visión (YOLOv8-nano / FastViT) preentrenado para fauna silvestre y detección de intrusos.
  2. Ejecutar el pipeline de cuantización de precisión (FP32 a INT8 y FP16) mediante herramientas de Vitis AI Quantizer o ONNX Runtime Quantization.
  3. Evaluar la degradación de precisión asegurando la retención de mAP@50 superior al 94% tras la cuantización INT8.
  4. Diseñar y programar la suite de benchmarking automatizado (`src/engine/benchmark.py`) para calcular latencia P50/P95/P99 y throughput en FPS.
* **Entregables:**
  - Modelos optimizados en formato `.onnx` (FP16 e INT8 listos para ejecución en borde).
  - Reporte cuantitativo con tablas comparativas de throughput (FPS) y latencia (ms) exportado a Markdown/JSON.
* **Protocolo de Cobertura:** Respalda a Emilio en la ejecución sobre silicio AMD; provee a Cisneros los artefactos del modelo listos para consumo sin fricción.

---

### 2.3 Cisneros — Arquitectura de Software & Backend (Frente B - Titular)
* **Rol:** Diseño de la arquitectura de software desacoplada, pipeline de video y capa de servicios web.
* **Tareas Técnicas Directas:**
  1. Diseñar la arquitectura de software modular en Python: clase abstracta `BaseInferenceEngine` desacoplada del proveedor de hardware.
  2. Implementar el bucle de captura y preprocesamiento de video con OpenCV multihilo (`src/pipeline/video_capture.py`) para eliminar cuellos de botella de lectura I/O.
  3. Desarrollar la API en FastAPI (`src/api/main_server.py`) exponiendo endpoints de telemetría y WebSocket para streaming de frames a más de 30 FPS.
  4. Gestionar el ciclo de vida de la aplicación, control de señales del sistema (`SIGINT`/`SIGTERM`) y manejo robusto ante cortes del flujo de video.
* **Entregables:**
  - Módulo central de orquestación `src/pipeline/pipeline.py` y servidor `src/api/main_server.py`.
  - Documentación interactiva de endpoints en OpenAPI/Swagger accesible en `/docs`.
* **Protocolo de Cobertura:** Si Cisneros se ausenta, Angel asume el mantenimiento de las rutas de la API y la conexión con el bucle de captura.

---

### 2.4 Angel — Lógica de Visión & Tracking (Frente B - Backup Cisneros)
* **Rol:** Seguimiento espacio-temporal de especímenes, reglas de conteo direccional y persistencia local.
* **Tareas Técnicas Directas:**
  1. Implementar tracking de objetos mediante seguimiento por centroides o ByteTrack liviano (`src/pipeline/tracker.py`) para asignar IDs únicos persistentes.
  2. Programar el motor de conteo direccional (`src/pipeline/counting.py`) por cruce de líneas virtuales o zonas poligonales de interés (ROI).
  3. Configurar almacenamiento persistente en SQLite local (`src/storage/storage.py`) en modo WAL para registrar marcas de tiempo, clase, confianza y trayectorias sin pérdida de datos.
* **Entregables:**
  - Módulos `src/pipeline/tracker.py`, `src/pipeline/counting.py` y `src/storage/storage.py` verificados con pruebas unitarias y datos simulados.
* **Protocolo de Cobertura:** Cisneros valida la integración de tracking en el pipeline central; Angel puede sustituir a Cisneros en ajustes del backend.

---

### 2.5 Elida — Frontend & Dashboard Reactivo (Frente C - Titular)
* **Rol:** Interfaz web visual, paneles de telemetría e interactividad para la evaluación del jurado.
* **Tareas Técnicas Directas:**
  1. Diseñar el dashboard web interactivo en `frontend/` con visualización en tiempo real del stream de video y bounding boxes superpuestos.
  2. Construir el módulo de telemetría: contadores dinámicos por especie detectada y gráfico de distribución temporal de cruces.
  3. Crear el panel visual de benchmarks para la demo del jurado (comparativa gráfica en vivo: latencia CPU vs. AMD, FPS y uso de recursos).
  4. Integrar consumo de WebSocket en vivo y soporte para simulador mock (`mock_stream.py`).
* **Entregables:**
  - Directorio `frontend/` completamente funcional, responsive y listo para desplegar vía navegador.
* **Protocolo de Cobertura:** Mario asiste con estilos visuales y armado de componentes UI si Elida requiere apoyo.

---

### 2.6 Mario — DevOps, Documentación & Video Demo (Frente C - Backup Elida)
* **Rol:** Empaquetado reproducible, documentación de primer nivel para evaluadores y producción audiovisual.
* **Tareas Técnicas Directas:**
  1. Construir el `Dockerfile` y configuración Docker Compose (`docker/`) garantizando reproducibilidad en máquinas limpias de los evaluadores.
  2. Redactar el `README.md` estructurado con guía de instalación en 3 pasos, diagramas de arquitectura, especificaciones de hardware y badges de AMD.
  3. Grabar, editar y producir el video demo de 3 minutos destacando la ejecución en silicio AMD, métricas en vivo y el dashboard final.
  4. Realizar auditorías de compilación y pruebas de clonación limpia.
* **Entregables:**
  - Contenedor reproducible testeado en máquina limpia.
  - Video demo en alta resolución (1080p, 3 minutos) listo para subir a la plataforma del hackathon.
* **Protocolo de Cobertura:** Respalda a Elida en frontend; Angela audita la documentación técnica generada.

---

## 3. Cronograma Diario de Ejecución (12 al 18 de Octubre)

| Fecha | Hito Principal | Tareas Operativas por Frente | Criterio de Éxito / Entregable |
| :--- | :--- | :--- | :--- |
| **Lun 12** | **Setup & Baseline** | • Repo y estructura configurados.<br>• Angela y Emilio exportan modelo base a ONNX y miden CPU baseline.<br>• Cisneros inicia clases base y captura multihilo. | Repositorio inicializado, `BaseInferenceEngine` definida y baseline CPU medido. |
| **Mar 13** | **Aceleración AMD** | • Emilio activa Execution Provider AMD (DirectML / Vitis AI / ROCm).<br>• Angela cuantiza modelo a INT8/FP16.<br>• Cisneros y Angel arman captura y tracker.<br>• Elida inicia UI con mock data. | Inferencia funcional en hardware AMD con fallback a CPU. Mock stream alimentando frontend. |
| **Mié 14** | **Tracking & API** | • Angel conecta tracking y persistencia SQLite.<br>• Cisneros expone streaming por WebSockets a 30+ FPS.<br>• Mario prepara Dockerfile preliminar y valida dependencias. | Flujo de video con bounding boxes e IDs únicos emitido por WebSocket; persistencia en SQLite activa. |
| **Jue 15** | **Integración Total** | • Pipeline unificado en vivo: Captura $\rightarrow$ Inferencia AMD $\rightarrow$ Tracking $\rightarrow$ FastAPI $\rightarrow$ Dashboard Elida.<br>• Depuración de I/O y latencias. | Sistema completo de punta a punta operando de forma continua sin caídas ni fugas de memoria. |
| **Vie 16** | **Benchmarks & Freeze** | • Angela y Emilio ejecutan suite de benchmarking final (P50/P95/P99, FPS, VRAM).<br>• Pruebas de degradación y robustez.<br>• **Code Freeze estricto a las 18:00 hrs**. | Reporte formal de métricas generado; código congelado en rama principal `main`. |
| **Sáb 17** | **Video Demo & Docs** | • Mario y Elida graban y editan el video de pitch (máx 3 minutos).<br>• Redacción final de `README.md` con tablas comparativas y badges AMD. | Video demo en alta calidad renderizado; documentación lista para evaluación externa. |
| **Dom 18** | **Release & Entrega** | • Prueba de clonación limpia en equipo ajeno.<br>• Verificación de permisos de repositorio y enlaces.<br>• Envío formal antes de la hora límite. | Release tag v1.0.0 creado; entrega del proyecto registrada con éxito. |

---

## 4. Políticas de Autonomía y Cero Bloqueos

1. **Fallback Mandatorio a CPU:**  
   Todo comando y módulo acepta `--device cpu` por defecto. Si un desarrollador no cuenta con tarjeta gráfica o NPU AMD en su estación de trabajo local, puede programar y testear el 100% del software sin impedimentos.

2. **Desacoplamiento con Mocks (`mock_stream.py`):**  
   El frontend consume un simulador local que emite frames sintéticos y telemetría de prueba con la misma especificación WebSocket, lo que permite a Elida y Mario desarrollar la interfaz visual desde el día 1 sin esperar que el pipeline de IA esté finalizado.

3. **Standup Asíncrono Diario:**  
   Cada miembro reporta al cierre de jornada:
   - ¿Qué completó hoy? (con enlace al commit correspondiente).
   - ¿Qué tarea acometerá mañana?
   - ¿Qué bloqueo o riesgo identifica?

4. **Matriz de Cobertura Inmediata (Pairing Asíncrono):**  
   - Si falta Emilio $\rightarrow$ cubre Angela.
   - Si falta Cisneros $\rightarrow$ cubre Angel.
   - Si falta Elida $\rightarrow$ cubre Mario.  
   *(Y viceversa dentro de cada frente).*
