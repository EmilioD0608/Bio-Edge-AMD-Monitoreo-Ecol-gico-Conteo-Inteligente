# Technical Design Document (TDD) & Testing Specification
## Bio-Edge AMD • Monitoreo Ecológico & Conteo Inteligente en el Borde

**Hackathon AMD Acto 3**  
**Versión del Documento:** 1.0.0  
**Enfoque:** Especificación Técnica Rigurosa & Test-Driven Development (TDD)  

---

## 1. Contratos de Interfaces y Clases Principales

### 1.1 Motor de Inferencia (`src/engine/base_engine.py`)
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class Detection:
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox # [x1, y1, x2, y2] normalizado o en pixeles
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name
        }

class BaseInferenceEngine(ABC):
    """
    Contrato base desacoplado para cualquier motor de inferencia neuronal.
    Garantiza que el pipeline no dependa del silicio subyacente.
    """
    def __init__(self, model_path: str, device: str = "cpu", conf_threshold: float = 0.45, iou_threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.active_provider = "UNKNOWN"

    @abstractmethod
    def load_model(self) -> None:
        """Carga el modelo en memoria y enlaza el Execution Provider correspondiente."""
        pass

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Acondiciona la imagen de entrada (letterbox, RGB, normalización [0, 1], transpose)."""
        pass

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> Any:
        """Ejecuta el paso hacia adelante en el silicio especificado."""
        pass

    @abstractmethod
    def postprocess(self, outputs: Any, original_shape: Tuple[int, int]) -> List[Detection]:
        """Aplica Non-Maximum Suppression (NMS) y devuelve detecciones calibradas."""
        pass

    def predict(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Método plantilla que orquesta preprocess -> infer -> postprocess.
        Retorna (lista_de_detecciones, latencia_en_milisegundos).
        """
        pass

    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """Devuelve metadatos del proveedor de aceleración activo y capacidades."""
        pass
```

### 1.2 Implementación AMD & Fallback (`src/engine/inference_amd.py`)
* **Negociación de Proveedores (Priority Order):**
  1. `DmlExecutionProvider` (DirectML para GPUs AMD Radeon en Windows).
  2. `VitisAIExecutionProvider` (NPU Ryzen AI en Windows/Linux con librería XDNA).
  3. `ROCMExecutionProvider` (ROCm para GPUs AMD en Linux).
  4. `CPUExecutionProvider` (Fallback obligatorio e incondicional).
* **Manejo de Excepciones:** Si el runtime detecta incompatibilidad de DLL o ausencia de hardware, registra un aviso en el logger y degrada suavemente a `CPUExecutionProvider` sin provocar un `crash`.

---

## 2. Esquema de Base de Datos SQLite (`src/storage/storage.py`)

Para garantizar la autonomía total sin conexión a red, SQLite se configura en modo WAL (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`).

### 2.1 Tablas Principales (DDL)

```sql
-- Sesiones de ejecución del sistema en el borde
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    device_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    video_source TEXT NOT NULL
);

-- Registro individual de detecciones por frame
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    track_id INTEGER,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- Registro de cruces direccionales (conteo inteligente de fauna e intrusos)
CREATE TABLE IF NOT EXISTS crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    track_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('ENTRANDO', 'SALIENDO', 'ZONA_PROTEGIDA')),
    confidence REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- Muestreo periódico de telemetría y rendimiento
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fps REAL NOT NULL,
    latency_ms REAL NOT NULL,
    cpu_percent REAL,
    gpu_percent REAL,
    temperature_c REAL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- Índices de consulta rápida
CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_crossings_class_time ON crossings(class_name, timestamp);
```

---

## 3. Protocolos de Comunicación (WebSocket & REST API)

### 3.1 Streaming de Video y Anotaciones (`/ws/stream`)
* **Cadencia:** 30+ FPS adaptativo.
* **Payload JSON + Frame:**
```json
{
  "type": "frame",
  "frame_id": 14205,
  "timestamp": 1728860400.123,
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "detections": [
    {
      "track_id": 4,
      "class_name": "jaguar",
      "confidence": 0.942,
      "bbox": [120, 310, 450, 680],
      "direction_status": "ENTRANDO"
    }
  ],
  "metrics": {
    "fps": 48.2,
    "latency_ms": 19.8,
    "provider": "DmlExecutionProvider (AMD Radeon)"
  }
}
```

### 3.2 Telemetría en Vivo (`/ws/telemetry`)
* **Cadencia:** 2 Hz (cada 500 ms).
```json
{
  "type": "telemetry",
  "timestamp": "2026-10-12T22:30:00Z",
  "device": "AMD Ryzen 7 + Radeon Graphics",
  "active_provider": "DmlExecutionProvider",
  "instant_fps": 49.1,
  "avg_latency_ms": 20.3,
  "total_crossings": {
    "jaguar": {"entrando": 3, "saliendo": 2},
    "venado": {"entrando": 12, "saliendo": 8},
    "humano_intruso": {"entrando": 1, "saliendo": 1}
  },
  "hardware_load": {
    "cpu_percent": 18.5,
    "gpu_memory_used_mb": 420.0
  }
}
```

### 3.3 Endpoints REST Principales
| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Estado del servicio, hardware detectado y tiempo de actividad. |
| `GET` | `/api/stats/summary` | Conteo acumulado por especie, total de cruces y eventos de intrusión. |
| `GET` | `/api/benchmarks/compare` | Métricas comparativas históricas CPU vs. Acelerador AMD. |
| `POST` | `/api/config/roi` | Configuración dinámica de líneas de cruce virtual y coordenadas ROI. |

---

## 4. Estrategia de Pruebas Unitarias y de Integración (TDD)

El desarrollo sigue la metodología Test-Driven Development (TDD):

```mermaid
flowchart LR
    Red["1. RED: Escribir Prueba Fallida"] --> Green["2. GREEN: Implementar Código Mínimo"]
    Green --> Refactor["3. REFACTOR: Optimizar y Acelerar"]
    Refactor --> Benchmark["4. BENCHMARK: Medir Latencia en Silicio"]
```

### 4.1 Matriz de Pruebas Automatizadas

1. **`tests/test_base_engine.py`:**
   - Verifica que clases abstractas impongan el contrato `predict`, `preprocess`, `postprocess`.
   - Verifica que el fallback a CPU se active si se pasa un hardware no soportado.
2. **`tests/test_tracker.py`:**
   - Verifica asociación correcta de centroides entre fotogramas consecutivos.
   - Verifica la eliminación de IDs tras $N$ fotogramas desaparecidos (`max_disappeared`).
3. **`tests/test_counting.py`:**
   - Simula trayectorias que cruzan una línea de izquierda a derecha (`ENTRANDO`) y viceversa (`SALIENDO`).
   - Verifica que un objeto estacionario dentro de la zona no compute múltiples cruces erróneos.
4. **`tests/test_storage.py`:**
   - Verifica la creación de tablas SQLite y el registro de sesiones, detecciones y cruces.
   - Verifica la consistencia transaccional y consultas de resumen por especie.
5. **`tests/test_api.py`:**
   - Verifica respuesta 200 en endpoints `/api/health`, `/api/stats/summary`.
   - Verifica conexión exitosa a WebSockets `/ws/stream` y `/ws/telemetry`.
6. **`tests/test_benchmark.py`:**
   - Verifica el cálculo de percentiles P50, P95, P99 y FPS en el módulo `benchmark.py`.

---

## 5. Protocolo de Benchmarking y Validación de Silicio AMD

El script `src/engine/benchmark.py` ejecuta el siguiente protocolo estandarizado:
1. **Warmup Phase:** 20 ejecuciones de descarte para estabilizar cachés de memoria y compilación JIT/DirectML de grafos.
2. **Execution Phase:** 100 iteraciones consecutivas midiendo latencia por fotograma con `time.perf_counter_ns()`.
3. **Métricas Calculadas:**
   $$\text{Latencia Media} = \frac{1}{N}\sum_{i=1}^N L_i$$
   $$\text{P50 (Mediana)}, \quad \text{P95 (Cola)}, \quad \text{P99 (Peor caso)}$$
   $$\text{Throughput} = \frac{1000}{\text{Latencia Media (ms)}} \text{ FPS}$$
4. **Exportación:** Reporte estructurado en formato JSON y Markdown para visualización en el dashboard y entrega al jurado.
