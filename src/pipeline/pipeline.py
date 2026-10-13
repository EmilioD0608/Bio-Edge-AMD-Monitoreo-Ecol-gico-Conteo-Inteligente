"""
Módulo Central del Pipeline para Bio-Edge AMD.
Diseñado por Cisneros (Frente B).
Orquesta Captura Multihilo -> Inferencia Acelerada AMD -> Tracking -> Conteo -> Almacenamiento.
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

from ..engine.base_engine import BaseInferenceEngine
from ..engine.inference_amd import AMDInferenceEngine
from .video_capture import VideoCaptureThread
from .tracker import CentroidTracker
from .counting import DirectionalCounter
from ..storage.storage import SQLiteStorage

logger = logging.getLogger("BioEdge.Pipeline")


class EdgePipeline:
    """Orquestador integral del sistema de visión y telemetría de borde."""

    def __init__(
        self,
        source: str = "0",
        model_path: Optional[str] = None,
        device: str = "directml",
        conf_threshold: float = 0.45,
        db_path: str = "data/bio_edge.db",
        line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ):
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.source = source
        self.device = device
        self.conf_threshold = conf_threshold

        # Componentes del pipeline
        self.engine: BaseInferenceEngine = AMDInferenceEngine(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold
        )
        self.capture = VideoCaptureThread(source=source)
        self.tracker = CentroidTracker(max_disappeared=20, max_distance=150.0)

        # Configuración de línea de cruce virtual
        if line_coords:
            p1, p2 = line_coords
            self.counter = DirectionalCounter(line_start=p1, line_end=p2)
        else:
            self.counter = DirectionalCounter(line_start=(640, 0), line_end=(640, 720))

        self.storage = SQLiteStorage(db_path=db_path)
        self.is_running = False

        # Métricas en tiempo real
        self.fps_metric: float = 0.0
        self.latency_metric: float = 0.0
        self.frame_count: int = 0

    def start(self) -> "EdgePipeline":
        logger.info("Iniciando EdgePipeline con ID de sesión: %s", self.session_id)
        self.storage.start_session(
            session_id=self.session_id,
            provider=self.engine.active_provider,
            model_name=self.engine.model_path or "DefaultModel",
            source=str(self.source)
        )
        self.capture.start()
        self.is_running = True
        return self

    def process_step(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Ejecuta un ciclo completo sobre el fotograma más reciente.
        Retorna (exito, frame_anotado, telemetria_dict).
        """
        if not self.is_running:
            return False, None, {}

        success, frame, _ = self.capture.read()
        if not success or frame is None:
            return False, None, {}

        h, w = frame.shape[:2]

        # 1. Inferencia Neuronal en Silicio AMD (o Fallback)
        detections, latency_ms = self.engine.predict(frame)
        self.latency_metric = latency_ms

        # 2. Tracking de Entidades
        tracked_objects = self.tracker.update(detections)

        # 3. Conteo Direccional por Cruce
        crossing_events = self.counter.update(tracked_objects)

        # 4. Persistencia en SQLite
        track_ids = [obj.track_id for obj in tracked_objects]
        self.storage.record_detections(self.session_id, detections, track_ids)
        if crossing_events:
            self.storage.record_crossings(self.session_id, crossing_events)

        # 5. Renderizado Visual sobre el Frame para el Dashboard
        annotated = self._render_annotations(frame, tracked_objects, crossing_events)

        # 6. Actualización de Métricas de Cadencia
        self.frame_count += 1
        self.fps_metric = self.capture.fps_actual

        if self.frame_count % 30 == 0:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            self.storage.log_telemetry(self.session_id, self.fps_metric, self.latency_metric, cpu_usage)

        telemetry = {
            "session_id": self.session_id,
            "frame_id": self.frame_count,
            "fps": round(self.fps_metric, 1),
            "latency_ms": round(self.latency_metric, 1),
            "hardware": self.engine.get_hardware_info(),
            "tracked_count": len(tracked_objects),
            "summary_counts": self.counter.get_summary()
        }

        return True, annotated, telemetry

    def _render_annotations(self, frame: np.ndarray, tracked_objects: list, crossings: list) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Dibujar línea virtual de cruce
        p1 = self.counter.line_start
        p2 = self.counter.line_end
        cv2.line(vis, p1, p2, (0, 255, 255), 2)
        cv2.putText(vis, "LINEA DE MONITOREO VIRTUAL", (p1[0] - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Dibujar bounding boxes y trayectorias
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            is_threat = "intruso" in obj.class_name or "vehiculo" in obj.class_name
            color = (0, 0, 240) if is_threat else (0, 230, 70)

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Etiqueta de ID y clase
            label = f"#{obj.track_id} {obj.class_name} ({int(obj.confidence * 100)}%)"
            cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + len(label) * 9, y1), color, -1)
            cv2.putText(vis, label, (x1 + 3, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Trayectoria histórica
            if len(obj.history) > 1:
                pts = np.array(obj.history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], False, color, 1)

        # Overlay HUD de Métricas Bio-Edge
        hud_bg = vis.copy()
        cv2.rectangle(hud_bg, (10, 10), (330, 110), (15, 15, 15), -1)
        cv2.addWeighted(hud_bg, 0.75, vis, 0.25, 0, vis)

        prov = self.engine.active_provider
        prov_label = "AMD SILICON" if "Dml" in prov or "Vitis" in prov or "ROCM" in prov else "CPU BASELINE"
        status_color = (0, 255, 120) if "AMD" in prov_label else (0, 165, 255)

        cv2.putText(vis, f"BIO-EDGE AMD • {prov_label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        cv2.putText(vis, f"FPS: {self.fps_metric:.1f} | Latencia: {self.latency_metric:.1f} ms", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        summary = self.counter.get_summary()
        cv2.putText(vis, f"Total In: {summary['total_in']} | Total Out: {summary['total_out']}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return vis

    def stop(self) -> None:
        self.is_running = False
        self.capture.stop()
        logger.info("EdgePipeline detenido.")
