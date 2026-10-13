"""
Captura de Video Multihilo con Control de Latencia para Bio-Edge AMD.
Diseñado por Cisneros (Frente B) para eliminar cuellos de botella de I/O en OpenCV.
"""

import time
import threading
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("BioEdge.VideoCapture")


class VideoCaptureThread:
    """
    Lee continuamente fotogramas de una cámara RTSP, Webcam o archivo en un hilo separado.
    Mantiene siempre el fotograma más reciente en memoria (drop-oldest), previniendo lag acumulado.
    """

    def __init__(self, source: str = "0", target_fps: int = 30):
        # Si source es un dígito, convertir a int para webcam USB
        self.source = int(source) if str(source).isdigit() else source
        self.target_fps = target_fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._is_mock_feed = False
        self._mock_step = 0
        self.fps_actual: float = 0.0

    def start(self) -> "VideoCaptureThread":
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="BioEdge-CaptureThread")
        self._thread.start()
        logger.info("Hilo de captura de video iniciado para la fuente: %s", self.source)
        return self

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Genera un fotograma sintético cuando no hay cámara física disponible."""
        self._mock_step += 1
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Gradiente de fondo tipo reserva natural nocturna/diurna
        frame[:, :, 0] = 20  # B
        frame[:, :, 1] = int(35 + 10 * np.sin(self._mock_step * 0.05))  # G
        frame[:, :, 2] = 20  # R

        # Patrón de prueba con coordenadas
        import cv2
        cv2.putText(
            frame,
            f"Bio-Edge AMD Live Stream [Mock Feed] Frame: {self._mock_step}",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 180),
            2
        )
        cv2.putText(
            frame,
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        # Elemento móvil simulando fauna
        cx = int((self._mock_step * 10) % (w - 150)) + 60
        cy = int(h * 0.5 + np.cos(self._mock_step * 0.1) * 60)
        cv2.rectangle(frame, (cx, cy), (cx + 100, cy + 80), (40, 200, 80), -1)
        cv2.putText(frame, "Fauna", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _capture_loop(self) -> None:
        try:
            import cv2
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.warning("No se pudo abrir la cámara en '%s'. Usando generador sintético.", self.source)
                self._is_mock_feed = True
        except Exception as e:
            logger.warning("Error inicializando OpenCV VideoCapture (%s). Activando generador sintético.", e)
            self._is_mock_feed = True
            cap = None

        frame_interval = 1.0 / max(1, self.target_fps)
        last_time = time.perf_counter()
        frames_counted = 0
        fps_timer = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()

            if self._is_mock_feed or cap is None or not cap.isOpened():
                frame = self._generate_synthetic_frame()
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Pérdida de señal de video. Reintentando...")
                    time.sleep(0.5)
                    continue

            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_timestamp = time.perf_counter()

                frames_counted += 1
                if time.perf_counter() - fps_timer >= 1.0:
                    self.fps_actual = frames_counted / (time.perf_counter() - fps_timer)
                    frames_counted = 0
                    fps_timer = time.perf_counter()

            # Throttle para respetar framerate deseado
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if cap is not None:
            cap.release()
        logger.info("Hilo de captura detenido correctamente.")

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Retorna (éxito, fotograma_más_reciente, timestamp)."""
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._frame_timestamp

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
