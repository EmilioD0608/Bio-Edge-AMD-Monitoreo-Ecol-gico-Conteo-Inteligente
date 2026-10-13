"""
Base Inference Engine and Data Contracts for Bio-Edge AMD.
Diseñado para desacoplar el pipeline de visión del runtime o silicio específico.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np


class Detection:
    """Representa una detección de objeto/fauna delimitada por un Bounding Box."""
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str
    ):
        # bbox en formato [x1, y1, x2, y2]
        self.bbox = tuple(map(float, bbox))
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

    @property
    def center(self) -> Tuple[float, float]:
        """Calcula el centroide (cx, cy) del bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": [round(c, 2) for c in self.center]
        }


class BaseInferenceEngine(ABC):
    """
    Clase abstracta que define el contrato de inferencia neuronal para Bio-Edge AMD.
    Cualquier acelerador (DirectML, Vitis AI, ROCm o CPU) implementa esta interfaz.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        target_size: Tuple[int, int] = (640, 640)
    ):
        self.model_path = model_path
        self.device = device.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.active_provider: str = "Uninitialized"
        self.class_names: List[str] = [
            "jaguar", "tapir", "venado", "pecari", "mono_arana",
            "ocelote", "humano_intruso", "vehiculo_furtivo"
        ]

    @abstractmethod
    def load_model(self) -> None:
        """Carga los pesos/grafo del modelo e inicializa el proveedor de hardware."""
        pass

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocesa el frame para el modelo (redimensionamiento, normalización, transposicion).
        Retorna (tensor_listo, ratio_escala, (pad_w, pad_h)).
        """
        pass

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> Any:
        """Ejecuta el grafo computacional en el hardware seleccionado."""
        pass

    @abstractmethod
    def postprocess(
        self,
        raw_outputs: Any,
        scale_ratio: float,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Aplica filtrado por confianza y NMS para extraer detecciones finales."""
        pass

    def predict(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Método plantilla que orquesta la inferencia completa y mide latencia exacta en milisegundos.
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        tensor, scale, pad = self.preprocess(frame)
        outputs = self.infer(tensor)
        detections = self.postprocess(outputs, scale, pad, (orig_h, orig_w))

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return detections, latency_ms

    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """Obtiene información del proveedor de silicio, dispositivo y estado operativo."""
        pass


class MockInferenceEngine(BaseInferenceEngine):
    """
    Motor sintético de alta fidelidad para pruebas sin hardware o sin modelo binario.
    Genera detecciones coherentes y simula latencias según el dispositivo elegido.
    """

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(model_path="mock://wildlife_v1", device=device, **kwargs)
        self.active_provider = f"MockEngine ({self.device.upper()})"
        self._step = 0

    def load_model(self) -> None:
        pass

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        return np.zeros((1, 3, 640, 640), dtype=np.float32), 1.0, (0, 0)

    def infer(self, input_tensor: np.ndarray) -> Any:
        # Simula latencia del hardware especificado
        if self.device in ["directml", "rocm", "vitisai", "amd"]:
            # Simula acelerador AMD (~15-22 ms)
            time.sleep(0.018)
        else:
            # Simula CPU estándar (~85-115 ms)
            time.sleep(0.095)
        return []

    def postprocess(
        self,
        raw_outputs: Any,
        scale_ratio: float,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        self._step += 1
        h, w = original_shape
        # Generar fauna sintética en movimiento
        x_pos = int((self._step * 15) % (w - 200)) + 50
        y_pos = int(h * 0.45 + np.sin(self._step * 0.2) * 50)

        species_cycle = ["jaguar", "tapir", "venado", "mono_arana"]
        specie = species_cycle[(self._step // 30) % len(species_cycle)]

        detections = [
            Detection(
                bbox=(x_pos, y_pos, x_pos + 120, y_pos + 90),
                confidence=0.92,
                class_id=0,
                class_name=specie
            )
        ]

        if self._step % 50 > 30:
            detections.append(
                Detection(
                    bbox=(w - 180, int(h * 0.6), w - 60, int(h * 0.85)),
                    confidence=0.87,
                    class_id=6,
                    class_name="humano_intruso"
                )
            )

        return detections

    def get_hardware_info(self) -> Dict[str, Any]:
        simulated_lat = "<25 ms" if "amd" in self.device or self.device in ["directml", "rocm", "vitisai"] else ">100 ms"
        return {
            "device": self.device,
            "provider": self.active_provider,
            "is_accelerated": self.device in ["directml", "rocm", "vitisai", "amd"],
            "expected_latency": simulated_lat,
            "status": "MOCK_ACTIVE"
        }
