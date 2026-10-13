"""
AMD Silicon Inference Engine for Bio-Edge AMD.
Integra DirectML, Vitis AI (NPU Ryzen AI), ROCm y fallback transparente a CPU.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .base_engine import BaseInferenceEngine, Detection, MockInferenceEngine

logger = logging.getLogger("BioEdge.AMDInference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class AMDInferenceEngine(BaseInferenceEngine):
    """
    Motor de inferencia acelerado por silicio AMD vía ONNX Runtime.
    Soporta fallback transparente a CPU bajo la política de Cero Bloqueos.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "directml",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        target_size: Tuple[int, int] = (640, 640),
        use_mock_fallback: bool = True
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_size=target_size
        )
        self.session = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self.use_mock_fallback = use_mock_fallback
        self.mock_engine: Optional[MockInferenceEngine] = None
        self.load_model()

    def _resolve_providers(self) -> List[Any]:
        """Determina la lista priorizada de Execution Providers para ONNX Runtime."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info("Proveedores ONNX Runtime disponibles en el sistema: %s", available)
        except ImportError:
            logger.warning("onnxruntime no está instalado en este entorno. Activando modo mock.")
            return ["CPUExecutionProvider"]

        selected = []
        dev = self.device.lower()

        if dev in ["directml", "dml", "amd_gpu"]:
            if "DmlExecutionProvider" in available:
                selected.append("DmlExecutionProvider")
            else:
                logger.warning("DmlExecutionProvider no disponible. Verifique onnxruntime-directml.")

        elif dev in ["vitisai", "npu", "ryzen_ai"]:
            if "VitisAIExecutionProvider" in available:
                selected.append("VitisAIExecutionProvider")
            else:
                logger.warning("VitisAIExecutionProvider no detectado. Verifique Ryzen AI NPU Software.")

        elif dev in ["rocm", "linux_gpu"]:
            if "ROCMExecutionProvider" in available:
                selected.append("ROCMExecutionProvider")
            else:
                logger.warning("ROCMExecutionProvider no disponible en esta plataforma.")

        elif dev in ["amd", "auto"]:
            # Auto-detección de silicio AMD disponible
            for ep in ["DmlExecutionProvider", "VitisAIExecutionProvider", "ROCMExecutionProvider"]:
                if ep in available:
                    selected.append(ep)
                    break

        # Fallback incondicional a CPU para evitar bloqueos
        selected.append("CPUExecutionProvider")
        return selected

    def load_model(self) -> None:
        """Inicializa la sesión ONNX o activa el fallback simulado si no existe el archivo."""
        if not self.model_path or not os.path.exists(self.model_path):
            if self.use_mock_fallback:
                logger.warning(
                    "Modelo no encontrado en '%s'. Activando MockInferenceEngine con perfil '%s'.",
                    self.model_path, self.device
                )
                self.mock_engine = MockInferenceEngine(device=self.device)
                self.active_provider = self.mock_engine.active_provider
                return
            else:
                raise FileNotFoundError(f"Archivo de modelo no encontrado: {self.model_path}")

        try:
            import onnxruntime as ort
            providers = self._resolve_providers()
            logger.info("Inicializando sesión ONNX con proveedores: %s", providers)

            # Opciones de sesión optimizadas para baja latencia
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
            self.active_provider = self.session.get_providers()[0]
            logger.info("Sesión inicializada exitosamente en silicio: %s", self.active_provider)

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
        except Exception as e:
            logger.error("Error al cargar modelo en ONNX Runtime (%s). Activando fallback.", e)
            if self.use_mock_fallback:
                self.mock_engine = MockInferenceEngine(device=self.device)
                self.active_provider = f"Fallback ({self.mock_engine.active_provider})"
            else:
                raise

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        if self.mock_engine:
            return self.mock_engine.preprocess(frame)

        import cv2
        ih, iw = frame.shape[:2]
        th, tw = self.target_size
        scale = min(th / ih, tw / iw)
        nh, nw = int(ih * scale), int(iw * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_top = (th - nh) // 2
        pad_left = (tw - nw) // 2
        padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        # BGR a RGB, normalización [0, 1] y canal primero (1, 3, H, W)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, (pad_left, pad_top)

    def infer(self, input_tensor: np.ndarray) -> Any:
        if self.mock_engine:
            return self.mock_engine.infer(input_tensor)

        return self.session.run(self.output_names, {self.input_name: input_tensor})

    def postprocess(
        self,
        raw_outputs: Any,
        scale_ratio: float,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        if self.mock_engine:
            return self.mock_engine.postprocess(raw_outputs, scale_ratio, padding, original_shape)

        # Decodificación de salidas de YOLOv8 estándar: shape (1, 84, 8400)
        output = raw_outputs[0]
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))  # -> (1, 8400, 84)

        predictions = output[0]  # (8400, 84)
        boxes = predictions[:, :4]
        scores_matrix = predictions[:, 4:]

        class_ids = np.argmax(scores_matrix, axis=1)
        confidences = np.max(scores_matrix, axis=1)

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        pad_left, pad_top = padding
        orig_h, orig_w = original_shape
        detections: List[Detection] = []

        for box, conf, cid in zip(boxes, confidences, class_ids):
            # Formato cx, cy, w, h
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2.0 - pad_left) / scale_ratio
            y1 = (cy - bh / 2.0 - pad_top) / scale_ratio
            x2 = (cx + bw / 2.0 - pad_left) / scale_ratio
            y2 = (cy + bh / 2.0 - pad_top) / scale_ratio

            x1 = max(0.0, min(float(x1), float(orig_w)))
            y1 = max(0.0, min(float(y1), float(orig_h)))
            x2 = max(0.0, min(float(x2), float(orig_w)))
            y2 = max(0.0, min(float(y2), float(orig_h)))

            cname = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cid, class_name=cname))

        return detections

    def get_hardware_info(self) -> Dict[str, Any]:
        is_amd = "Dml" in self.active_provider or "VitisAI" in self.active_provider or "ROCM" in self.active_provider
        return {
            "device": self.device,
            "provider": self.active_provider,
            "is_accelerated": is_amd,
            "is_fallback": self.mock_engine is not None or "CPU" in self.active_provider,
            "model_path": self.model_path or "MockWildlifeModel"
        }
