import pytest
import numpy as np
from src.engine.base_engine import Detection, MockInferenceEngine
from src.engine.inference_amd import AMDInferenceEngine


def test_detection_dataclass():
    det = Detection(bbox=(10, 20, 110, 120), confidence=0.95, class_id=1, class_name="jaguar")
    assert det.center == (60.0, 70.0)
    data = det.to_dict()
    assert data["class_name"] == "jaguar"
    assert data["confidence"] == 0.95
    assert data["bbox"] == [10.0, 20.0, 110.0, 120.0]


def test_mock_inference_engine_execution():
    engine = MockInferenceEngine(device="directml")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, lat_ms = engine.predict(dummy_frame)

    assert isinstance(detections, list)
    assert len(detections) >= 1
    assert lat_ms > 0.0
    info = engine.get_hardware_info()
    assert info["is_accelerated"] is True


def test_amd_inference_engine_cpu_fallback():
    # Cuando no hay modelo ONNX o no hay hardware, debe activar fallback sin crashear
    engine = AMDInferenceEngine(model_path="non_existent.onnx", device="directml", use_mock_fallback=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, lat = engine.predict(dummy_frame)

    assert isinstance(detections, list)
    info = engine.get_hardware_info()
    assert info["is_fallback"] is True
