import pytest
import os
from src.storage.storage import SQLiteStorage
from src.engine.base_engine import Detection
from src.pipeline.counting import CrossingEvent


def test_sqlite_storage(tmp_path):
    db_file = str(tmp_path / "test_bio.db")
    storage = SQLiteStorage(db_path=db_file)

    session_id = "test_sess_001"
    storage.start_session(session_id, provider="DmlExecutionProvider", model_name="yolov8n", source="0")

    dets = [
        Detection(bbox=(10, 10, 50, 50), confidence=0.92, class_id=0, class_name="jaguar")
    ]
    storage.record_detections(session_id, dets, track_ids=[1])

    crossings = [
        CrossingEvent(track_id=1, class_name="jaguar", direction="ENTRANDO", confidence=0.92)
    ]
    storage.record_crossings(session_id, crossings)
    storage.log_telemetry(session_id, fps=48.5, latency_ms=20.1, cpu_percent=15.0)

    summary = storage.get_summary_stats(session_id)
    assert summary["total_detections"] == 1
    assert "jaguar" in summary["crossings_by_species"]
    assert summary["crossings_by_species"]["jaguar"]["ENTRANDO"] == 1
    assert summary["avg_fps"] == 48.5
