import pytest
from src.engine.base_engine import Detection
from src.pipeline.tracker import CentroidTracker


def test_tracker_registration_and_update():
    tracker = CentroidTracker(max_disappeared=5, max_distance=100.0)

    # Frame 1: Una detección
    d1 = [Detection(bbox=(100, 100, 150, 150), confidence=0.9, class_id=0, class_name="jaguar")]
    objects_f1 = tracker.update(d1)

    assert len(objects_f1) == 1
    t_id = objects_f1[0].track_id
    assert objects_f1[0].centroid == (125.0, 125.0)

    # Frame 2: El objeto se mueve ligeramente
    d2 = [Detection(bbox=(110, 105, 160, 155), confidence=0.88, class_id=0, class_name="jaguar")]
    objects_f2 = tracker.update(d2)

    assert len(objects_f2) == 1
    assert objects_f2[0].track_id == t_id  # Mantiene el mismo ID persistente
    assert len(objects_f2[0].history) == 2


def test_tracker_disappearance_deregistration():
    tracker = CentroidTracker(max_disappeared=2, max_distance=100.0)
    d1 = [Detection(bbox=(50, 50, 80, 80), confidence=0.9, class_id=1, class_name="venado")]
    tracker.update(d1)
    assert len(tracker.objects) == 1

    # Frames vacíos consecutivos
    tracker.update([])
    assert len(tracker.objects) == 1  # Todavía en periodo de gracia
    tracker.update([])
    tracker.update([])
    # Superó max_disappeared=2
    assert len(tracker.objects) == 0
