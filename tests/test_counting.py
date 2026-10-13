import pytest
from src.engine.base_engine import Detection
from src.pipeline.tracker import TrackedObject
from src.pipeline.counting import DirectionalCounter


def test_directional_counter_crossing():
    # Línea vertical en x=300
    counter = DirectionalCounter(line_start=(300, 0), line_end=(300, 500), orientation="vertical")

    # Objeto que cruza de izquierda a derecha (x=250 -> x=350)
    det1 = Detection(bbox=(240, 100, 260, 140), confidence=0.9, class_id=0, class_name="tapir")
    obj = TrackedObject(track_id=1, detection=det1)

    # Frame 2: ya cruzó la línea
    det2 = Detection(bbox=(340, 100, 360, 140), confidence=0.9, class_id=0, class_name="tapir")
    obj.update(det2)

    events = counter.update([obj])
    assert len(events) == 1
    assert events[0].direction == "ENTRANDO"
    assert events[0].class_name == "tapir"

    summary = counter.get_summary()
    assert summary["total_in"] == 1
    assert summary["total_out"] == 0

    # Si se actualiza de nuevo en el mismo lugar, no debe contar doble
    events_again = counter.update([obj])
    assert len(events_again) == 0
