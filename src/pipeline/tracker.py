"""
Módulo de Rastreo de Objetos (Tracker) para Bio-Edge AMD.
Implementado por Angel (Frente B) mediante seguimiento por centroides e IDs persistentes.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import OrderedDict
from ..engine.base_engine import Detection


class TrackedObject:
    """Representa una entidad biológica rastreada a través del tiempo."""

    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.centroid = detection.center
        self.history: List[Tuple[float, float]] = [self.centroid]
        self.disappeared_frames = 0
        self.counted_events: set = set()

    def update(self, detection: Detection) -> None:
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.centroid = detection.center
        self.history.append(self.centroid)
        if len(self.history) > 30:
            self.history.pop(0)
        self.disappeared_frames = 0

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 1) for c in self.bbox],
            "centroid": [round(c, 1) for c in self.centroid],
            "trajectory_points": len(self.history)
        }


class CentroidTracker:
    """
    Rastreador liviano basado en distancias euclidianas entre centroides consecutivos.
    Óptimo para hardware de borde por su mínimo consumo de ciclos de reloj.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 120.0):
        self.next_id = 1
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, detection: Detection) -> TrackedObject:
        obj = TrackedObject(self.next_id, detection)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj

    def _deregister(self, track_id: int) -> None:
        if track_id in self.objects:
            del self.objects[track_id]

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        # Si no hay detecciones en el fotograma
        if len(detections) == 0:
            for track_id in list(self.objects.keys()):
                self.objects[track_id].disappeared_frames += 1
                if self.objects[track_id].disappeared_frames > self.max_disappeared:
                    self._deregister(track_id)
            return list(self.objects.values())

        # Si no había objetos rastreados previamente, registrar todas las detecciones
        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return list(self.objects.values())

        # Matriz de distancias euclidianas entre centroides existentes y nuevos
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]
        input_centroids = [det.center for det in detections]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, (ox, oy) in enumerate(object_centroids):
            for j, (ix, iy) in enumerate(input_centroids):
                D[i, j] = np.hypot(ox - ix, oy - iy)

        # Emparejamiento greedy por menor distancia
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Si la distancia excede el umbral máximo, no emparejar
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id].update(detections[col])
            used_rows.add(row)
            used_cols.add(col)

        # Incrementar desaparecidos para objetos no emparejados
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.objects[obj_id].disappeared_frames += 1
            if self.objects[obj_id].disappeared_frames > self.max_disappeared:
                self._deregister(obj_id)

        # Registrar nuevas detecciones no emparejadas
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self._register(detections[col])

        return list(self.objects.values())
