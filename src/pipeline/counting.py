"""
Lógica de Conteo Direccional y Cruce de Líneas / Zonas ROI para Bio-Edge AMD.
Implementado por Angel (Frente B).
"""

from typing import List, Tuple, Dict, Any, Optional
import time
from .tracker import TrackedObject


class CrossingEvent:
    """Registra el cruce formal de una entidad biológica por una línea de control."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        direction: str,
        confidence: float,
        timestamp: Optional[float] = None
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.direction = direction  # 'ENTRANDO', 'SALIENDO', 'ZONA_PROTEGIDA'
        self.confidence = float(confidence)
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp
        }


class DirectionalCounter:
    """
    Evalúa la intersección geométrica entre las trayectorias de los objetos rastreados
    y una línea virtual de referencia o un polígono de zona protegida (ROI).
    """

    def __init__(
        self,
        line_start: Tuple[int, int] = (640, 0),
        line_end: Tuple[int, int] = (640, 720),
        orientation: str = "vertical"
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.orientation = orientation  # 'vertical' o 'horizontal'
        self.total_counts: Dict[str, Dict[str, int]] = {}

    def _ccw(self, A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        """Determina si tres puntos están listados en sentido antihorario."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A: Tuple[float, float], B: Tuple[float, float],
                   C: Tuple[float, float], D: Tuple[float, float]) -> bool:
        """Verifica si el segmento AB interseca al segmento CD."""
        return (self._ccw(A, C, D) != self._ccw(B, C, D)) and (self._ccw(A, B, C) != self._ccw(A, B, D))

    def update(self, tracked_objects: List[TrackedObject]) -> List[CrossingEvent]:
        events: List[CrossingEvent] = []

        for obj in tracked_objects:
            if len(obj.history) < 2:
                continue

            prev_pt = obj.history[-2]
            curr_pt = obj.history[-1]

            # Verificar si el segmento interseca la línea virtual
            if self._intersect(prev_pt, curr_pt, self.line_start, self.line_end):
                event_key = f"cross_line_{self.line_start}_{self.line_end}"
                if event_key not in obj.counted_events:
                    # Determinar dirección según la orientación
                    if self.orientation == "vertical":
                        direction = "ENTRANDO" if curr_pt[0] > prev_pt[0] else "SALIENDO"
                    else:
                        direction = "ENTRANDO" if curr_pt[1] > prev_pt[1] else "SALIENDO"

                    evt = CrossingEvent(
                        track_id=obj.track_id,
                        class_name=obj.class_name,
                        direction=direction,
                        confidence=obj.confidence
                    )
                    events.append(evt)
                    obj.counted_events.add(event_key)

                    # Actualizar contadores acumulados
                    if obj.class_name not in self.total_counts:
                        self.total_counts[obj.class_name] = {"ENTRANDO": 0, "SALIENDO": 0}
                    self.total_counts[obj.class_name][direction] += 1

        return events

    def get_summary(self) -> Dict[str, Any]:
        """Obtiene el resumen consolidado de cruces por especie."""
        return {
            "counts": self.total_counts,
            "total_in": sum(v.get("ENTRANDO", 0) for v in self.total_counts.values()),
            "total_out": sum(v.get("SALIENDO", 0) for v in self.total_counts.values())
        }
