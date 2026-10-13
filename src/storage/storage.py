"""
Persistencia Local en SQLite para Bio-Edge AMD.
Implementado por Angel & Cisneros (Frente B) con modo WAL para máxima resiliencia en el borde.
"""

import sqlite3
import os
import time
import logging
from typing import Dict, Any, List, Optional
from ..pipeline.counting import CrossingEvent
from ..engine.base_engine import Detection

logger = logging.getLogger("BioEdge.Storage")


class SQLiteStorage:
    """Manejador transaccional local para eventos de bio-monitoreo en SQLite."""

    def __init__(self, db_path: str = "data/bio_edge.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Activar Write-Ahead Logging para tolerancia a cortes de energía
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                device_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                video_source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                track_id INTEGER,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS crossings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fps REAL NOT NULL,
                latency_ms REAL NOT NULL,
                cpu_percent REAL,
                gpu_memory_mb REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id);
            CREATE INDEX IF NOT EXISTS idx_cross_class ON crossings(class_name);
            """)
        logger.info("Base de datos SQLite inicializada en: %s", self.db_path)

    def start_session(self, session_id: str, provider: str, model_name: str, source: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, device_provider, model_name, video_source)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, provider, model_name, str(source))
            )

    def record_detections(self, session_id: str, detections: List[Detection], track_ids: Optional[List[int]] = None) -> None:
        if not detections:
            return
        records = []
        for i, det in enumerate(detections):
            tid = track_ids[i] if track_ids and i < len(track_ids) else None
            x1, y1, x2, y2 = det.bbox
            records.append((session_id, tid, det.class_name, det.confidence, x1, y1, x2, y2))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO detections (session_id, track_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def record_crossings(self, session_id: str, events: List[CrossingEvent]) -> None:
        if not events:
            return
        records = [
            (session_id, evt.track_id, evt.class_name, evt.direction, evt.confidence)
            for evt in events
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO crossings (session_id, track_id, class_name, direction, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                records
            )

    def log_telemetry(self, session_id: str, fps: float, latency_ms: float, cpu_percent: float, gpu_memory_mb: float = 0.0) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_logs (session_id, fps, latency_ms, cpu_percent, gpu_memory_mb)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, fps, latency_ms, cpu_percent, gpu_memory_mb)
            )

    def get_summary_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            query = """
            SELECT class_name, direction, COUNT(*) as total
            FROM crossings
            """
            params = []
            if session_id:
                query += " WHERE session_id = ?"
                params.append(session_id)
            query += " GROUP BY class_name, direction"

            cursor = conn.execute(query, params)
            counts_by_specie: Dict[str, Dict[str, int]] = {}
            for row in cursor.fetchall():
                specie = row["class_name"]
                direction = row["direction"]
                count = row["total"]
                if specie not in counts_by_specie:
                    counts_by_specie[specie] = {}
                counts_by_specie[specie][direction] = count

            # Detecciones totales
            det_query = "SELECT COUNT(*) as total FROM detections"
            if session_id:
                det_query += " WHERE session_id = ?"
            cur2 = conn.execute(det_query, params)
            total_detections = cur2.fetchone()["total"]

            # Promedio de rendimiento reciente
            tel_query = "SELECT AVG(fps) as avg_fps, AVG(latency_ms) as avg_lat FROM (SELECT fps, latency_ms FROM telemetry_logs ORDER BY id DESC LIMIT 50)"
            cur3 = conn.execute(tel_query)
            tel_row = cur3.fetchone()

            return {
                "crossings_by_species": counts_by_specie,
                "total_detections": total_detections,
                "avg_fps": round(tel_row["avg_fps"] or 0.0, 1),
                "avg_latency_ms": round(tel_row["avg_lat"] or 0.0, 1)
            }
