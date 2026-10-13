"""
Simulador Local Mock Stream para Bio-Edge AMD.
Implementa la Política de Cero Bloqueos: permite al equipo de Frontend (Elida & Mario)
desarrollar y probar toda la interfaz reactiva sin depender de cámaras ni del backend completo.
"""

import asyncio
import base64
import json
import time
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="Bio-Edge AMD Mock Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MockWildlifeGenerator:
    def __init__(self):
        self.step = 0
        self.species = [
            {"name": "jaguar", "color": (0, 180, 255), "threat": False},
            {"name": "tapir", "color": (200, 150, 50), "threat": False},
            {"name": "venado", "color": (50, 220, 50), "threat": False},
            {"name": "humano_intruso", "color": (0, 0, 240), "threat": True}
        ]
        self.crossings = {"jaguar": {"ENTRANDO": 4, "SALIENDO": 2}, "venado": {"ENTRANDO": 14, "SALIENDO": 9}}

    def generate_frame_and_meta(self):
        self.step += 1
        w, h = 960, 540
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Gradiente ambiental tipo selva
        frame[:, :, 1] = int(35 + 12 * np.sin(self.step * 0.04))
        frame[:, :, 0] = 20
        frame[:, :, 2] = 20

        # Línea virtual de conteo
        line_x = w // 2
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)
        cv2.putText(frame, "LINEA DE CONTEO VIRTUAL (MOCK)", (line_x - 140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Fauna animada
        pos_x = int((self.step * 8) % (w - 180)) + 60
        pos_y = int(h * 0.45 + np.sin(self.step * 0.08) * 45)

        spec = self.species[(self.step // 80) % len(self.species)]
        bw, bh = 130, 85
        cv2.rectangle(frame, (pos_x, pos_y), (pos_x + bw, pos_y + bh), spec["color"], 2)
        label = f"#102 {spec['name']} (96%)"
        cv2.putText(frame, label, (pos_x, pos_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, spec["color"], 1)

        # HUD informativo
        is_amd = (self.step // 120) % 2 == 0
        device_label = "AMD Radeon 780M (DirectML)" if is_amd else "CPU x86_64 Baseline"
        lat = 18.4 if is_amd else 104.2
        fps = 54.0 if is_amd else 9.6

        cv2.rectangle(frame, (10, 10), (320, 95), (10, 10, 10), -1)
        cv2.putText(frame, f"SIMULADOR BIO-EDGE • {device_label}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120) if is_amd else (0, 160, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Latencia: {lat:.1f} ms", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(frame, f"Frame: {self.step} | Modo Mock Activo", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Codificar JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_str = base64.b64encode(buf).decode("utf-8")

        telemetry = {
            "session_id": "mock_session_zero_block",
            "frame_id": self.step,
            "fps": fps,
            "latency_ms": lat,
            "hardware": {
                "device": "amd_directml" if is_amd else "cpu",
                "provider": "DmlExecutionProvider" if is_amd else "CPUExecutionProvider",
                "is_accelerated": is_amd
            },
            "summary_counts": {
                "counts": self.crossings,
                "total_in": 18,
                "total_out": 11
            }
        }

        return f"data:image/jpeg;base64,{b64_str}", telemetry


generator = MockWildlifeGenerator()


@app.get("/api/health")
def mock_health():
    return {
        "status": "MOCK_ACTIVE",
        "message": "Simulador de pruebas operativo sin bloqueos",
        "mock_feed": True
    }


@app.get("/api/benchmarks/compare")
def mock_benchmark_compare():
    return {
        "status": "SUCCESS",
        "cpu_baseline": {
            "device": "cpu",
            "provider": "CPUExecutionProvider",
            "metrics": {"avg_latency_ms": 108.5, "throughput_fps": 9.2, "p95_latency_ms": 115.0}
        },
        "amd_accelerator": {
            "device": "directml",
            "provider": "DmlExecutionProvider (AMD Radeon)",
            "metrics": {"avg_latency_ms": 19.4, "throughput_fps": 51.5, "p95_latency_ms": 22.8}
        },
        "comparison": {
            "speedup_factor": 5.6,
            "latency_reduction_pct": 82.1,
            "fps_improvement_pct": 460.0,
            "kpi_target_met": True
        }
    }


@app.websocket("/ws/stream")
async def mock_ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            img_b64, telemetry = generator.generate_frame_and_meta()
            await websocket.send_text(json.dumps({
                "type": "frame",
                "image": img_b64,
                "telemetry": telemetry
            }))
            await asyncio.sleep(0.033)  # ~30 FPS
    except WebSocketDisconnect:
        pass


frontend_path = "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    print("\n[BIO-EDGE AMD] Iniciando Simulador Mock en http://127.0.0.1:8001")
    print("Frontend puede conectarse directamente a ws://127.0.0.1:8001/ws/stream\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
