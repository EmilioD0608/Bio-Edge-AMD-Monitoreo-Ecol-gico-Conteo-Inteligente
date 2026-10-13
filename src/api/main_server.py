"""
Servidor Central FastAPI y WebSockets para Bio-Edge AMD.
Diseñado por Cisneros (Frente B).
Provee streaming en vivo a 30+ FPS, telemetría en tiempo real y endpoints REST.
"""

import os
import asyncio
import base64
import json
import logging
from typing import Set, Optional
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from ..pipeline.pipeline import EdgePipeline
from ..engine.benchmark import BenchmarkSuite, AMDInferenceEngine

logger = logging.getLogger("BioEdge.Server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Bio-Edge AMD API",
    description="Backend de Borde con Aceleración Silicio AMD, Streaming WebSocket y Telemetría Ecológica",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global del pipeline
pipeline_instance: Optional[EdgePipeline] = None
active_stream_connections: Set[WebSocket] = set()
active_telemetry_connections: Set[WebSocket] = set()
streaming_task: Optional[asyncio.Task] = None


class DeviceSwitchRequest(BaseModel):
    device: str  # 'directml', 'vitisai', 'rocm', 'cpu'


@app.on_event("startup")
async def startup_event():
    global pipeline_instance, streaming_task
    device = os.environ.get("BIOEDGE_DEVICE", "directml")
    source = os.environ.get("BIOEDGE_SOURCE", "0")
    model = os.environ.get("BIOEDGE_MODEL", None)

    logger.info("Iniciando Bio-Edge AMD con dispositivo: %s, fuente: %s", device, source)
    pipeline_instance = EdgePipeline(source=source, model_path=model, device=device)
    pipeline_instance.start()

    streaming_task = asyncio.create_task(background_broadcast_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global pipeline_instance, streaming_task
    if streaming_task:
        streaming_task.cancel()
    if pipeline_instance:
        pipeline_instance.stop()
    logger.info("Servidor detenido correctamente.")


async def background_broadcast_loop():
    """Bucle asíncrono de emisión continua para WebSockets a 30+ FPS."""
    global pipeline_instance
    while True:
        try:
            if not pipeline_instance or not pipeline_instance.is_running:
                await asyncio.sleep(0.1)
                continue

            success, annotated_frame, telemetry = pipeline_instance.process_step()
            if not success or annotated_frame is None:
                await asyncio.sleep(0.02)
                continue

            # Si hay clientes en el WebSocket de video, codificar a JPEG
            if active_stream_connections:
                ret, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    b64_img = base64.b64encode(buffer).decode("utf-8")
                    payload = json.dumps({
                        "type": "frame",
                        "image": f"data:image/jpeg;base64,{b64_img}",
                        "telemetry": telemetry
                    })

                    dead_sockets = set()
                    for ws in active_stream_connections:
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            dead_sockets.add(ws)
                    active_stream_connections.difference_update(dead_sockets)

            # Si hay clientes en el WebSocket de telemetría pura
            if active_telemetry_connections and telemetry.get("frame_id", 0) % 5 == 0:
                tel_payload = json.dumps({"type": "telemetry", "data": telemetry})
                dead_tel = set()
                for ws in active_telemetry_connections:
                    try:
                        await ws.send_text(tel_payload)
                    except Exception:
                        dead_tel.add(ws)
                active_telemetry_connections.difference_update(dead_tel)

            await asyncio.sleep(0.01)  # Cede el ciclo de eventos
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error en bucle de streaming: %s", e)
            await asyncio.sleep(0.1)


# ===================== ENDPOINTS REST =====================

@app.get("/api/health")
def get_health():
    """Estado de salud del sistema y silicio activo."""
    if not pipeline_instance:
        return {"status": "INITIALIZING"}
    hw = pipeline_instance.engine.get_hardware_info()
    return {
        "status": "HEALTHY",
        "session_id": pipeline_instance.session_id,
        "hardware": hw,
        "fps": pipeline_instance.fps_metric,
        "latency_ms": pipeline_instance.latency_metric
    }


@app.get("/api/stats/summary")
def get_stats_summary():
    """Estadísticas biológicas y de conteo acumulado."""
    if not pipeline_instance:
        return {}
    return pipeline_instance.storage.get_summary_stats(pipeline_instance.session_id)


@app.get("/api/benchmarks/compare")
def get_benchmark_comparison(runs: int = Query(25, ge=5, le=100)):
    """Ejecuta una comparativa en vivo entre la CPU estándar y el Acelerador AMD."""
    try:
        # 1. Medir CPU Baseline
        cpu_eng = AMDInferenceEngine(device="cpu")
        cpu_res = BenchmarkSuite(cpu_eng, warmup_runs=5, test_runs=runs).run()

        # 2. Medir Acelerador AMD
        current_dev = pipeline_instance.device if pipeline_instance else "directml"
        amd_eng = AMDInferenceEngine(device=current_dev)
        amd_res = BenchmarkSuite(amd_eng, warmup_runs=5, test_runs=runs).run()

        comparison = BenchmarkSuite.compare_cpu_vs_amd(cpu_res, amd_res)
        return {
            "status": "SUCCESS",
            "cpu_baseline": cpu_res,
            "amd_accelerator": amd_res,
            "comparison": comparison
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/config/device")
def switch_device(req: DeviceSwitchRequest):
    """Permite alternar en vivo entre acelerador AMD y CPU para demostración al jurado."""
    global pipeline_instance
    if not pipeline_instance:
        return JSONResponse(status_code=400, content={"error": "Pipeline no iniciado"})

    logger.info("Cambiando dispositivo a: %s", req.device)
    curr_source = pipeline_instance.source
    pipeline_instance.stop()
    pipeline_instance = EdgePipeline(source=curr_source, device=req.device)
    pipeline_instance.start()

    return {
        "status": "DEVICE_SWITCHED",
        "new_device": req.device,
        "hardware_info": pipeline_instance.engine.get_hardware_info()
    }


# ===================== WEBSOCKETS =====================

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_stream_connections.add(websocket)
    logger.info("Cliente conectado a /ws/stream. Clientes activos: %d", len(active_stream_connections))
    try:
        while True:
            # Mantener conexión viva
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        active_stream_connections.discard(websocket)
        logger.info("Cliente desconectado de /ws/stream")


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_telemetry_connections.add(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        active_telemetry_connections.discard(websocket)


# Montar interfaz frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Bio-Edge AMD Backend Activo. Frontend en construcción."}
