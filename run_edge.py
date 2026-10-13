"""
Script de Ejecución Maestro para Bio-Edge AMD.
Diseñado por Emilio (Frente A).
Orquesta arranque con banderas de configuración dinámica, inspección de hardware y benchmarks.
"""

import sys
import os
import argparse
import logging
import uvicorn

# Añadir directorio raíz al sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def check_amd_hardware():
    print("\n=======================================================")
    print("      BIO-EDGE AMD • INSPECCIÓN DE SILICIO & DRIVERS    ")
    print("=======================================================\n")
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        print(f"[INFO] Versión ONNX Runtime instalada: {ort.__version__}")
        print(f"[INFO] Proveedores de ejecución detectados: {available}\n")

        amd_found = False
        if "DmlExecutionProvider" in available:
            print("[OK] DirectML Execution Provider: ACTIVO (GPU AMD Radeon)")
            amd_found = True
        if "VitisAIExecutionProvider" in available:
            print("[OK] Vitis AI Execution Provider: ACTIVO (NPU AMD Ryzen AI XDNA)")
            amd_found = True
        if "ROCMExecutionProvider" in available:
            print("[OK] ROCm Execution Provider: ACTIVO (GPU AMD Radeon en Linux)")
            amd_found = True

        if not amd_found:
            print("[AVISO] No se detectaron librerías nativas de aceleración AMD.")
            print("        El sistema operará con CPUExecutionProvider (Fallback Cero Bloqueo).")
            print("        Consulte SETUP_AMD.md para habilitar DirectML o Vitis AI EP.")
        else:
            print("[ÉXITO] Silicio AMD verificado y listo para inferencia acelerada.")

    except ImportError:
        print("[AVISO] onnxruntime no instalado en este entorno. Bio-Edge operará en modo simulado.")


def main():
    parser = argparse.ArgumentParser(
        description="Bio-Edge AMD: Monitoreo Ecológico & Conteo Inteligente en el Borde"
    )
    parser.add_argument("--device", type=str, default="directml", choices=["directml", "vitisai", "rocm", "cpu", "auto"],
                        help="Proveedor de silicio (default: directml)")
    parser.add_argument("--source", type=str, default="0",
                        help="Fuente de video: webcam '0', URL RTSP o archivo de video")
    parser.add_argument("--model", type=str, default="models/yolov8n_wildlife_int8.onnx",
                        help="Ruta al modelo ONNX optimizado")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Umbral de confianza mínima para detección")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host para el servidor web/API (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Puerto HTTP/WebSocket (default: 8000)")
    parser.add_argument("--check-hardware", action="store_true",
                        help="Verificar compatibilidad de silicio AMD y salir")
    parser.add_argument("--benchmark", action="store_true",
                        help="Ejecutar suite de benchmarking de latencia y salir")

    args = parser.parse_args()

    if args.check_hardware:
        check_amd_hardware()
        sys.exit(0)

    if args.benchmark:
        from src.engine.benchmark import BenchmarkSuite, AMDInferenceEngine
        print(f"\n[BENCHMARK] Evaluando rendimiento en silicio: {args.device}...")
        engine = AMDInferenceEngine(model_path=args.model, device=args.device)
        suite = BenchmarkSuite(engine, warmup_runs=10, test_runs=50)
        res = suite.run()
        print("\nResultados:")
        for k, v in res["metrics"].items():
            print(f"  • {k}: {v}")
        sys.exit(0)

    # Configurar variables de entorno para FastAPI startup
    os.environ["BIOEDGE_DEVICE"] = args.device
    os.environ["BIOEDGE_SOURCE"] = args.source
    if args.model and os.path.exists(args.model):
        os.environ["BIOEDGE_MODEL"] = args.model

    print("\n=======================================================")
    print("   INICIANDO SISTEMA AUTÓNOMO BIO-EDGE AMD (BORDE)     ")
    print(f"   • Dispositivo:      {args.device}")
    print(f"   • Fuente Video:     {args.source}")
    print(f"   • Modelo:           {args.model}")
    print(f"   • Dashboard Web:    http://localhost:{args.port}")
    print("=======================================================\n")

    uvicorn.run("src.api.main_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
