"""
Suite de Benchmarking Automatizado para Bio-Edge AMD.
Calcula latencias P50/P95/P99, cadencia en FPS y comparativas CPU vs. Silicio AMD.
Entregable de Angela & Emilio (Frente A).
"""

import time
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Optional
import psutil

from .base_engine import BaseInferenceEngine, MockInferenceEngine
from .inference_amd import AMDInferenceEngine


class BenchmarkSuite:
    """Ejecuta pruebas estadísticas rigurosas de rendimiento sobre motores de inferencia."""

    def __init__(self, engine: BaseInferenceEngine, warmup_runs: int = 15, test_runs: int = 100):
        self.engine = engine
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def run(self, frame_shape: tuple = (1080, 1920, 3)) -> Dict[str, Any]:
        dummy_frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)

        # Fase de Calentamiento (Warmup)
        for _ in range(self.warmup_runs):
            _ = self.engine.predict(dummy_frame)

        latencies_ms: List[float] = []
        cpu_usage_samples: List[float] = []

        # Fase de Medición Estadística
        start_total = time.perf_counter()
        for _ in range(self.test_runs):
            cpu_usage_samples.append(psutil.cpu_percent(interval=None))
            _, lat = self.engine.predict(dummy_frame)
            latencies_ms.append(lat)
        end_total = time.perf_counter()

        total_elapsed_s = end_total - start_total
        avg_latency = float(np.mean(latencies_ms))
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        std_dev = float(np.std(latencies_ms))
        min_lat = float(np.min(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        throughput_fps = float(self.test_runs / total_elapsed_s)

        hardware_info = self.engine.get_hardware_info()

        results = {
            "device": self.engine.device,
            "provider": hardware_info.get("provider", "Unknown"),
            "is_accelerated": hardware_info.get("is_accelerated", False),
            "iterations": self.test_runs,
            "resolution": f"{frame_shape[1]}x{frame_shape[0]}",
            "metrics": {
                "avg_latency_ms": round(avg_latency, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "std_dev_ms": round(std_dev, 2),
                "min_latency_ms": round(min_lat, 2),
                "max_latency_ms": round(max_lat, 2),
                "throughput_fps": round(throughput_fps, 1),
                "avg_cpu_percent": round(float(np.mean(cpu_usage_samples)), 1)
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        return results

    @staticmethod
    def compare_cpu_vs_amd(cpu_results: Dict[str, Any], amd_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula el factor de aceleración (Speedup) y reducción de latencia."""
        cpu_lat = cpu_results["metrics"]["avg_latency_ms"]
        amd_lat = amd_results["metrics"]["avg_latency_ms"]
        cpu_fps = cpu_results["metrics"]["throughput_fps"]
        amd_fps = amd_results["metrics"]["throughput_fps"]

        speedup = (cpu_lat / amd_lat) if amd_lat > 0 else 1.0
        fps_gain_pct = ((amd_fps - cpu_fps) / cpu_fps * 100.0) if cpu_fps > 0 else 0.0

        return {
            "cpu_provider": cpu_results.get("provider"),
            "amd_provider": amd_results.get("provider"),
            "speedup_factor": round(speedup, 2),
            "latency_reduction_pct": round((1.0 - (amd_lat / cpu_lat)) * 100.0, 1) if cpu_lat > 0 else 0.0,
            "fps_improvement_pct": round(fps_gain_pct, 1),
            "kpi_target_met": amd_lat < 25.0 and amd_fps >= 45.0
        }


def run_benchmark_cli():
    parser = argparse.ArgumentParser(description="Bio-Edge AMD Benchmarking Tool")
    parser.add_argument("--device", type=str, default="directml", help="Dispositivo (directml, vitisai, rocm, cpu)")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo ONNX")
    parser.add_argument("--runs", type=int, default=50, help="Número de iteraciones")
    parser.add_argument("--compare", action="store_true", help="Ejecutar comparativa automática CPU vs AMD")
    parser.add_argument("--output", type=str, default=None, help="Ruta para guardar reporte JSON")
    args = parser.parse_args()

    print("\n=======================================================")
    print("   BIO-EDGE AMD • SUITE DE BENCHMARKING DE SILICIO    ")
    print("=======================================================\n")

    if args.compare:
        print("[1/2] Evaluando Línea Base en CPU...")
        cpu_engine = AMDInferenceEngine(model_path=args.model, device="cpu")
        cpu_suite = BenchmarkSuite(cpu_engine, test_runs=args.runs)
        cpu_res = cpu_suite.run()
        print(f"      CPU Latencia Media: {cpu_res['metrics']['avg_latency_ms']} ms | Throughput: {cpu_res['metrics']['throughput_fps']} FPS")

        print(f"\n[2/2] Evaluando Acelerador AMD ({args.device})...")
        amd_engine = AMDInferenceEngine(model_path=args.model, device=args.device)
        amd_suite = BenchmarkSuite(amd_engine, test_runs=args.runs)
        amd_res = amd_suite.run()
        print(f"      AMD Latencia Media: {amd_res['metrics']['avg_latency_ms']} ms | Throughput: {amd_res['metrics']['throughput_fps']} FPS")

        comp = BenchmarkSuite.compare_cpu_vs_amd(cpu_res, amd_res)
        print("\n------------------ RESUMEN COMPARATIVO ------------------")
        print(f"• Factor de Aceleración: {comp['speedup_factor']}x")
        print(f"• Reducción de Latencia: {comp['latency_reduction_pct']}%")
        print(f"• Incremento de FPS:     +{comp['fps_improvement_pct']}%")
        print(f"• Meta KPI (<25ms, >45FPS): {'[CUMPLIDA]' if comp['kpi_target_met'] else '[EN EVALUACIÓN]'}")

        final_data = {"cpu": cpu_res, "amd": amd_res, "comparison": comp}
    else:
        engine = AMDInferenceEngine(model_path=args.model, device=args.device)
        suite = BenchmarkSuite(engine, test_runs=args.runs)
        final_data = suite.run()
        print(f"Dispositivo: {final_data['device']} ({final_data['provider']})")
        print(f"Latencia Media: {final_data['metrics']['avg_latency_ms']} ms")
        print(f"Percentil P50:  {final_data['metrics']['p50_latency_ms']} ms")
        print(f"Percentil P95:  {final_data['metrics']['p95_latency_ms']} ms")
        print(f"Percentil P99:  {final_data['metrics']['p99_latency_ms']} ms")
        print(f"Cadencia:       {final_data['metrics']['throughput_fps']} FPS")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2)
        print(f"\nReporte guardado exitosamente en: {args.output}")


if __name__ == "__main__":
    run_benchmark_cli()
