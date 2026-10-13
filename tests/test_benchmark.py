import pytest
from src.engine.base_engine import MockInferenceEngine
from src.engine.benchmark import BenchmarkSuite


def test_benchmark_suite_execution():
    engine = MockInferenceEngine(device="directml")
    suite = BenchmarkSuite(engine, warmup_runs=2, test_runs=5)
    results = suite.run(frame_shape=(240, 320, 3))

    assert "metrics" in results
    assert "avg_latency_ms" in results["metrics"]
    assert "p50_latency_ms" in results["metrics"]
    assert "p95_latency_ms" in results["metrics"]
    assert "p99_latency_ms" in results["metrics"]
    assert "throughput_fps" in results["metrics"]
    assert results["metrics"]["throughput_fps"] > 0


def test_benchmark_comparison_logic():
    cpu_res = {
        "provider": "CPUExecutionProvider",
        "metrics": {"avg_latency_ms": 100.0, "throughput_fps": 10.0}
    }
    amd_res = {
        "provider": "DmlExecutionProvider",
        "metrics": {"avg_latency_ms": 20.0, "throughput_fps": 50.0}
    }

    comp = BenchmarkSuite.compare_cpu_vs_amd(cpu_res, amd_res)
    assert comp["speedup_factor"] == 5.0
    assert comp["latency_reduction_pct"] == 80.0
    assert comp["fps_improvement_pct"] == 400.0
    assert comp["kpi_target_met"] is True
