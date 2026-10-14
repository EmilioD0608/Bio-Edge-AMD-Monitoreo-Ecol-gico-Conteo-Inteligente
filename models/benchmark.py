import os
import time
import numpy as np
import onnxruntime as ort

def run_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "yolov8n_wildlife_int8.onnx")
    
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo INT8 en {model_path}")
        return

    print(f"Cargando modelo optimizado para benchmark: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Obtener el nombre de la entrada y la forma esperada
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Normalizar dimensiones dinámicas si las hay (ej: batch 1, 3 canales, 640x640)
    batch_size = 1
    input_shape_fixed = [batch_size, 3, 640, 640]
    
    # Crear un tensor de prueba aleatorio simulando un frame de video
    dummy_input = np.random.randn(*input_shape_fixed).astype(np.float32)
    
    print("Ejecutando pruebas de rendimiento (Warm-up)...")
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    print("Calculando latencias (P50, P95, P99) y throughput...")
    latencies = []
    num_iterations = 100
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)
        
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / np.mean(latencies)
    
    print("\n========================================")
    print("      REPORTE DE BENCHMARKING INT8      ")
    print("========================================")
    print(f"Latencia P50 (Mediana): {p50:.2f} ms")
    print(f"Latencia P95:           {p95:.2f} ms")
    print(f"Latencia P99:           {p99:.2f} ms")
    print(f"Rendimiento Promedio:   {fps:.2f} FPS")
    print("========================================")

if __name__ == "__main__":
    run_benchmark()