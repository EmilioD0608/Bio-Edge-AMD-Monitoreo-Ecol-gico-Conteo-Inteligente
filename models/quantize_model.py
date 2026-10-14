import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_yolov8_int8():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Posibles rutas donde podría estar el archivo onnx
    possible_paths = [
        os.path.join(base_dir, "yolov8n.onnx"),
        os.path.join(base_dir, "models", "yolov8n.onnx"),
        "yolov8n.onnx",
        "models/yolov8n.onnx"
    ]
    
    input_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_model_path = path
            break
            
    output_model_path = os.path.join(base_dir, "models", "yolov8n_wildlife_int8.onnx")
    
    if not input_model_path:
        print(f"Error crítico: No se encuentra el archivo 'yolov8n.onnx' en ninguna ruta conocida.")
        return

    print(f"¡Modelo base encontrado en: {input_model_path}")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    print("Iniciando la cuantización del modelo de FP32 a INT8...")
    
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"¡Cuantización completada con éxito!")
    print(f"Modelo INT8 guardado en: {output_model_path}")

if __name__ == "__main__":
    quantize_yolov8_int8()