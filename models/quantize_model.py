import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_yolov8_int8():
    input_model_path = "yolov8n.onnx"  # Modelo exportado previamente en la raíz
    output_model_path = "models/yolov8n_wildlife_int8.onnx"
    

    os.makedirs("models", exist_ok=True)
    
    print("Iniciando la cuantización del modelo de FP32 a INT8...")
    
    # Aplicar cuantización dinámica optimizada para pesos de 8 bits
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"¡Cuantización completada con éxito!")
    print(f"Modelo INT8 guardado en: {output_model_path}")

if __name__ == "__main__":
    quantize_yolov8_int8()