import os
from ultralytics import YOLO

def export_yolov8_fp16():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Cargando modelo base YOLOv8-nano...")
    model = YOLO("yolov8n.pt")
    
    print("Exportando a formato ONNX en precisión FP16 (half=True)...")
    # half=True activa la exportación a FP16
    success_path = model.export(format="onnx", imgsz=640, half=True, dynamic=False)
    
    # Mover el archivo generado a la carpeta models con la nomenclatura requerida
    target_path = os.path.join(output_dir, "yolov8n_wildlife_fp16.onnx")
    
    if os.path.exists(success_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(success_path, target_path)
        print(f"¡Modelo FP16 guardado con éxito en: {target_path}")
    else:
        # Si ultralytics lo dejó en la raíz
        root_default = os.path.join(base_dir, "yolov8n.onnx")
        if os.path.exists(root_default):
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(root_default, target_path)
            print(f"¡Modelo FP16 movido y guardado en: {target_path}")

if __name__ == "__main__":
    export_yolov8_fp16()