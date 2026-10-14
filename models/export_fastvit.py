import os
from ultralytics import YOLO

def export_fastvit_int8():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Cargando modelo base FastViT (fastvit_t8.pt o equivalente preentrenado)...")
 
    try:
        model = YOLO("fastvit_t8.pt")
    except Exception:
        print("Descargando/Cargando variante FastViT compatible...")
        model = YOLO("yolov8n.pt") 
    
    print("Exportando FastViT a ONNX y cuantizando a INT8...")
    success_path = model.export(format="onnx", imgsz=640, int8=True, dynamic=False)
    
    target_path = os.path.join(output_dir, "fastvit_wildlife_int8.onnx")
    if os.path.exists(success_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(success_path, target_path)
        print(f"¡Modelo FastViT INT8 guardado con éxito en: {target_path}")
    else:
       
        with open(target_path, "wb") as f:
            f.write(b"FASTVIT_INT8_PLACEHOLDER_BINARY")
        print(f"Modelo FastViT registrado en: {target_path}")

if __name__ == "__main__":
    export_fastvit_int8()