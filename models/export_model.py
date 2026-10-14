from ultralytics import YOLO

def export_yolo_to_onnx():
    print("Cargando el modelo base YOLOv8-nano...")
   
    model = YOLO("yolov8n.pt")
    
    print("Exportando el modelo a formato ONNX...")
 
    success = model.export(format="onnx", imgsz=640, dynamic=False)
    
    print(f"¡Exportación exitosa! Archivo guardado en: {success}")

if __name__ == "__main__":
    export_yolo_to_onnx()