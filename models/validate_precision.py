import os
from ultralytics import YOLO

def validate_map_retention():
    print("Cargando modelo para validación de precisión (mAP@50)...")
    model = YOLO("yolov8n.pt")
    
    print("Ejecutando validación en dataset de prueba (streaming/coco8 de referencia)...")
    # Validamos contra el dataset estándar configurado para medir mAP@50
    metrics = model.val(data="coco8.yaml", imgsz=640, verbose=True)
    
    map50 = metrics.box.map50 * 100
    print("\n========================================")
    print("      REPORTE DE VALIDACIÓN mAP@50      ")
    print("========================================")
    print(f"mAP@50 Medido: {map50:.2f}%")
    print(f"Meta Requerida del Proyecto: > 94.0%")
    if map50 >= 94.0:
        print("Estado: ¡APROBADO (Cumple con el KPI de retención)!")
    else:
        print("Estado: Requiere ajuste de calibración.")
    print("========================================")

if __name__ == "__main__":
    validate_map_retention()