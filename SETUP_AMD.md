# Guía de Configuración & Matriz de Compatibilidad Silicio AMD
## Bio-Edge AMD • Aceleración en Borde

Esta guía detalla la configuración de controladores, runtimes y proveedores de ejecución (Execution Providers) de ONNX Runtime para acelerar la inferencia neuronal en silicio AMD.

---

## 1. Matriz de Compatibilidad de Hardware

| Proveedor de Ejecución (EP) | Hardware Objetivo | Sistema Operativo | Paquete Python Requerido | Precisión Recomendada |
| :--- | :--- | :--- | :--- | :--- |
| **`DmlExecutionProvider`** | • AMD Radeon RX 6000 / 7000 / 8000<br>• Gráficos integrados Radeon 600M / 700M / 800M | Windows 10 / 11 (DirectX 12) | `onnxruntime-directml` | FP16 / FP32 |
| **`VitisAIExecutionProvider`** | • Procesadores AMD Ryzen AI (Serie 7040, 8040, 300 Strix Point con NPU XDNA 1 / 2) | Windows 11 / Ubuntu Linux | `onnxruntime-vitisai` + Ryzen AI Software | INT8 (Cuantizado) |
| **`ROCMExecutionProvider`** | • AMD Radeon RX 7900 / Radeon Pro / Instinct | Linux (Ubuntu 22.04 / 24.04) | `onnxruntime-rocm` + ROCm v6.x | FP16 / FP32 |
| **`CPUExecutionProvider`** | • Cualquier procesador x86_64 / ARM | Windows / Linux / macOS | `onnxruntime` | FP32 (Fallback Base) |

---

## 2. Instalación por Plataforma

### Opción A: Windows con GPU AMD Radeon (DirectML) — *Recomendado para desarrollo rápido*
DirectML aprovecha cualquier GPU AMD compatible con DirectX 12 sin necesidad de compilar drivers complejos:

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\activate

# 2. Instalar dependencias con soporte DirectML
pip install onnxruntime-directml opencv-python fastapi uvicorn websockets pydantic
```

### Opción B: Windows con NPU AMD Ryzen AI (Vitis AI EP)
Para estaciones y laptops equipadas con procesador AMD Ryzen AI:

1. Instalar el driver oficial de la NPU AMD XDNA desde la página oficial de AMD Drivers.
2. Instalar el paquete **Ryzen AI Software** (v1.2+):
   ```bash
   pip install voe-4.0-win_amd64.whl
   pip install onnxruntime-vitisai
   ```
3. Configurar el archivo de configuración de Vitis AI `vaip_config.json` apuntando al backend XDNA.

### Opción C: Linux con AMD ROCm
Para servidores de borde o estaciones de trabajo con Linux:

```bash
# 1. Instalar ROCm 6.x según distribución oficial de AMD
sudo apt update && sudo apt install rocm-hip-sdk

# 2. Instalar ONNX Runtime para ROCm
pip install onnxruntime-rocm -f https://download.onnxruntime.ai/rocm/
```

### Opción D: Fallback Universal a CPU (Cero Bloqueos)
Si un miembro del equipo no dispone de silicio AMD o está desarrollando la interfaz de usuario:

```bash
pip install onnxruntime opencv-python fastapi uvicorn websockets pydantic
```
El sistema detectará automáticamente la ausencia de librerías AMD y cambiará a `CPUExecutionProvider` sin lanzar errores.

---

## 3. Verificación de Silicio AMD

Para verificar qué proveedores de aceleración reconoce tu instalación actual, ejecuta:

```bash
python -c "import onnxruntime as ort; print('Proveedores disponibles:', ort.get_available_providers())"
```

**Salida esperada con DirectML (Windows):**
```text
Proveedores disponibles: ['DmlExecutionProvider', 'CPUExecutionProvider']
```

**Verificación automática con Bio-Edge AMD:**
```bash
python run_edge.py --check-hardware
```
Salida esperada:
```text
[INFO] Verificando aceleradores de silicio AMD...
[OK] Proveedor detectado: DmlExecutionProvider (AMD Radeon Graphics)
[OK] Fallback disponible: CPUExecutionProvider
[INFO] Sistema listo para inferencia en borde.
```

---

## 4. Opciones de Cuantización (FP32 -> FP16 / INT8)

* **Generación de modelo FP16:**  
  Adecuado para GPUs Radeon para doblar el throughput de cálculo reduciendo la memoria ocupada a la mitad.
* **Generación de modelo INT8:**  
  Requerido para la NPU Ryzen AI con Vitis AI. Emplea cuantización post-entrenamiento (PTQ) con calibración en 100 imágenes del dominio ecológico para mantener mAP@50 >94%.
