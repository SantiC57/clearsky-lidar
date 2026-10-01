# ClearSky LiDAR

Biblioteca Python para captura, procesamiento y visualización de datos LiDAR del sensor **SLAMTEC M1M1** en el proyecto ClearSky.

## Qué hace este repo

| Módulo | Función |
|--------|---------|
| `slamtec.py` | Cliente TCP crudo para el protocolo JSON del M1M1 (puerto 1445) |
| `clearsky_lidar/mapper.py` | Wrapper alto nivel: escaneo, pose, guardado PCD (Open3D) |
| `clearsky_lidar/processing.py` | Lee `dump/laser-full.csv`, filtra, convierte polar→cartesiano, genera gráfica 2-panel (polar + top-down con ConvexHull) |
| `clearsky_lidar/detection.py` | Detección y clasificación de residuos (`WasteClassifier`, `WasteDetectorRemote`) |
| `clearsky_lidar/remote_api.py` | Inferencia remota con API de Roboflow (`WasteDetectorRemote`) |
| `clearsky_lidar/fusion.py` | (pendiente) Fusión LiDAR + cámara |
| `clearsky_lidar/classification` | Entrenamiento, exportación e inferencia YOLOv8-cls |

## Instalación y uso por sistema operativo

> **Requisitos previos:** Python 3.10+ (3.12 recomendado para `open3d`)

| OS / Shell | Comando único (instala + activa) |
|------------|----------------------------------|
| **Linux / macOS** · **fish** | `source ./setup.fish` |
| **Linux / macOS** · **bash / zsh** | `source ./setup.sh` |
| **Windows** · **PowerShell** | `. .\setup.ps1` |

> **Nota:** Si solo ejecutas `./setup.fish` o `./setup.sh` **sin `source`**, te muestra el comando de activación pero **no lo activa**. Con `source` / `.` te deja dentro del venv listo para trabajar.

## Comandos de uso

### Captura real (requiere M1M1 conectado por Ethernet 192.168.11.1:1445)

```bash
# Script standalone - genera dump/laser-full.csv
python slamtec.py
```

### Procesamiento offline (requiere dump/laser-full.csv existente)

```bash
# Módulo - genera Graphics/lidar_resultado.png
python -m clearsky_lidar.processing
```

### Clasificación de residuos (YOLOv8-cls)

#### Entrenamiento

```bash
# Entrenar con dataset Roboflow (6 clases: cardboard, glass, metal, paper, plastic, trash)
python -m clearsky_lidar.classification.train --data-root /path/to/dataset \
    --epochs 100 --batch 16 --lr0 0.002 --patience 20 --imgsz 640 --freeze-epochs 10

# Dataset esperado en /path/to/dataset con estructura:
# train/_classes.csv, valid/_classes.csv, test/_classes.csv
```

#### Exportar a TensorRT (para Jetson Nano)

```bash
# Exportar best.pt a best.engine (TensorRT) con verificación
python -m clearsky_lidar.classification.export_engine \
    --model-path models/best.pt \
    --output-dir models \
    --imgsz 640 \
    --half \
    --verify \
    --data-root /path/to/dataset
```

> **Nota:** En Jetson Nano, si TensorRT no está disponible, genera ONNX y usa `trtexec`:
> ```bash
> trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16 --workspace=2048 --explicitBatch
> ```

#### Inferencia (Python API)

```python
from clearsky_lidar import WasteClassifier
import numpy as np

# Carga automática: explicit path → CLEARSKY_MODEL_DIR → ~/ClearSky/weights/
classifier = WasteClassifier(conf_threshold=0.55)

# Imagen como numpy array (H, W, C) - YOLO maneja resize interno
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Predicción simple
class_name, confidence, probs = classifier.predict(image)
print(f"Class: {class_name}, Confidence: {confidence:.3f}")

# Predicción por lotes
images = [image, image, image]
results = classifier.predict_batch(images)
for class_name, confidence, probs in results:
    print(f"Class: {class_name}, Confidence: {confidence:.3f}")
```

#### Variables de entorno

```bash
# Directorio personalizado para modelos (revisado antes de ~/ClearSky/weights/)
export CLEARSKY_MODEL_DIR=/custom/models/path
```

Estructura esperada en el directorio de modelos:
```
models/
├── best.pt          # PyTorch (PC)
├── best.engine      # TensorRT (Jetson)
└── data.yaml        # Nombres de clases
```

### Inferencia remota con API (Roboflow)

Para probar rápidamente sin entrenar localmente, usa la API de Roboflow:

#### Instalación

```bash
pip install "clearsky-lidar[remote]"
# o directamente:
pip install inference-sdk opencv-python
```

#### Configuración

```bash
# Establecer API key de Roboflow
export ROBOFLOW_API_KEY=your_api_key_here
```

#### Uso con cámara

```bash
# Probar inferencia remota con cámara
python scripts/test_camera_api.py

# Con opciones personalizadas
python scripts/test_camera_api.py --camera 0 --width 1280 --height 720 \
    --classes paper plastic glass metal cardboard --save
```

#### Uso desde Python

```python
from clearsky_lidar import WasteDetectorRemote
import cv2

# Inicializar detector remoto
detector = WasteDetectorRemote(
    api_key="your_api_key",  # o usa ROBOFLOW_API_KEY env var
    workspace_name="yolov8-ofcbj",
    workflow_id="general-segmentation-api-9",
    classes=["paper", "plastic", "glass", "metal", "cardboard"]
)

# Inferencia desde imagen
image = cv2.imread("test.jpg")
result = detector.predict(image)
print(result)

# Inferencia desde archivo
result = detector.predict_from_file("test.jpg")
print(result)
```

#### Notas

- La API remota tiene límites de tasa (rate limits). El script `test_camera_api.py` hace inferencia cada 1 segundo por defecto.
- Requiere conexión a internet.
- Para producción, considera entrenar localmente y desplegar en Jetson.

### Tests

```bash
python -m pytest tests/ -v
```

## Despliegue en ClearSky (Jetson Nano)

```bash
# 1. En PC: entrenar y exportar
python -m clearsky_lidar.classification.train --data-root /dataset ...
python -m clearsky_lidar.classification.export_engine --model-path models/best.pt --output-dir models

# 2. Copiar modelos a Jetson
scp models/best.pt models/best.engine models/data.yaml jetson:~/ClearSky/weights/

# 3. En Jetson: inferencia automática (usa .engine si existe, sino .pt)
# WasteClassifier resuelve automáticamente:
#   1. model_path explícito
#   2. $CLEARSKY_MODEL_DIR
#   3. ~/ClearSky/weights/
```

## Clases de residuos (6)

| Índice | Clase |
|--------|-------|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

## Dependencias opcionales

```bash
# Solo LiDAR
pip install "clearsky-lidar[lidar]"

# Solo clasificación (entrenamiento)
pip install "clearsky-lidar[training]"

# Solo inferencia local
pip install "clearsky-lidar[inference]"

# Inferencia remota con API (Roboflow)
pip install "clearsky-lidar[remote]"

# Todo
pip install "clearsky-lidar[all]"
```