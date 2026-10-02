# Guía de Instalación para Jetson Nano

## Requisitos

- Jetson Nano 4GB
- JetPack 4.6.1 (L4T 32.7.1)
- Python 3.6-3.8 (incluido en JetPack)
- Cámara USB

## Instalación

### 1. Actualizar sistema

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Instalar dependencias del sistema

```bash
sudo apt install -y \
    python3-pip \
    python3-dev \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    git \
    v4l-utils
```

### 3. Instalar ClearSky LiDAR con detección remota

```bash
git clone https://github.com/SantiC57/clearsky-lidar.git
cd clearsky-lidar
pip3 install -e ".[remote]"
```

Esto instalará:
- `supervision` - para visualización de detecciones
- `inference-sdk` - para comunicación con Roboflow API
- `opencv-python` - para procesamiento de imágenes

### 4. Verificar instalación

```bash
python3 -c "
from clearsky_lidar.remote_detection import WasteDetectorRemote, REMOTE_DETECTION_AVAILABLE
print(f'Remote detection available: {REMOTE_DETECTION_AVAILABLE}')
if REMOTE_DETECTION_AVAILABLE:
    detector = WasteDetectorRemote()
    print(f'✓ Detector creado: {detector.model_id}')
"
```

## Uso

### Inferencia con cámara

```bash
python3 scripts/test_camera_remote.py
```

### Parámetros

- `--camera`: Índice de la cámara (default: 0)
- `--list-cameras`: Lista cámaras disponibles
- `--fps-limit`: Límite de llamadas API por segundo (default: 4.0)

### Ejemplo con parámetros

```bash
python3 scripts/test_camera_remote.py --camera 0 --fps-limit 3.0
```

## Rendimiento Esperado

La detección remota usa la API de Roboflow, por lo que el rendimiento depende de:
- Velocidad de internet (latencia)
- Tamaño de imagen (downscaling a 640px)
- Rate limiting (4 FPS por defecto)

**Rendimiento típico:**
- Latencia por inferencia: ~190ms (con downscaling)
- FPS efectivo: 3-4 FPS (limitado por rate limit)
- Uso de CPU: bajo (inferencia en la nube)
- Uso de RAM: ~200MB

## Ventajas de Detección Remota

1. **No requiere GPU local** - La inferencia se hace en la nube
2. **Modelo actualizado** - Usa el modelo `yolov8-trash-detections/6` de Roboflow
3. **Optimizado** - Downscaling automático y rate limiting
4. **Threaded** - Inferencia en background, no bloquea la UI
5. **Manejo de errores** - Backoff automático y detección de errores fatales

## Solución de Problemas

### Error: "Remote detection dependencies not available"

Asegúrate de instalar con el extra `remote`:
```bash
pip3 install -e ".[remote]"
```

### Error: "API key is required"

Configura la API key de Roboflow:
```bash
export ROBOFLOW_API_KEY="tu_api_key"
```

O pasa la API key al detector:
```python
detector = WasteDetectorRemote(api_key="tu_api_key")
```

### Error: "Camera not found"

Verifica que la cámara está conectada:
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
python3 scripts/test_camera_remote.py --list-cameras
```

### Baja velocidad de inferencia

- Verifica tu conexión a internet
- Reduce `--fps-limit` si hay errores
- Asegúrate de que la cámara no esté siendo usada por otra aplicación

## Notas

- La API key está hardcodeada en el código (`REMOVED_API_KEY`)
- El modelo usado es `yolov8-trash-detections/6`
- Las detecciones se hacen en la nube, no localmente
- El rate limit es de 4 FPS por defecto para evitar saturar la API
- Funciona en Python 3.6-3.13 (compatible con Jetson Nano)
