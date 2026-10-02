# Guía de Instalación para Jetson Nano

## Requisitos

- Jetson Nano 4GB
- JetPack 4.6.1 (L4T 32.7.1)
- Python 3.6-3.8
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

### 3. Instalar PyTorch para Jetson

```bash
# PyTorch 1.10 para JetPack 4.6.1 (Python 3.6)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wnb535f7p.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Para Python 3.8:
# wget https://nvidia.box.com/shared/static/veo87travc28kz5lq4m8q7y w4z5h8vq.whl -O torch-1.10.0-cp38-cp38-linux_aarch64.whl
# pip3 install torch-1.10.0-cp38-cp38-linux_aarch64.whl
```

### 4. Instalar TorchVision

```bash
sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
```

### 5. Instalar ONNX Runtime (alternativa a PyTorch)

```bash
# ONNX Runtime para Jetson
sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev
pip3 install onnxruntime-gpu==1.10.0
```

### 6. Instalar ClearSky LiDAR

```bash
git clone https://github.com/SantiC57/clearsky-lidar.git
cd clearsky-lidar
pip3 install -e ".[inference,remote]"
```

### 7. Verificar instalación

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Dispositivo: {torch.cuda.get_device_name(0)}')

from clearsky_lidar.local_detection import WasteDetectorLocal
detector = WasteDetectorLocal(model_path='models/best.onnx')
print(f'✓ Modelo cargado: {detector.classes}')
"
```

## Uso

### Inferencia con cámara

```bash
python3 scripts/test_camera_local.py --model models/best.onnx --confidence 0.5
```

### Parámetros

- `--camera`: Índice de la cámara (default: 0)
- `--width`: Ancho del frame (default: 640)
- `--height`: Alto del frame (default: 480)
- `--model`: Ruta al modelo (default: models/best.pt)
- `--confidence`: Umbral de confianza (default: 0.5)
- `--device`: Dispositivo ('cpu', 'cuda', o None para auto)
- `--inference-every`: Ejecutar inferencia cada N frames (default: 1)

## Rendimiento Esperado

- **Modelo .pt en GPU**: ~15-20 FPS
- **Modelo .onnx en GPU**: ~20-30 FPS
- **Modelo .onnx en CPU**: ~2-5 FPS

## Solución de Problemas

### Error: "CUDA out of memory"

Reducir el tamaño del frame:
```bash
python3 scripts/test_camera_local.py --width 320 --height 240
```

### Error: "Model file not found"

Asegúrate de que el modelo está en `models/best.pt` o especifica la ruta:
```bash
python3 scripts/test_camera_local.py --model /ruta/al/modelo.pt
```

### Error: "Camera not found"

Verifica que la cámara está conectada:
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

## Notas

- El modelo `best.pt` fue entrenado con las mismas 6 clases que el modelo de ClearSky
- Para mejor rendimiento en Jetson, usa el modelo ONNX con GPU
- La cámara USB puede limitar los FPS (ver diagnóstico en PC: ~7 FPS)
