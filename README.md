# ClearSky LiDAR

Biblioteca de captura y procesamiento LiDAR para el proyecto **ClearSky** — detección de residuos sólidos en cuerpos de agua mediante dron con LiDAR.

## Descripción

ClearSky LiDAR proporciona una API Python para conectar con el sensor SLAMTEC M1M1, capturar nubes de puntos 3D, y procesarlas para detectar y clasificar residuos. Está diseñado para ejecutarse en una NVIDIA Jetson Nano como parte del sistema de navegación autónoma del dron.

## Estado Actual

**v0.1.0** — Subtarea 1 completa (infraestructura base).

| Módulo | Estado | Descripción |
|--------|--------|-------------|
| `mapper.py` | ✅ Funcional | Conexión y captura con M1M1 |
| `processing.py` | 🔲 Placeholder | Procesamiento de nubes de puntos |
| `detection.py` | 🔲 Placeholder | Detección y clasificación de residuos |
| `fusion.py` | 🔲 Placeholder | Fusión LiDAR + cámara |

## Estructura del Proyecto

```
clearsky-lidar/
├── clearsky_lidar/          # Paquete principal
│   ├── __init__.py          # API pública
│   ├── mapper.py            # Conexión con SLAMTEC M1M1
│   ├── processing.py        # Procesamiento de nubes de puntos
│   ├── detection.py         # Detección de residuos
│   └── fusion.py            # Fusión de sensores
├── config/
│   └── config_lidar.yaml    # Parámetros del sensor
├── scans/                   # Archivos PCD generados (no en repo)
├── tests/
│   ├── test_mapper.py       # Tests unitarios del mapper
│   └── test_integration.py  # Tests de integración
├── slamtec.py               # Driver base (no modificar)
├── pyproject.toml            # Configuración del paquete
└── CHANGELOG.md
```

## Instalación

### Requisitos
- Python >= 3.6
- numpy
- open3d >= 0.14 (opcional, para operaciones avanzadas de nubes de puntos)

### Instalación editable
```bash
pip install -e .
```

### Con dependencias opcionales
```bash
pip install -e ".[lidar]"   # Incluye open3d
pip install -e ".[dev]"     # Incluye pytest, pyyaml
```

## Uso Básico

### Captura de nube de puntos
```python
from clearsky_lidar import Mapper

mapper = Mapper()
mapper.connect()
points = mapper.scan()          # Lista de (x, y, z)
mapper.save_pcd(points)         # Guarda primer_escaneo.pcd
mapper.disconnect()
```

### Lectura de configuración
```python
import yaml

with open("config/config_lidar.yaml") as f:
    config = yaml.safe_load(f)

lidar = config["lidar"]
print(f"Sensor en {lidar['host']}:{lidar['port']}")
print(f"Rango: {lidar['rango_min_m']}m - {lidar['rango_max_m']}m")
```

### Módulos placeholder (futuro)
```python
from clearsky_lidar import PointCloudProcessor, WasteDetector, SensorFusion

processor = PointCloudProcessor()
# processor.filter_noise(points)       # NotImplementedYet
# processor.downsample(points)         # NotImplementedYet
```

## Hardware Requerido

- **Sensor**: SLAMTEC M1M1 (LiDAR 2D)
- **Computadora**: NVIDIA Jetson Nano 4GB (ARM64, Ubuntu 18.04)
- **Red**: Conexión LAN entre Jetson y M1M1 (192.168.11.1:1445)

## Limitaciones

- La captura real de datos requiere el hardware M1M1 conectado.
- Los módulos `processing.py`, `detection.py` y `fusion.py` son placeholders.
- La detección de residuos aún no está implementada.
- No incluye soporte ROS2 (solo ROS1 Melodic).

## Ejecución de Tests

```bash
# Tests unitarios (no requieren hardware)
pytest tests/ -v --ignore=tests/test_integration.py

# Todos los tests
pytest tests/ -v
```

## Créditos

Basado en [python-slamtec-mapper](https://github.com/SantiC57/clearsky-lidar) — driver Python para SLAMTEC M1M1 desarrollado mediante ingeniería inversa del protocolo JSON por puerto 1445.

## Licencia

MIT
