# ClearSky LiDAR

Biblioteca Python para captura, procesamiento y visualización de datos LiDAR del sensor **SLAMTEC M1M1** en el proyecto ClearSky.

## Qué hace este repo

| Módulo | Función |
|--------|---------|
| `slamtec.py` | Cliente TCP crudo para el protocolo JSON del M1M1 (puerto 1445) |
| `clearsky_lidar/mapper.py` | Wrapper alto nivel: escaneo, pose, guardado PCD (Open3D) |
| `clearsky_lidar/processing.py` | Lee `dump/laser-full.csv`, filtra, convierte polar→cartesiano, genera gráfica 2-panel (polar + top-down con ConvexHull) |
| `clearsky_lidar/detection.py` | (pendiente) Detección y clasificación de residuos |
| `clearsky_lidar/fusion.py` | (pendiente) Fusión LiDAR + cámara |

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

# Desde el paquete - guarda PCD (requiere extra lidar: open3d)
python -c "from clearsky_lidar import Mapper; m=Mapper(); m.connect(); pts=m.scan(); m.save_pcd(pts); m.disconnect()"
```

### Procesamiento offline (requiere dump/laser-full.csv existente)

```bash
# Módulo - genera Graphics/lidar_resultado.png
python -m clearsky_lidar.processing

# Directo
python clearsky_lidar/processing.py
```

### Tests

```bash
python -m pytest tests/ -v
```

### Extras opcionales

```bash
uv pip install -e .[processing]      # scipy para ConvexHull
uv pip install -e .[visualization]   # Pillow para show_map
uv pip install -e .[lidar]           # open3d para PCD (Python 3.10-3.12)
uv pip install -e .[all]             # todo desarrollo (sin open3d en Py3.14+)
```