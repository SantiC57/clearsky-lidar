"""ClearSky LiDAR — Biblioteca de captura y procesamiento LiDAR.

Módulos principales:
    - mapper: Conexión con el sensor SLAMTEC M1M1.
    - processing: Procesamiento de escaneos LiDAR (funciones).
    - detection: Detección y clasificación de residuos (pendiente).
    - fusion: Fusión de sensores LiDAR + cámara (pendiente).
"""

from .mapper import Mapper

__version__ = "0.1.0"
__all__ = [
    "Mapper",
    "processing",
]

# Lazy import para evitar warning con: python -m clearsky_lidar.processing
def __getattr__(name: str):
    if name == "processing":
        import importlib
        return importlib.import_module(".processing", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")