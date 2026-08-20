"""ClearSky LiDAR — Biblioteca de captura y procesamiento LiDAR.

Módulos principales:
    - mapper: Conexión con el sensor SLAMTEC M1M1.
    - processing: Procesamiento de escaneos LiDAR (funciones).
    - detection: Detección y clasificación de residuos (pendiente).
    - fusion: Fusión de sensores LiDAR + cámara (pendiente).
"""

from .mapper import Mapper
from . import processing

__version__ = "0.1.0"
__all__ = [
    "Mapper",
    "processing",
]