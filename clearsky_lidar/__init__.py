"""ClearSky LiDAR — Biblioteca de captura y procesamiento LiDAR.

Módulos principales:
    - mapper: Conexión con el sensor SLAMTEC M1M1.
    - processing: Procesamiento de nubes de puntos.
    - detection: Detección y clasificación de residuos.
    - fusion: Fusión de sensores LiDAR + cámara.
"""

from .mapper import Mapper
from .processing import Processor, PointCloudProcessor
from .detection import WasteDetector
from .fusion import SensorFusion

__version__ = "0.1.0"
__all__ = [
    "Mapper",
    "Processor",
    "PointCloudProcessor",
    "WasteDetector",
    "SensorFusion",
]