"""Tests de integración básica — verificación de imports y estructura.

Estos tests NO requieren hardware real. Verifican que los módulos se
importan correctamente y que la estructura del proyecto es válida.
"""

import pytest
import os

import clearsky_lidar
from clearsky_lidar import Mapper, WasteDetectorRemote


def test_imports_modulos_principales():
    """Verifica que los módulos principales se importan sin error."""
    assert Mapper is not None
    assert WasteDetectorRemote is not None


def test_version_exportada():
    """Verifica que __version__ está definida y es string."""
    assert isinstance(clearsky_lidar.__version__, str)
    assert clearsky_lidar.__version__ == "0.1.0"


def test_all_exportado():
    """Verifica que __all__ contiene las clases esperadas."""
    assert "Mapper" in clearsky_lidar.__all__
    assert "WasteDetectorRemote" in clearsky_lidar.__all__
    assert "processing" in clearsky_lidar.__all__


def test_waste_detector_remote_import_root():
    """Verifica que WasteDetectorRemote se puede importar desde la raíz."""
    from clearsky_lidar import WasteDetectorRemote as WDR
    assert WDR is WasteDetectorRemote


def test_waste_detector_remote_import_detection():
    """Verifica que WasteDetectorRemote se puede importar desde detection."""
    from clearsky_lidar.detection import WasteDetectorRemote as WDR
    assert WDR is WasteDetectorRemote


def test_detection_module_exports():
    """Verifica que el módulo detection exporta WasteDetectorRemote."""
    from clearsky_lidar.detection import __all__ as detection_all
    assert "WasteDetectorRemote" in detection_all


def test_scans_directorio_existe():
    """Verifica que el directorio scans/ existe (opcional, solo para desarrollo)."""
    import pytest
    scans_path = os.path.join(os.path.dirname(__file__), "..", "scans")
    if not os.path.isdir(scans_path):
        pytest.skip(f"Directorio scans/ no encontrado en {scans_path} (opcional)")


def test_primer_escaneo_no_en_repo():
    """Verifica que primer_escaneo.pcd NO está en el repo (se genera con hardware)."""
    pcd_path = os.path.join(os.path.dirname(__file__), "..", "primer_escaneo.pcd")
    assert not os.path.exists(pcd_path), "primer_escaneo.pcd no debería existir en el repo"
