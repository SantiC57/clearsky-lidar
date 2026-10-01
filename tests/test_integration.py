"""Tests de integración básica — verificación de imports y estructura.

Estos tests NO requieren hardware real. Verifican que los módulos se
importan correctamente y que la estructura del proyecto es válida.
"""

import pytest
import os

import clearsky_lidar
from clearsky_lidar import Mapper, WasteClassifier


def test_imports_modulos_principales():
    """Verifica que los módulos principales se importan sin error."""
    assert Mapper is not None
    assert WasteClassifier is not None


def test_version_exportada():
    """Verifica que __version__ está definida y es string."""
    assert isinstance(clearsky_lidar.__version__, str)
    assert clearsky_lidar.__version__ == "0.1.0"


def test_all_exportado():
    """Verifica que __all__ contiene las clases esperadas."""
    assert "Mapper" in clearsky_lidar.__all__
    assert "WasteClassifier" in clearsky_lidar.__all__
    assert "processing" in clearsky_lidar.__all__


def test_waste_classifier_import_root():
    """Verifica que WasteClassifier se puede importar desde la raíz."""
    from clearsky_lidar import WasteClassifier as WC
    assert WC is WasteClassifier


def test_waste_classifier_import_detection():
    """Verifica que WasteClassifier se puede importar desde detection."""
    from clearsky_lidar.detection import WasteClassifier as WC
    assert WC is WasteClassifier


def test_waste_classifier_import_classification():
    """Verifica que WasteClassifier se puede importar desde classification."""
    from clearsky_lidar.classification import WasteClassifier as WC
    assert WC is WasteClassifier


def test_waste_classifier_instancia():
    """Verifica que WasteClassifier se instancia con defaults correctos."""
    # Mock the model loading since we don't have real weights
    from unittest.mock import patch, MagicMock
    with patch("clearsky_lidar.classification.inference.load_model") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        with patch("clearsky_lidar.classification.inference.Path.exists", return_value=True):
            with patch("clearsky_lidar.classification.inference.Path.is_file", return_value=True):
                wc = WasteClassifier(model_path="dummy.pt")
                assert wc.conf_threshold == 0.55
                assert wc.model == mock_model


def test_waste_classifier_instancia_custom_threshold():
    """Verifica que WasteClassifier acepta threshold custom."""
    from unittest.mock import patch, MagicMock
    with patch("clearsky_lidar.classification.inference.load_model") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        with patch("clearsky_lidar.classification.inference.Path.exists", return_value=True):
            with patch("clearsky_lidar.classification.inference.Path.is_file", return_value=True):
                wc = WasteClassifier(model_path="dummy.pt", conf_threshold=0.8)
                assert wc.conf_threshold == 0.8


def test_waste_classifier_interface():
    """Verifica que WasteClassifier tiene la interfaz esperada."""
    from unittest.mock import patch, MagicMock
    import numpy as np
    with patch("clearsky_lidar.classification.inference.load_model") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        # Mock predict result for batch of 2
        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        # Return two results for batch of 2
        mock_model.return_value = [mock_result, mock_result]
        
        with patch("clearsky_lidar.classification.inference.Path.exists", return_value=True):
            with patch("clearsky_lidar.classification.inference.Path.is_file", return_value=True):
                wc = WasteClassifier(model_path="dummy.pt")
                
                # Test predict method exists and returns tuple
                image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                result = wc.predict(image)
                assert isinstance(result, tuple)
                assert len(result) == 3
                class_name, confidence, probs_dict = result
                assert isinstance(class_name, str)
                assert isinstance(confidence, float)
                assert isinstance(probs_dict, dict)
                assert len(probs_dict) == 6
                
                # Test predict_batch method exists and returns list
                images = [image, image]
                results = wc.predict_batch(images)
                assert isinstance(results, list)
                assert len(results) == 2


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


def test_classification_package_exports():
    """Verifica que el paquete classification exporta WasteClassifier."""
    from clearsky_lidar.classification import __all__ as classification_all
    assert "WasteClassifier" in classification_all


def test_detection_module_exports():
    """Verifica que el módulo detection exporta WasteClassifier."""
    from clearsky_lidar.detection import __all__ as detection_all
    assert "WasteClassifier" in detection_all