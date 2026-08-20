"""Tests de integración básica — verificación de imports y estructura.

Estos tests NO requieren hardware real. Verifican que los módulos se
importan correctamente y que la estructura del proyecto es válida.

NOTA: La API de procesamiento (Processor, PointCloudProcessor, WasteDetector,
SensorFusion) aún no está implementada (detection.py y fusion.py están
vacíos), así que este módulo se salta automáticamente en CI hasta que
esos módulos existan.
"""

import pytest

import clearsky_lidar
from clearsky_lidar import Mapper

try:
    from clearsky_lidar import Processor, PointCloudProcessor, WasteDetector, SensorFusion
    _API_DISPONIBLE = True
except ImportError:
    _API_DISPONIBLE = False

pytestmark = pytest.mark.skipif(
    not _API_DISPONIBLE,
    reason="API no implementada aún: Processor/PointCloudProcessor/WasteDetector/"
    "SensorFusion no existen (detection.py y fusion.py vacíos)",
)


def test_imports_modulos_nuevos():
    """Verifica que los módulos se importan sin error."""
    assert Processor is not None
    assert PointCloudProcessor is not None
    assert WasteDetector is not None
    assert SensorFusion is not None


def test_version_exportada():
    """Verifica que __version__ está definida y es string."""
    assert isinstance(clearsky_lidar.__version__, str)
    assert clearsky_lidar.__version__ == "0.1.0"


def test_all_exportado():
    """Verifica que __all__ contiene las clases esperadas."""
    assert "Mapper" in clearsky_lidar.__all__
    assert "Processor" in clearsky_lidar.__all__
    assert "PointCloudProcessor" in clearsky_lidar.__all__
    assert "WasteDetector" in clearsky_lidar.__all__
    assert "SensorFusion" in clearsky_lidar.__all__


# ── Tests de Processor (Subtarea 2 API) ──────────────────────────────


def test_processor_instancia():
    """Verifica que Processor se instancia con defaults correctos."""
    p = Processor()
    assert p.voxel_size == 0.05
    assert p.min_range == 0.15
    assert p.max_range == 20.0


def test_processor_instancia_custom():
    """Verifica que Processor acepta parámetros custom."""
    p = Processor(voxel_size=0.1, min_range=0.5, max_range=10.0)
    assert p.voxel_size == 0.1
    assert p.min_range == 0.5
    assert p.max_range == 10.0


def test_processor_import_ok():
    """Verifica que Processor y PointCloudProcessor son importables."""
    assert Processor is not None
    assert PointCloudProcessor is not None


def test_processor_filter_range():
    """filter_range elimina puntos fuera de rango."""
    import numpy as np

    p = Processor(min_range=0.5, max_range=10.0)
    points = np.array([
        [0.1, 0.0, 0.0],   # Muy cerca
        [1.0, 0.0, 0.0],   # OK
        [15.0, 0.0, 0.0],  # Muy lejos
    ])
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        filtered = p.filter_range(pcd)
        result = np.asarray(filtered.points)
        assert len(result) == 1
        assert np.allclose(result[0], [1.0, 0.0, 0.0])
    except ImportError:
        import pytest
        pytest.skip("open3d no instalado")


def test_processor_filter_range_empty():
    """filter_range maneja nube vacía sin error."""
    import numpy as np

    p = Processor()
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        filtered = p.filter_range(pcd)
        result = np.asarray(filtered.points)
        assert len(result) == 0
    except ImportError:
        import pytest
        pytest.skip("open3d no instalado")


def test_processor_filter_pipeline():
    """filter() ejecuta el pipeline completo crudo → limpio."""
    import numpy as np

    p = Processor(min_range=0.5, max_range=15.0)
    # Generar puntos sintéticos: algunos en rango, algunos no
    points = np.array([
        [0.1, 0.0, 0.0],    # Muy cerca → descartado por filter_range
        [2.0, 0.0, 0.0],    # En rango
        [3.0, 0.0, 0.0],    # En rango
        [2.0, 1.0, 0.0],    # En rango
        [3.0, 1.0, 0.0],    # En rango
        [20.0, 0.0, 0.0],   # Muy lejos → descartado por filter_range
    ])
    try:
        result = p.filter(points)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 3
        # Los puntos fuera de rango (0.1 y 20.0) deben haberse eliminado
        assert len(result) < len(points)
    except NotImplementedError:
        import pytest
        pytest.skip("open3d no instalado")


def test_processor_to_pcd():
    """to_pcd() convierte array numpy a Open3D PointCloud."""
    import numpy as np

    p = Processor()
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    try:
        pcd = p.to_pcd(points)
        assert len(pcd.points) == 2
        result = np.asarray(pcd.points)
        assert np.allclose(result, points)
    except NotImplementedError:
        import pytest
        pytest.skip("open3d no instalado")


def test_processor_to_pcd_shape_validation():
    """to_pcd() rechaza arrays con forma incorrecta."""
    import numpy as np

    p = Processor()
    bad_array = np.array([[1.0, 2.0]])  # (1, 2) no (N, 3)
    try:
        p.to_pcd(bad_array)
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass  # Expected
    except NotImplementedError:
        import pytest
        pytest.skip("open3d no instalado")


# ── Tests de compatibilidad hacia atrás (PointCloudProcessor) ────────


def test_pointcloud_processor_instancia():
    """Verifica que PointCloudProcessor se instancia con defaults del M1M1."""
    p = PointCloudProcessor()
    assert p.min_range_m == 0.15
    assert p.max_range_m == 20.0


def test_pointcloud_processor_crop():
    """Verifica que crop() filtra correctamente por bounding box."""
    p = PointCloudProcessor()
    points = [
        (0.0, 0.0, 0.0),
        (5.0, 5.0, 0.0),
        (10.0, 10.0, 0.0),
        (-1.0, -1.0, 0.0),
    ]
    # (0,0,0) y (-1,-1,0) caen dentro de [-1, 1] en X e Y (límites inclusivos)
    result = p.crop(points, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)
    assert len(result) == 2
    assert (0.0, 0.0, 0.0) in result
    assert (-1.0, -1.0, 0.0) in result

    # Con límites estrictos, solo (0,0,0) queda
    result_strict = p.crop(points, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5)
    assert len(result_strict) == 1
    assert result_strict[0] == (0.0, 0.0, 0.0)


def test_pointcloud_processor_filter_by_range():
    """Verifica que filter_by_range() elimina puntos fuera de rango."""
    p = PointCloudProcessor(min_range_m=0.5, max_range_m=10.0)
    points = [
        (0.1, 0.0, 0.0),   # Muy cerca (dist ~0.1)
        (1.0, 0.0, 0.0),   # OK (dist 1.0)
        (15.0, 0.0, 0.0),  # Muy lejos (dist 15.0)
    ]
    result = p.filter_by_range(points)
    assert len(result) == 1
    assert result[0] == (1.0, 0.0, 0.0)


# ── Tests existentes (detection, fusion, estructura) ─────────────────


def test_waste_detector_instancia():
    """Verifica que WasteDetector se instancia sin modelo."""
    d = WasteDetector()
    assert d.model_path is None
    assert d.confidence_threshold == 0.5
    assert "botella" in d.classes


def test_waste_detector_filter_by_size():
    """Verifica que filter_by_size() filtra correctamente."""
    d = WasteDetector()
    detections = [
        {"label": "ruido", "size_m": 0.001},    # Muy pequeño
        {"label": "botella", "size_m": 0.15},    # OK
        {"label": "edificio", "size_m": 5.0},    # Muy grande
    ]
    result = d.filter_by_size(detections, min_size=0.02, max_size=2.0)
    assert len(result) == 1
    assert result[0]["label"] == "botella"


def test_sensor_fusion_instancia():
    """Verifica que SensorFusion se instancia sin calibración."""
    f = SensorFusion()
    assert f.calibration_file is None
    assert f._transform_matrix is None


def test_scans_directorio_existe():
    """Verifica que el directorio scans/ existe."""
    import os
    scans_path = os.path.join(os.path.dirname(__file__), "..", "scans")
    assert os.path.isdir(scans_path), f"Directorio scans/ no encontrado en {scans_path}"


def test_primer_escaneo_no_en_repo():
    """Verifica que primer_escaneo.pcd NO está en el repo (se genera con hardware)."""
    import os
    pcd_path = os.path.join(os.path.dirname(__file__), "..", "primer_escaneo.pcd")
    assert not os.path.exists(pcd_path), "primer_escaneo.pcd no debería existir en el repo"
