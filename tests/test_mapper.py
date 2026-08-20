"""Tests unitarios para el módulo mapper y configuración."""

from clearsky_lidar import Mapper
import clearsky_lidar


def test_mapper_instancia():
    m = Mapper(host="192.168.11.1", port=1445)
    assert m.host == "192.168.11.1"
    assert m.port == 1445


def test_mapper_version():
    assert clearsky_lidar.__version__ == "0.1.0"


def test_mapper_sin_conexion():
    """Verifica que scan() lanza RuntimeError sin conexión activa."""
    m = Mapper()
    try:
        m.scan()
        assert False, "Debería haber lanzado RuntimeError"
    except RuntimeError as e:
        assert "No hay conexión activa" in str(e)


def test_mapper_pose_sin_conexion():
    """Verifica que get_pose() lanza RuntimeError sin conexión activa."""
    m = Mapper()
    try:
        m.get_pose()
        assert False, "Debería haber lanzado RuntimeError"
    except RuntimeError as e:
        assert "No hay conexión activa" in str(e)


def test_config_lidar_carga():
    """Verifica que config_lidar.yaml se puede leer y tiene las claves esperadas."""
    import os

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config_lidar.yaml"
    )
    if not os.path.exists(config_path):
        # En CI el path puede variar
        config_path = "config/config_lidar.yaml"

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "lidar" in config
        lidar = config["lidar"]
        assert "host" in lidar
        assert "port" in lidar
        assert "valid_only" in lidar
        assert "rango_min_m" in lidar
        assert "rango_max_m" in lidar
        assert "frecuencia_hz" in lidar
        assert "altura_montaje_m" in lidar
    except ImportError:
        # Si yaml no está instalado, parseamos manualmente
        with open(config_path) as f:
            content = f.read()
        assert "host:" in content
        assert "rango_min_m:" in content
        assert "rango_max_m:" in content
        assert "frecuencia_hz:" in content
        assert "altura_montaje_m:" in content