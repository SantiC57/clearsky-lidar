from clearsky_lidar import Mapper
import clearsky_lidar

def test_mapper_instancia():
    m = Mapper(host="192.168.11.1", port=1445)
    assert m.host == "192.168.11.1"
    assert m.port == 1445

def test_mapper_version():
    assert clearsky_lidar.__version__ == "0.1.0"