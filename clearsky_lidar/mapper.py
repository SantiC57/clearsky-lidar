import sys
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slamtec import SlamtecMapper


class Mapper:
    def __init__(self, host="192.168.11.1", port=1445):
        self.host = host
        self.port = port
        self._driver = None

    def connect(self):
        self._driver = SlamtecMapper(host=self.host, port=self.port)
        print(f"[ClearSky] Conectado al M1M1 en {self.host}:{self.port}")

    def disconnect(self):
        if self._driver:
            self._driver.disconnect()
            print("[ClearSky] Desconectado del M1M1")

    def scan(self, valid_only=True):
        if not self._driver:
            raise RuntimeError("No hay conexión activa. Llama connect() primero.")
        raw = self._driver.get_laser_scan(valid_only=valid_only)
        points = []
        for angle_rad, distance, valid in raw:
            x = distance * math.cos(angle_rad)
            y = distance * math.sin(angle_rad)
            points.append((x, y, 0.0))
        return points

    def get_pose(self):
        if not self._driver:
            raise RuntimeError("No hay conexión activa. Llama connect() primero.")
        return self._driver.get_pose()

    def save_pcd(self, points, filepath="primer_escaneo.pcd"):
        import open3d as o3d
        import numpy as np
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        o3d.io.write_point_cloud(filepath, cloud)
        print(f"[ClearSky] Nube de puntos guardada en {filepath}")