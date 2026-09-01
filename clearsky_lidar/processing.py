import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class PointCloudProcessor:
    """Procesador de nubes de puntos del LiDAR M1M1 — ClearSky"""

    # Límites de las zonas (tus variables originales)
    UMBRAL_CERCA = 1.0   # <= 1m  → rojo  → zona de residuos
    UMBRAL_MEDIO = 3.0   # <= 3m  → azul  → zona intermedia
                         # >  3m  → verde → entorno fijo

    def __init__(self):
        # Variables que se llenan cuando cargas y procesas
        self.df          = None   # datos crudos
        self.df_valido   = None   # puntos válidos (<19m)
        self.df_invalido = None   # puntos sin retorno (>=19m)
        self.df_cercano  = None   # zona roja  <= 1m
        self.df_mediano  = None   # zona azul  1-3m
        self.df_lejano   = None   # zona verde > 3m
        self.df_bordes   = None   # contornos de objetos

    def cargar(self, ruta_csv):
        """Carga el CSV que genera el mapper.py"""
        self.df = pd.read_csv(ruta_csv, header=None)
        self.df.columns = ["angulo_rad", "distancia_m", "angulo_deg"]

    def procesar(self):
        """Filtra, convierte a X Y y separa en 3 zonas"""

        # Filtrar inválidos — alcance real M1M1 = 19m
        self.df_valido   = self.df[self.df["distancia_m"] < 19].copy()
        self.df_invalido = self.df[self.df["distancia_m"] >= 19]

        # Convertir (ángulo, distancia) → (x, y)
        self.df_valido["x"] = self.df_valido["distancia_m"] * np.cos(self.df_valido["angulo_rad"])
        self.df_valido["y"] = self.df_valido["distancia_m"] * np.sin(self.df_valido["angulo_rad"])

        # Separar en 3 zonas
        self.df_cercano = self.df_valido[self.df_valido["distancia_m"] <= self.UMBRAL_CERCA]
        self.df_mediano = self.df_valido[
            (self.df_valido["distancia_m"] > self.UMBRAL_CERCA) &
            (self.df_valido["distancia_m"] <= self.UMBRAL_MEDIO)
        ]
        self.df_lejano = self.df_valido[self.df_valido["distancia_m"] > self.UMBRAL_MEDIO]

        # Detectar bordes de objetos
        df_sorted = self.df_valido.sort_values("angulo_deg").copy()
        df_sorted["delta_dist"] = df_sorted["distancia_m"].diff().abs()
        self.df_bordes = df_sorted[df_sorted["delta_dist"] > 0.5]

    def resumen(self):
        """Imprime el resumen de los datos"""
        print("===== RESUMEN DE TUS DATOS =====")
        print(f"Total de puntos escaneados : {len(self.df)}")
        print(f"Puntos válidos             : {len(self.df_valido)}")
        print(f"Puntos sin retorno         : {len(self.df_invalido)}")
        print(f"Objetos cercanos  (<=1m)   : {len(self.df_cercano)}")
        print(f"Objetos medianos  (1-3m)   : {len(self.df_mediano)}")
        print(f"Objetos lejanos   (>3m)    : {len(self.df_lejano)}")
        print(f"Bordes detectados          : {len(self.df_bordes)}")
        print(f"Distancia mínima detectada : {self.df_valido['distancia_m'].min():.3f} m")
        print(f"Distancia máxima detectada : {self.df_valido['distancia_m'].max():.3f} m")
        print(f"Distancia promedio         : {self.df_valido['distancia_m'].mean():.3f} m")

    def graficar(self, guardar_como="lidar_resultado.png"):
        """Genera las 3 gráficas y las guarda"""
        fig = plt.figure(figsize=(17, 5.5))

        # Gráfica 1: Vista polar — 3 colores
        ax1 = fig.add_subplot(131, projection='polar')
        ax1.scatter(self.df_lejano["angulo_rad"],  self.df_lejano["distancia_m"],
                    c="green",     s=6,  alpha=0.55, label="Lejano  (>3m)")
        ax1.scatter(self.df_mediano["angulo_rad"], self.df_mediano["distancia_m"],
                    c="steelblue", s=9,  alpha=0.7,  label="Mediano (1-3m)")
        ax1.scatter(self.df_cercano["angulo_rad"], self.df_cercano["distancia_m"],
                    c="red",       s=12, alpha=0.85, label="Cercano (<=1m)")
        ax1.set_title("Vista polar 360°", fontsize=11, pad=14)
        ax1.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.45, 1.12))

        # Gráfica 2: Vista de planta
        ax2 = fig.add_subplot(132)
        ax2.scatter(self.df_lejano["x"],  self.df_lejano["y"],
                    c="green",     s=9,  alpha=0.55, label="Lejano  (>3m)")
        ax2.scatter(self.df_mediano["x"], self.df_mediano["y"],
                    c="steelblue", s=12, alpha=0.7,  label="Mediano (1-3m)")
        ax2.scatter(self.df_cercano["x"], self.df_cercano["y"],
                    c="red",       s=18, alpha=0.9,  label="Cercano (<=1m)")
        ax2.plot(0, 0, "k^", markersize=11, label="Sensor LiDAR", zorder=6)

        if len(self.df_cercano) >= 3:
            pts   = self.df_cercano[["x","y"]].values
            hull  = ConvexHull(pts)
            area  = hull.volume
            verts = pts[hull.vertices]
            verts = np.vstack([verts, verts[0]])
            ax2.fill(verts[:,0], verts[:,1], alpha=0.15, color="red")
            ax2.plot(verts[:,0], verts[:,1], "r-", lw=1.5, alpha=0.7)
            ax2.set_title(f"Vista de planta\nÁrea zona cercana: {area:.3f} m²", fontsize=11)

        ax2.set_xlabel("X (metros)")
        ax2.set_ylabel("Y (metros)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

        # Gráfica 3: Histograma
        ax3 = fig.add_subplot(133)
        bins = np.arange(0, 19.2, 0.15)
        ax3.hist(self.df_cercano["distancia_m"], bins=bins, color="red",
                 alpha=0.75, label="Cercano (<=1m)", edgecolor="white")
        ax3.hist(self.df_mediano["distancia_m"], bins=bins, color="steelblue",
                 alpha=0.6,  label="Mediano (1-3m)", edgecolor="white")
        ax3.hist(self.df_lejano["distancia_m"],  bins=bins, color="green",
                 alpha=0.5,  label="Lejano  (>3m)",  edgecolor="white")
        ax3.axvline(self.UMBRAL_CERCA, color="red",  lw=2.5,
                    linestyle="--", label=f"Umbral cerca: {self.UMBRAL_CERCA}m")
        ax3.axvline(self.UMBRAL_MEDIO, color="blue", lw=2.5,
                    linestyle="--", label=f"Umbral medio: {self.UMBRAL_MEDIO}m")
        ax3.set_xlabel("Distancia (metros)")
        ax3.set_ylabel("Cantidad de puntos")
        ax3.set_title("Distribución de distancias")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.suptitle("Análisis LiDAR 2D — Escaneo 360°", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Gráfica guardada como {guardar_como}")


# Marcadores de posición que el __init__.py espera importar
class Processor(PointCloudProcessor):
    pass