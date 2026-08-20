"""
LiDAR Data Processor for ClearSky.

Reads raw LiDAR scan data from a CSV, converts polar coordinates to
cartesian (x, y), filters out-of-range error values, classifies points
by distance, computes the ConvexHull of nearby points, and generates
a two-panel figure (polar + top-down view) saved to Graphics/.

Execution (from repo root):
    python clearsky_lidar/proccesing.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib


def _detect_interactive_backend() -> bool:
    """
    Try common interactive GUI backends and fall back to 'Agg' if none works.
    Returns ``True`` when an interactive backend could be loaded so
    ``plt.show()`` will actually display a window.
    """
    for backend in ("TkAgg", "Qt5Agg", "Qt4Agg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as _plt

            _fig = _plt.figure()
            _plt.close(_fig)
            return True
        except Exception:
            continue
    # Final safe fallback for headless environments
    matplotlib.use("Agg", force=True)
    return False


INTERACTIVE = _detect_interactive_backend()
import matplotlib.pyplot as plt
from pathlib import Path

# Optional dependency – gracefully degrade if missing
try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Paths (relative to repo root — script must be run from clearsky-lidar/)
# ---------------------------------------------------------------------------
DATA_PATH = Path("dump") / "laser-full.csv"
GRAPHICS_DIR = Path("Graphics")

# ---------------------------------------------------------------------------
# CSV columns
# The source file has no header row.  Column order in the raw CSV is:
#   angle_rad, distance, angle_deg
# Distances in this dataset are already in **metres** (confirmed by the
# magnitude of valid values ~0.66 m).
# ---------------------------------------------------------------------------
COLUMN_NAMES = ["angle_rad", "distance", "angle_deg"]

# LiDAR sensors often emit a sentinel value (e.g. 100 000) when a
# measurement fails or is out of range.  Anything above this threshold
# is discarded so it does not distort the plots.
MAX_VALID_DISTANCE: float = 50_000.0

# Threshold used to colour-code “nearby” vs “far” points (metres).
UMBRAL: float = 1.0


def validate_dataframe(df: pd.DataFrame) -> None:
    """Ensure the DataFrame is not empty and contains every expected column."""
    if df.empty:
        raise ValueError("El archivo CSV está vacío.")

    missing = [col for col in COLUMN_NAMES if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def read_lidar_data(path: Path) -> pd.DataFrame:
    """
    Load LiDAR data from *path*.

    Validates that the file exists, is non-empty, and contains the expected
    columns (added manually because the CSV has no header).
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")

    if path.stat().st_size == 0:
        raise ValueError(f"El archivo está vacío: {path.resolve()}")

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    validate_dataframe(df)
    return df


def filter_valid_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove saturated / out-of-range measurements.

    Returns a new DataFrame containing only rows whose distance is
    below *MAX_VALID_DISTANCE*.
    """
    filtered = df[df["distance"] < MAX_VALID_DISTANCE].copy()
    if filtered.empty:
        raise ValueError(
            "No quedaron puntos válidos después de filtrar valores de error."
        )
    return filtered


def polar_to_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert polar coordinates (angle_rad, distance) to cartesian (x, y).

    Formulas:
        x = distance * cos(angle_rad)
        y = distance * sin(angle_rad)
    """
    points = df.copy()
    points["x"] = points["distance"] * np.cos(points["angle_rad"])
    points["y"] = points["distance"] * np.sin(points["angle_rad"])
    return points


def classify_points(df: pd.DataFrame, umbral: float = UMBRAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into *nearby* (distance < umbral) and *far*
    (distance >= umbral) subsets.
    """
    cercano = df[df["distance"] < umbral].copy()
    lejano = df[df["distance"] >= umbral].copy()
    return cercano, lejano


def _draw_polar_panel(ax: plt.Axes, df_cercano: pd.DataFrame, df_lejano: pd.DataFrame) -> None:
    """Render the polar scatter panel."""
    ax.scatter(
        df_lejano["angle_rad"],
        df_lejano["distance"],
        c="steelblue",
        s=6,
        alpha=0.55,
        label="Lejano (>1 m)",
        edgecolors="none",
    )
    ax.scatter(
        df_cercano["angle_rad"],
        df_cercano["distance"],
        c="red",
        s=12,
        alpha=0.85,
        label="Cercano (<1 m)",
        edgecolors="none",
    )
    ax.set_title("Vista polar 360°", fontsize=11, pad=14)
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.45, 1.12))
    ax.grid(True, alpha=0.3)


def _draw_top_view_panel(
    ax: plt.Axes,
    df_cercano: pd.DataFrame,
    df_lejano: pd.DataFrame,
) -> None:
    """Render the top-down (vista de planta) panel."""
    ax.scatter(
        df_lejano["x"],
        df_lejano["y"],
        c="steelblue",
        s=9,
        alpha=0.55,
        label="Lejano",
        edgecolors="none",
    )
    ax.scatter(
        df_cercano["x"],
        df_cercano["y"],
        c="red",
        s=18,
        alpha=0.9,
        label="Cercano (<1 m)",
        edgecolors="none",
    )
    ax.plot(0, 0, "k^", markersize=11, label="Sensor LiDAR", zorder=6)

    # ConvexHull for the nearby cluster
    if ConvexHull is not None and len(df_cercano) >= 3:
        pts = df_cercano[["x", "y"]].values
        hull = ConvexHull(pts)
        area = hull.volume  # volume == area in 2‑D
        verts = pts[hull.vertices]
        verts = np.vstack([verts, verts[0]])
        ax.fill(verts[:, 0], verts[:, 1], alpha=0.15, color="red")
        ax.plot(verts[:, 0], verts[:, 1], "r-", lw=1.5, alpha=0.7)
        ax.set_title(f"Vista de planta\nÁrea zona cercana: {area:.3f} m²", fontsize=11)
    else:
        ax.set_title("Vista de planta", fontsize=11)

    ax.set_xlabel("X (metros)")
    ax.set_ylabel("Y (metros)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def generate_figure(df: pd.DataFrame, output_path: Path) -> None:
    """
    Build the two-panel LiDAR analysis figure and save it.

    Panels:
        1. Polar scatter   (angle vs distance)
        2. Top-down view   (x, y) with ConvexHull for nearby cluster
    """
    fig = plt.figure(figsize=(12, 5.5))

    # Convert to cartesian first so both panels can reuse the coordinates
    points = polar_to_cartesian(df)

    df_cercano, df_lejano = classify_points(points)

    ax1 = fig.add_subplot(121, projection="polar")
    _draw_polar_panel(ax1, df_cercano, df_lejano)

    ax2 = fig.add_subplot(122)
    _draw_top_view_panel(ax2, df_cercano, df_lejano)

    fig.suptitle("Análisis LiDAR 2D — Escaneo 360°", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada como {output_path}")

    if INTERACTIVE:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Orchestrate reading, filtering, processing and visualisation."""
    try:
        GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

        raw_df = read_lidar_data(DATA_PATH)
        print(f"Puntos leídos: {len(raw_df)}")

        clean_df = filter_valid_points(raw_df)
        print(f"Puntos válidos después de filtrar: {len(clean_df)}")

        # Single high-quality figure with all three panels
        generate_figure(clean_df, GRAPHICS_DIR / "lidar_resultado.png")

        print("Procesamiento completado.")

    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[ERROR] Error inesperado: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
