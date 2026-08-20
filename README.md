# ClearSky LiDAR

Biblioteca Python para captura, procesamiento y visualización de datos LiDAR del sensor **SLAMTEC M1M1** en el proyecto ClearSky.

## Qué hace este repo

| Módulo | Función |
|--------|---------|
| `slamtec.py` | Cliente TCP crudo para el protocolo JSON del M1M1 (puerto 1445) |
| `clearsky_lidar/mapper.py` | Wrapper alto nivel: escaneo, pose, guardado PCD (Open3D) |
| `clearsky_lidar/processing.py` | Lee `dump/laser-full.csv`, filtra, convierte polar→cartesiano, genera gráfica 2-panel (polar + top-down con ConvexHull) |
| `clearsky_lidar/detection.py` | (pendiente) Detección y clasificación de residuos |
| `clearsky_lidar/fusion.py` | (pendiente) Fusión LiDAR + cámara |

---

## Instalación y uso por sistema operativo

> **Requisitos previos:** Python 3.10+ (3.12 recomendado para `open3d`)

| OS / Shell | Comando único (instala + activa) | Qué hace |
|------------|----------------------------------|----------|
| **Linux / macOS** · **fish** | `source ./setup.fish` | Crea `.venv`, instala `uv`, instala `.[all]`, **activa el venv** |
| **Linux / macOS** · **bash / zsh** | `source ./setup.sh` | Igual que arriba, para bash/zsh |
| **Windows** · **PowerShell** | `. .\setup.ps1` | Igual, usa dot-source (`. `) para activar en la sesión actual |
| **Windows** · **cmd** | `setup-auto.bat` | Lanza `cmd /k`, instala y **se queda en el prompt activado** |
| **Windows + Git Bash** | `source ./setup.sh` | Git Bash trae `bash` nativo; funciona igual que Linux |

> **Nota:** Si solo ejecutas `./setup.fish` o `./setup.sh` **sin `source`**, te muestra el comando de activación pero **no lo activa**. Con `source` / `.` te deja dentro del venv listo para trabajar.

---

## Estructura de carpetas y requisitos de ejecución

```
clearsky-lidar/
├── clearsky_lidar/          # Paquete principal
│   ├── __init__.py          # Exporta: Mapper, processing (lazy)
│   ├── mapper.py            # Requiere conexión TCP al M1M1
│   ├── processing.py        # Requiere dump/laser-full.csv
│   ├── detection.py         # (pendiente)
│   └── fusion.py            # (pendiente)
├── config/
│   └── config_lidar.yaml    # IP, puerto, rangos, frecuencia, altura montaje
├── dump/                    # ← DEBE EXISTIR para processing
│   └── laser-full.csv       # Generado por slamtec.py (ver abajo)
├── Graphics/                # Salida de gráficas (se crea solo)
├── test_data/               # JSONs de prueba para tests
├── tests/                   # pytest unitarios + integración
├── slamtec.py               # Script standalone para captura
├── pyproject.toml           # Deps: numpy, pandas, matplotlib, scipy, Pillow, pytest, pyyaml
├── setup.fish / setup.sh    # Linux/macOS (source para auto-activar)
├── setup.ps1                # Windows PowerShell (. para auto-activar)
└── setup-auto.bat           # Windows cmd (un clic, se queda activado)
```

### 1. `slamtec.py` / `clearsky_lidar/mapper.py` — Captura real

**Requisitos de hardware:**
- M1M1 conectado por **Ethernet** (IP por defecto `192.168.11.1`, puerto `1445`)
- **Opcional:** USB para alimentación (el M1M1 suele alimentarse por PoE o USB)
- Tu PC en la misma subred (ej. `192.168.11.x`)

**Ejecutar:**
```bash
# Activa venv primero (ver tabla arriba)
python slamtec.py
# O desde el paquete:
python -c "from clearsky_lidar import Mapper; m=Mapper(); m.connect(); pts=m.scan(); m.save_pcd(pts); m.disconnect()"
```

**Qué hace:**
1. Conecta por TCP al M1M1
2. Pide `getlaserscan` → recibe puntos RLE comprimidos
3. Descomprime, filtra `distance == 100000` (inválidos)
4. Guarda `dump/laser-full.csv` con columnas: `angle_rad,distance,angle_deg`
5. Opcional: `save_pcd()` escribe `.pcd` (requiere extra `lidar` → `open3d`)

> El CSV generado en `dump/` es **exactamente lo que necesita `processing.py`**.

---

### 2. `clearsky_lidar/processing.py` — Procesamiento offline

**Requisitos:**
- Carpeta `dump/` **existente**
- Archivo `dump/laser-full.csv` **presente** (generado por `slamtec.py` o puesto manualmente)
- Formato CSV **sin cabecera**, 3 columnas: `angle_rad, distance, angle_deg`
- Distancias en **metros** (valores válidos ~0.15–20 m; sentinela 100000 = error)

**Ejecutar:**
```bash
# Activa venv primero
python -m clearsky_lidar.processing
# O directo:
python clearsky_lidar/processing.py
```

**Qué hace:**
1. Lee `dump/laser-full.csv`
2. Fila sentinelas (`distance >= 50000`)
3. Convierte polar → cartesiano (`x = d·cos(θ)`, `y = d·sin(θ)`)
4. Clasifica: **cercano** (< 1 m) vs **lejano** (≥ 1 m)
5. Calcula **ConvexHull** de puntos cercanos (requiere `scipy`, extra `processing`)
6. Genera `Graphics/lidar_resultado.png` con 2 paneles:
   - **Izq:** Scatter polar (ángulo vs distancia)
   - **Der:** Vista top-down (x, y) + hull rojo + sensor en (0,0)
7. Si hay backend GUI (TkAgg/Qt5Agg) → **muestra ventana interactiva** (`plt.show()`)

---

### 3. Tests

```bash
python -m pytest tests/ -v
```

- `test_mapper.py`: unidad (sin hardware) + carga de `config_lidar.yaml`
- `test_integration.py`: integra `open3d` (marca `skip` si no instalado)

---

## Extras opcionales

```bash
# Solo lo que usa processing.py (scipy para ConvexHull)
uv pip install -e .[processing]

# Solo visualización (Pillow para show_map en slamtec.py)
uv pip install -e .[visualization]

# open3d (guardar PCD) — Python 3.10-3.12 solamente
uv pip install -e .[lidar]

# Todo desarrollo
uv pip install -e .[all]      # (no incluye open3d en Py3.14+)
uv pip install -e .[all,lidar]  # si tu Python soporta open3d
```

---

## Configuración (`config/config_lidar.yaml`)

```yaml
lidar:
  host: "192.168.11.1"     # IP del M1M1
  port: 1445               # Puerto TCP JSON
  valid_only: true         # Descartar lecturas inválidas
  pcd_output: "primer_escaneo.pcd"
  rango_min_m: 0.15
  rango_max_m: 20.0
  frecuencia_hz: 8
  altura_montaje_m: 4.0    # Altura del LiDAR sobre el suelo (dron)
```

---

## Solución de problemas comunes

| Problema | Causa | Solución |
|----------|-------|----------|
| `ModuleNotFoundError: numpy` | Venv no activado | Ejecuta el comando `source` / `.` de la tabla |
| `libtk8.6.so not found` | Falta `tk` en Linux | `sudo pacman -S tk` / `sudo apt install tk-dev` / `brew install python-tk` |
| Ventana no aparece (`plt.show()`) | Backend `Agg` (headless) | Instala `tk` o `PyQt5`; el script detecta y usa `TkAgg`/`Qt5Agg` |
| `open3d` no instala | Python 3.14+ sin wheels | Usa Python 3.12 o instala desde source |
| `Connection refused` en slamtec.py | IP/puerto incorrecto o M1M1 apagado | Verifica `config_lidar.yaml`, ping al M1M1, firewall |

---

## Licencia

MIT — ver `LICENSE` (pendiente de añadir).