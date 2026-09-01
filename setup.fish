#!/usr/bin/env fish
# setup.fish - Un comando para hacerlo TODO (versión fish)
# Uso: ./setup.fish   (o: fish setup.fish)

set VENV ".venv"

echo "🔧 Configurando entorno completo..."

# 1. Crear venv si no existe
if not test -d $VENV
    echo "  → Creando venv..."
    python -m venv $VENV
end

# 2. Instalar uv dentro del venv
echo "  → Instalando uv..."
$VENV/bin/pip install -q uv

# 3. Instalar dependencias del proyecto
echo "  → Instalando dependencias..."
$VENV/bin/uv pip install -e .[all]

echo ""
echo "✅ ¡TODO LISTO!"
echo ""
echo "Para activar el entorno:"
echo "  source $VENV/bin/activate.fish"
echo ""
echo "Luego puedes usar:"
echo "  uv pip install <paquete>   # instalar más cosas"
echo "  python -m pytest tests/    # correr tests"
echo "  python slamtec.py          # ejecutar slamtec"
echo "  python -m clearsky_lidar.processing   # procesamiento LiDAR"

# 4. Si se ejecuta con source, activar automáticamente
if status --is-interactive
    echo ""
    echo "🔄 Activando automáticamente..."
    source $VENV/bin/activate.fish
    echo "✅ Entorno activado. Prompt listo."
end