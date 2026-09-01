#!/usr/bin/env bash
# setup.sh - Un comando para hacerlo TODO
# Uso: ./setup.sh   (o: bash setup.sh)

set -euo pipefail

VENV=".venv"

echo "🔧 Configurando entorno completo..."

# 1. Crear venv si no existe
if [[ ! -d "$VENV" ]]; then
    echo "  → Creando venv..."
    python -m venv "$VENV"
fi

# 2. Instalar uv dentro del venv
echo "  → Instalando uv..."
"$VENV/bin/pip" install -q uv

# 3. Instalar dependencias del proyecto
echo "  → Instalando dependencias..."
"$VENV/bin/uv" pip install -e .[all]

# 4. Detectar shell y mostrar comando de activación
SHELL_NAME=$(basename "${SHELL:-bash}")
if [[ "$SHELL_NAME" == "fish" ]]; then
    ACTIVATE="source $VENV/bin/activate.fish"
else
    ACTIVATE="source $VENV/bin/activate"
fi

echo ""
echo "✅ ¡TODO LISTO!"
echo ""
echo "Para activar el entorno:"
echo "  $ACTIVATE"
echo ""
echo "Luego puedes usar:"
echo "  uv pip install <paquete>   # instalar más cosas"
echo "  python -m pytest tests/    # correr tests"
echo "  python slamtec.py          # ejecutar slamtec"
echo "  python -m clearsky_lidar.processing   # procesamiento LiDAR"

# 5. Si el script se ejecuta con source (source ./setup.sh), activar automáticamente
# Nota: esto solo funciona si haces: source ./setup.sh
if [[ "${BASH_SOURCE[0]}" != "${0}" ]] || [[ -n "${ZSH_EVAL_CONTEXT:-}" && "${ZSH_EVAL_CONTEXT}" =~ ^file ]]; then
    echo ""
    echo "🔄 Activando automáticamente..."
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
    echo "✅ Entorno activado. Prompt listo."
fi