#!/usr/bin/env bash
# setup-jetson.sh - Setup para Jetson (Ubuntu 18.04 / Python 3.7)
# Uso: source ./setup-jetson.sh

set -euo pipefail

VENV=".venv"
PYTHON="python3.7"

echo "🔧 Configurando entorno para Jetson (Python 3.7)..."

# 1. Verificar Python 3.7
if ! command -v "$PYTHON" &>/dev/null; then
    echo "❌ No se encuentra $PYTHON. Instalalo primero:"
    echo "   sudo apt-get update && sudo apt-get install -y python3.7 python3.7-dev python3.7-distutils python3.7-venv"
    return 1 2>/dev/null || exit 1
fi

# 2. Crear venv con Python 3.7
if [[ ! -d "$VENV" ]]; then
    echo "  → Creando venv con $PYTHON..."
    "$PYTHON" -m venv "$VENV"
fi

# 3. Pip/setuptools compatibles con 3.7
echo "  → Actualizando pip/setuptools (compatibles con 3.7)..."
"$VENV/bin/pip" install -q --upgrade "pip<24" "setuptools<69"

# 4. Instalar SOLO processing + dev (SIN open3d)
echo "  → Instalando dependencias base + scipy..."
"$VENV/bin/pip" install -q -e ".[processing,dev]"

# 5. Activar si se ejecuta con source
if [[ "${BASH_SOURCE[0]}" != "${0}" ]] || [[ -n "${ZSH_EVAL_CONTEXT:-}" && "${ZSH_EVAL_CONTEXT}" =~ ^file ]]; then
    echo ""
    echo "🔄 Activando automáticamente..."
    source "$VENV/bin/activate"
    echo "✅ Entorno Jetson listo."
else
    echo ""
    echo "✅ ¡LISTO! Activá con:"
    echo "  source $VENV/bin/activate"
fi