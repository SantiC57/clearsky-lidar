.PHONY: setup install test clean shell

# Config
VENV := .venv
UV := $(VENV)/bin/uv
PYTHON := $(VENV)/bin/python

# Detectar shell actual
CURRENT_SHELL := $(shell basename $$SHELL)
ifeq ($(CURRENT_SHELL),fish)
  ACTIVATE := source $(VENV)/bin/activate.fish
else
  ACTIVATE := source $(VENV)/bin/activate
endif

# Un solo comando para todo: crea venv, instala uv, instala deps
setup:
	@echo "🔧 Configurando entorno..."
	python -m venv $(VENV)
	$(VENV)/bin/pip install -q uv
	$(UV) pip install -e .[all]
	@echo "✅ Listo. Activa con: $(ACTIVATE)"

# Solo instalar/actualizar deps (venv ya existe)
install:
	$(UV) pip install -e .[all]

# Tests
test:
	$(PYTHON) -m pytest tests/ -v

# Limpieza
clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache *.egg-info

# Muestra comando de activación según tu shell
shell:
	@echo "$(ACTIVATE)"

# Ayuda
help:
	@echo "Comandos:"
	@echo "  make setup   - Crea venv + instala uv + dependencias (primera vez)"
	@echo "  make install - Actualiza dependencias"
	@echo "  make test    - Ejecuta tests"
	@echo "  make clean   - Borra venv y cachés"
	@echo "  make shell   - Muestra comando para activar el venv"