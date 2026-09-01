name: PR Template
description: Plantilla para Pull Requests
title: "<tipo>(scope): descripción breve"
body:
  - type: checkboxes
    id: preflight
    attributes:
      label: Pre-flight
      description: Confirma antes de abrir el PR
      options:
        - label: El PR vincula un issue aprobado (Closes #N, Fixes #N, Resolves #N)
          required: true
        - label: El issue vinculado tiene etiqueta `approved`
          required: true
        - label: He ejecutado tests localmente: `pytest tests/ -v -k "not open3d"`
          required: true
        - label: El código pasa `ruff check .` / `black --check .` (si aplica)
          required: false

  - type: textarea
    id: what
    attributes:
      label: Qué Cambia (Lead with the answer)
      description: Una frase: qué se agrega/arregla y para qué
      placeholder: "Agrega setup-jetson.sh para configurar entorno en Jetson Nano (Ubuntu 18.04 / Python 3.7) sin compilar open3d"
    validations:
      required: true

  - type: textarea
    id: why
    attributes:
      label: Por Qué / Contexto
      description: Qué problema resuelve, por qué esta aproximación
      placeholder: |
        El setup.sh original usa `python -m venv` (Python 3.6 en Jetson) e instala `[all]` (incluye open3d, que no tiene wheels ARM64 para Py3.7 y tarda horas en compilar).
        Este script usa python3.7 explícito, pinea pip/setuptools compatibles, e instala solo [processing,dev].

  - type: textarea
    id: how
    attributes:
      label: Cómo Probar (Verification)
      description: Pasos para que el reviewer verifique
      placeholder: |
        1. En Jetson (o docker ubuntu:18.04):
           source ./setup-jetson.sh
        2. Verificar:
           python -m clearsky_lidar.processing
           python -m pytest tests/ -v -k "not open3d"
        3. Confirmar que NO se instala open3d

  - type: textarea
    id: scope
    attributes:
      label: Fuera de Alcance (Out of Scope)
      description: Qué NO toca este PR — evita scope creep
      placeholder: |
        - No cambia slamtec.py ni processing.py
        - No modifica CI (jetson-compat ya valida Py3.7)
        - No actualiza versión (se hará en release aparte)

  - type: textarea
    id: related
    attributes:
      label: Issues / PRs Relacionados
      description: Links a issues, PRs previos, docs
      placeholder: |
        Closes #6
        Docs: docs/ci-jetson-setup.md, docs/pr-approval-workflow.md