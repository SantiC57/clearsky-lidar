# CI en la Jetson Nano — setup y notas

## Requisito de Python

**Este proyecto requiere Python 3.7+.**

La Jetson Nano con Ubuntu 18.04 trae Python 3.6 por defecto, que **no es
compatible** con este código:

- `from __future__ import annotations` (PEP 563) es sintaxis de Python 3.7+.
- Los type hints modernos (`tuple[...]`, `list[...]`) dependen de esa
  anotación diferida para funcionar en 3.7/3.8.
- Las dependencias base (numpy, pandas, matplotlib, scipy) no publican
  versiones recientes para Python 3.6.

## Instalación de Python 3.7 en la Jetson (Ubuntu 18.04)

Opciones:

1. **bionic-updates (recomendado)** — el paquete oficial viene en los repos
   de Ubuntu 18.04:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.7 python3.7-dev python3.7-distutils
   ```

2. **Deadsnakes PPA** — si bionic-updates no alcanza:

   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install -y python3.7 python3.7-dev python3.7-distutils
   ```

## Uso en CI

El job `jetson-compat` de `.github/workflows/ci.yml` valida la compatibilidad
del código contra el entorno de la Jetson (ARM64 · Ubuntu 18.04 · Python 3.7)
usando QEMU y el mismo flujo de instalación de arriba. Es el mismo script que
debería correr en un self-hosted runner etiquetado como `jetson-nano` cuando
haya acceso físico al hardware.

```bash
# Validación local con act (requiere QEMU binfmt arm64 registrado):
act -j jetson-compat --container-architecture linux/arm64 --container-options "--privileged"
```

## Nota: open3d

`open3d` se usa únicamente en `Mapper.save_pcd()` (I/O de CPU, sin CUDA).
El job `jetson-compat` lo excluye a propósito (`.[processing,dev]`) para
evitar compilarlo bajo emulación; en la Jetson real se instala igual que en
cualquier otra plataforma.