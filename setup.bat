@echo off
REM setup.bat - Un comando para hacerlo TODO en Windows (cmd)
REM Uso: setup.bat

set VENV=.venv

echo 🔧 Configurando entorno completo...

REM 1. Crear venv si no existe
if not exist %VENV% (
    echo   ^> Creando venv...
    python -m venv %VENV%
)

REM 2. Instalar uv dentro del venv
echo   ^> Instalando uv...
%VENV%\Scripts\pip.exe install -q uv

REM 3. Instalar dependencias del proyecto
echo   ^> Instalando dependencias...
%VENV%\Scripts\uv.exe pip install -e .[all]

echo.
echo ✅ ¡TODO LISTO!
echo.
echo Para activar el entorno:
echo   %VENV%\Scripts\activate.bat
echo.
echo Luego puedes usar:
echo   uv pip install ^<paquete^>         ^(instalar más cosas^)
echo   python -m pytest tests/            ^(correr tests^)
echo   python slamtec.py                  ^(ejecutar slamtec^)
echo   python -m clearsky_lidar.processing  ^(procesamiento LiDAR^)