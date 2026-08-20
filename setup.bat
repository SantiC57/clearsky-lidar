@echo off
REM setup.bat - Un comando para hacerlo TODO en Windows (cmd)
REM Uso: setup.bat
REM      O: cmd /k setup.bat   (para auto-activar y quedarse en el shell)

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

REM 4. Detectar si se llamó con "cmd /k" para auto-activar
if "%CMD_AUTO_ACTIVATE%"=="1" (
    echo 🔄 Activando entorno automáticamente...
    call %VENV%\Scripts\activate.bat
    echo ✅ Entorno activado. Prompt listo.
) else (
    echo Para activar el entorno AHORA:
    echo   cmd /k setup.bat
    echo.
    echo O en dos pasos:
    echo   setup.bat
    echo   %VENV%\Scripts\activate.bat
)