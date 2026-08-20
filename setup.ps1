#!/usr/bin/env pwsh
# setup.ps1 - Un comando para hacerlo TODO en Windows
# Uso: .\setup.ps1   (o: pwsh setup.ps1)
# NOTA: Ejecuta con: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\setup.ps1
# IMPORTANTE: mantener este archivo en ASCII puro. PowerShell 5.1 lee los .ps1
# como ANSI (CP1252): un caracter UTF-8 como la flecha '->' contiene el byte
# 0x92, que en CP1252 es una comilla y rompe el parser.

$ErrorActionPreference = "Stop"
$VENV = ".venv"

Write-Host "[setup] Configurando entorno completo..." -ForegroundColor Cyan

# 1. Crear venv si no existe
if (-not (Test-Path $VENV)) {
    Write-Host "  -> Creando venv..." -ForegroundColor Yellow
    python -m venv $VENV
}

# 2. Instalar uv dentro del venv
Write-Host "  -> Instalando uv..." -ForegroundColor Yellow
& "$VENV\Scripts\pip.exe" install -q uv

# 3. Instalar dependencias del proyecto
Write-Host "  -> Instalando dependencias..." -ForegroundColor Yellow
& "$VENV\Scripts\uv.exe" pip install -e .[all]

Write-Host ""
Write-Host "[OK] TODO LISTO!" -ForegroundColor Green
Write-Host ""
Write-Host "Para activar el entorno:" -ForegroundColor Cyan
Write-Host "  .\$VENV\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Luego puedes usar:" -ForegroundColor Cyan
Write-Host "  uv pip install <paquete>        # instalar mas cosas" -ForegroundColor White
Write-Host "  python -m pytest tests/         # correr tests" -ForegroundColor White
Write-Host "  python slamtec.py               # ejecutar slamtec" -ForegroundColor White
Write-Host "  python -m clearsky_lidar.processing  # procesamiento LiDAR" -ForegroundColor White

# 4. Si se ejecuta con . (dot-source), activar automaticamente
# NOTA: un script NO puede activar el venv en tu sesion actual salvo que se
# ejecute con dot-source (el mismo comando con un punto adelante):
#   . .\setup.ps1
if ($MyInvocation.InvocationName -eq ".") {
    Write-Host ""
    Write-Host "[!] Activando automaticamente..." -ForegroundColor Yellow
    . "$VENV\Scripts\Activate.ps1"
    Write-Host "[OK] Entorno activado. Prompt listo." -ForegroundColor Green
    Write-Host "Usa: python slamtec.py  |  python -m pytest tests/" -ForegroundColor White
}
else {
    Write-Host ""
    Write-Host "[TIP] Para instalar Y activar en un solo paso:" -ForegroundColor Cyan
    Write-Host "  . .\setup.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Es el mismo comando con un punto adelante (dot-source)." -ForegroundColor White
    Write-Host "Solo con el punto la activacion queda en tu sesion actual." -ForegroundColor White
}