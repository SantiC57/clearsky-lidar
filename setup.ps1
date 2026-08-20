#!/usr/bin/env pwsh
# setup.ps1 - Un comando para hacerlo TODO en Windows
# Uso: .\setup.ps1   (o: pwsh setup.ps1)
# NOTA: Ejecuta con: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\setup.ps1

$ErrorActionPreference = "Stop"
$VENV = ".venv"

Write-Host "🔧 Configurando entorno completo..." -ForegroundColor Cyan

# 1. Crear venv si no existe
if (-not (Test-Path $VENV)) {
    Write-Host "  → Creando venv..." -ForegroundColor Yellow
    python -m venv $VENV
}

# 2. Instalar uv dentro del venv
Write-Host "  → Instalando uv..." -ForegroundColor Yellow
& "$VENV\Scripts\pip.exe" install -q uv

# 3. Instalar dependencias del proyecto
Write-Host "  → Instalando dependencias..." -ForegroundColor Yellow
& "$VENV\Scripts\uv.exe" pip install -e .[all]

Write-Host ""
Write-Host "✅ ¡TODO LISTO!" -ForegroundColor Green
Write-Host ""
Write-Host "Para activar el entorno:" -ForegroundColor Cyan
Write-Host "  .\$VENV\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Luego puedes usar:" -ForegroundColor Cyan
Write-Host "  uv pip install <paquete>        # instalar más cosas" -ForegroundColor White
Write-Host "  python -m pytest tests/         # correr tests" -ForegroundColor White
Write-Host "  python slamtec.py               # ejecutar slamtec" -ForegroundColor White
Write-Host "  python -m clearsky_lidar.processing  # procesamiento LiDAR" -ForegroundColor White

# 4. Si se ejecuta con . (dot-source), activar automáticamente
if ($MyInvocation.InvocationName -eq ".") {
    Write-Host ""
    Write-Host "🔄 Activando automáticamente..." -ForegroundColor Yellow
    . "$VENV\Scripts\Activate.ps1"
    Write-Host "✅ Entorno activado. Prompt listo." -ForegroundColor Green
}