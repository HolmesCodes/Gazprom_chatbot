#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RepoDir

Write-Host "=== София — установка и запуск ===" -ForegroundColor Cyan

# Python check
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    Write-Host "Ошибка: Python не найден. Установи Python >= 3.10." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $(& python --version)" -ForegroundColor Green

# Virtual env
if (-not (Test-Path ".venv")) {
    Write-Host "=== Создаю виртуальное окружение ===" -ForegroundColor Yellow
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

Write-Host "=== Устанавливаю зависимости ===" -ForegroundColor Yellow
pip install -e .

# .env
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "=== Создаю .env из .env.example ===" -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "⚠️  Отредактируй .env — вставь свой LLM_API_KEY (polza.ai)" -ForegroundColor Magenta
    } else {
        Write-Host "⚠️  .env.example не найден. Создай .env вручную." -ForegroundColor Magenta
    }
} else {
    Write-Host "=== .env уже существует ===" -ForegroundColor Green
}

# Ingest
Write-Host "=== Индексирую документы ===" -ForegroundColor Yellow
python -m factory_assistant.cli ingest

# Start
Write-Host ""
Write-Host "=== Запускаю сервер ===" -ForegroundColor Cyan
Write-Host "Открой в браузере: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host ""
python -m factory_assistant.cli serve
