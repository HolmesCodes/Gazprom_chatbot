@echo off
chcp 65001 >nul 2>&1
title Factory Assistant

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ============================================
echo   Factory Assistant — ИИ-помощник на заводе
echo ============================================
echo.

REM --- Проверка Python ---
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python не найден.
    echo     Скачайте Python 3.10+ с https://www.python.org/downloads/
    echo     При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version_info[:2])"') do set "PYVER=%%i"
echo [OK] Python найден: %PYVER%
echo.

REM --- Виртуальное окружение ---
if not exist ".venv\Scripts\python.exe" (
    echo [*] Создаю виртуальное окружение...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [!] Ошибка создания venv
        pause
        exit /b 1
    )
    echo [OK] venv создан
) else (
    echo [OK] venv уже есть
)
echo.

REM --- Зависимости ---
".venv\Scripts\python.exe" -c "import langchain" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Устанавливаю зависимости (первый запуск — ~2 мин)...
    ".venv\Scripts\pip.exe" install -q -r requirements.txt
    ".venv\Scripts\pip.exe" install -q -e .
    echo [OK] Зависимости установлены
) else (
    echo [OK] Зависимости уже установлены
)
echo.

REM --- .env файл ---
if not exist ".env" (
    echo [*] Создаю .env из шаблона...
    copy .env.example .env >nul
    echo [!] Откройте .env и впишите ваш API-ключ (LLM_API_KEY=pza_...)
    echo.
    notepad .env
    echo.
    echo После сохранения .env нажмите любую клавишу для продолжения...
    pause >nul
) else (
    echo [OK] .env уже есть
)
echo.

REM --- Индексация ---
if not exist "chroma_db" (
    echo [*] Индексирую документы (первый запуск — ~3 мин)...
    ".venv\Scripts\python.exe" -m factory_assistant.cli ingest
    echo [OK] Документы проиндексированы
) else if not exist "chroma_db\chroma.sqlite3" (
    echo [*] Индексирую документы...
    ".venv\Scripts\python.exe" -m factory_assistant.cli ingest
    echo [OK] Документы проиндексированы
) else (
    echo [OK] Индекс уже есть
)
echo.

REM --- Запуск сервера ---
echo ============================================
echo   Сервер запущен: http://127.0.0.1:8000
echo   Нажмите Ctrl+C для остановки
echo ============================================
echo.
".venv\Scripts\python.exe" -m factory_assistant.cli serve

pause
