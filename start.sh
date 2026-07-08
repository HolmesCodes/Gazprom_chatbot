#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  Factory Assistant — ИИ-помощник на заводе"
echo "============================================"
echo

# --- Проверка Python ---
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[!] Python не найден. Установите Python 3.10+"
    exit 1
fi
PY=$(command -v python3 || command -v python)
echo "[OK] Python: $($PY --version)"
echo

# --- venv ---
if [ ! -f ".venv/bin/python" ] && [ ! -f ".venv/Scripts/python.exe" ]; then
    echo "[*] Создаю venv..."
    $PY -m venv .venv
    echo "[OK] venv создан"
fi
VENV_PY=".venv/bin/python"
[ -f ".venv/Scripts/python.exe" ] && VENV_PY=".venv/Scripts/python.exe"
echo "[OK] venv найден"
echo

# --- Зависимости ---
if ! $VENV_PY -c "import langchain" 2>/dev/null; then
    echo "[*] Устанавливаю зависимости..."
    $VENV_PY -m pip install -q -r requirements.txt
    $VENV_PY -m pip install -q -e .
    echo "[OK] Зависимости установлены"
else
    echo "[OK] Зависимости уже установлены"
fi
echo

# --- .env ---
if [ ! -f ".env" ]; then
    echo "[*] Создаю .env из шаблона..."
    cp .env.example .env
    echo "[!] Откройте .env и впишите LLM_API_KEY"
    ${EDITOR:-nano} .env
else
    echo "[OK] .env уже есть"
fi
echo

# --- Индексация ---
if [ ! -f "chroma_db/chroma.sqlite3" ]; then
    echo "[*] Индексирую документы..."
    $VENV_PY -m factory_assistant.cli ingest
    echo "[OK] Документы проиндексированы"
else
    echo "[OK] Индекс уже есть"
fi
echo

# --- Запуск ---
echo "============================================"
echo "  Сервер: http://127.0.0.1:8000"
echo "  Ctrl+C — остановка"
echo "============================================"
echo
$VENV_PY -m factory_assistant.cli serve
