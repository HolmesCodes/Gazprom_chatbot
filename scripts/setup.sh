#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== София — установка и запуск ==="

# Python check
PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Ошибка: Python не найден. Установи Python >= 3.10."
  exit 1
fi

echo "Python: $($PYTHON --version)"

# Virtual env
if [ ! -d .venv ]; then
  echo "=== Создаю виртуальное окружение ==="
  $PYTHON -m venv .venv
fi
source .venv/bin/activate

echo "=== Устанавливаю зависимости ==="
pip install -e .

# .env
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo "=== Создаю .env из .env.example ==="
    cp .env.example .env
    echo "⚠️  Отредактируй .env — вставь свой LLM_API_KEY (polza.ai)"
  else
    echo "⚠️  .env.example не найден. Создай .env вручную."
  fi
else
  echo "=== .env уже существует ==="
fi

# Ingest
echo "=== Индексирую документы ==="
$PYTHON -m factory_assistant.cli ingest

# Start
echo ""
echo "=== Запускаю сервер ==="
echo "Открой в браузере: http://127.0.0.1:8000"
echo ""
$PYTHON -m factory_assistant.cli serve
