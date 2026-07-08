# София — ИИ-помощник специалиста на заводе

RAG-бот для ответов на вопросы по производственной документации.
Отвечает на русском языке с указанием источников и картинками.

## Быстрый старт

### 1. Клонируй и запусти

**macOS / Linux:**
```bash
git clone https://github.com/HolmesCodes/Gazprom_chatbot.git
cd Gazprom_chatbot
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Windows:**
```powershell
git clone https://github.com/HolmesCodes/Gazprom_chatbot.git
cd Gazprom_chatbot
.\scripts\setup.ps1
```

### 2. Вставь API-ключ

При первом запуске скрипт создаст файл `.env` и попросит ввести ключ.
Открой `.env` и впиши свой ключ Polza.ai:

```
LLM_API_KEY=pza_твой_ключ_здесь
```

Получить ключ: [polza.ai](https://polza.ai)

### 3. Готово

Сервер запустится на **http://127.0.0.1:8000**

---

## Что внутри

| Компонент | Модель | Где работает |
|---|---|---|
| LLM (ответы) | `deepseek/deepseek-v4-flash` | Polza.ai API |
| Эмбеддинги (поиск) | `qwen/qwen3-embedding-4b` | Polza.ai API |
| Голос (STT) | `gigaam-v3` | Polza.ai API |
| Vision (картинки) | `google/gemini-2.5-flash-lite` | Polza.ai API |

**Всё через один API-ключ Polza.ai** — ничего локально устанавливать не нужно.

---

## Возможности

- **Документы**: PDF, DOCX, TXT, MD, XLSX, CSV, PPTX, HTML
- **RAG-поиск**: MMR + скоринг релевантности
- **Картинки из PDF**: автоизвлечение, галерея в ответе
- **Vision API**: описание картинок через Gemini 2.5 Flash Lite
- **Голос**: микрофон в браузере → GigaAM → текст
- **Инкрементальная индексация**: только изменённые файлы
- **Аутентификация**: логин/пароль
- **Категории**: техкарты, инструкции, регламенты, SOP, охрана труда

---

## Ручная установка

```bash
git clone https://github.com/HolmesCodes/Gazprom_chatbot.git
cd Gazprom_chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# Отредактируй .env — вставь LLM_API_KEY
python -m factory_assistant.cli ingest
python -m factory_assistant.cli serve
```

---

## Команды

| Команда | Описание |
|---|---|
| `python -m factory_assistant.cli serve` | Запуск сервера |
| `python -m factory_assistant.cli ingest` | Индексация документов |
| `python -m factory_assistant.cli ask "вопрос"` | Вопрос из терминала |
| `python -m factory_assistant.cli ingest --recreate` | Пересоздать базу |

---

## Структура

```
├── src/factory_assistant/   # Python-код
│   ├── api.py               # FastAPI-сервер
│   ├── rag.py               # RAG-движок
│   ├── ingest.py            # Парсинг/индексация
│   ├── service.py           # Сервис-слой
│   ├── config.py            # Настройки
│   ├── vision.py            # Vision API
│   ├── gemini_describe.py   # Описание страниц
│   └── prompts.py           # Промпты для LLM
├── web/                     # Фронтенд
├── data/documents/          # Твои документы
├── chroma_db/               # Векторная БД (авто)
└── scripts/                 # Скрипты запуска
```
