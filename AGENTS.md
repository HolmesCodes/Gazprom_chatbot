# AGENTS.md — Информация для ИИ-агента

## Проект

**ИИ-помощник специалиста на заводе** — RAG-бот для ГПН Школа А5.
Отвечает на вопросы по внутренней производственной документации на русском языке с указанием источников.

---

## Быстрый старт

```bash
cd /Users/vasiliyafonin/projects/gpn-factory-assistant
source .venv/bin/activate

# Запуск сервера
python -m factory_assistant.cli serve
# → http://127.0.0.1:8000

# Индексация
python -m factory_assistant.cli ingest

# Вопрос через CLI
python -m factory_assistant.cli ask "вопрос про документы"
```

---

## Архитектура

```
                   ┌─────────────┐
                   │  web/       │ ← HTML/CSS/JS (SPA)
                   │  index.html │
                   └──────┬──────┘
                          │ fetch /api/...
                          ▼
                   ┌─────────────┐
                   │  api.py     │ ← FastAPI сервер
                   │             │   routes, CORS, lifespan
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │  service.py │ ← FactoryAssistantService
                   │             │   ask(), transcribe(), reindex()
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │ rag.py      │ ← RagEngine
                   │             │   ChatOpenAI (Polza.ai)
                   │             │   OllamaEmbeddings (локально)
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │ ingest.py   │ ← загрузка/парсинг/чанкинг
                   │ ChromaDB    │ ← векторное хранилище
                   └─────────────┘
```

---

## Стек

| Компонент | Технология |
|---|---|
| Язык | Python 3.10+ |
| Веб-фреймворк | FastAPI + Uvicorn |
| RAG-пайплайн | LangChain |
| Векторная БД | ChromaDB 1.5.9 (Rust-бэкенд) |
| LLM | Polza.ai: `deepseek/deepseek-v4-flash` (API) |
| STT | Polza.ai: `openai/whisper-1` (API) |
| Эмбеддинги | Ollama: `nomic-embed-text:latest` (локально) |
| PDF-парсер | PyMuPDFLoader (pymupdf) |
| DOCX-парсер | Docx2txtLoader |
| Парсинг .txt/.md | TextLoader (UTF-8) |
| Фронтенд | HTML + CSS + vanilla JS |
| CLI | argparse |

---

## Структура файлов

### Папки верхнего уровня

| Путь | Назначение |
|---|---|
| `src/factory_assistant/` | Исходный код |
| `web/` | Веб-интерфейс (index.html, app.js, style.css) |
| `data/documents/` | Документы для индексации (PDF/DOCX/TXT/MD) |
| `chroma_db/` | Векторная база ChromaDB (создаётся автоматически) |
| `.venv/` | Виртуальное окружение Python |
| `scripts/` | (пусто) |
| `.env.example` | Пример настроек |

### Исходный код (`src/factory_assistant/`)

| Файл | Что делает | Ключевые классы/функции |
|---|---|---|
| `cli.py` | Точка входа: `factory-ingest`, `factory-serve`, `ask` | `ingest_main()`, `serve_main()`, `main()` |
| `api.py` | FastAPI-сервер, CORS, статика | `app`, `lifespan()`, все эндпоинты |
| `service.py` | Сервис-слой: связывает RAG + STT | `FactoryAssistantService` |
| `rag.py` | RAG-движок: поиск + LLM | `RagEngine`, `RagAnswer`, `SourceReference` |
| `ingest.py` | Загрузка, парсинг, чанкинг, индексация | `load_documents()`, `split_documents()`, `build_vectorstore()`, `ingest_documents()`, `_clean_pdf_text()`, `_loader_for()`, `_enrich_metadata()` |
| `config.py` | Настройки (pydantic-settings) | `Settings` |
| `admin.py` | Статус индекса, список моделей Ollama | `get_index_status()`, `list_ollama_models()`, `list_document_files()`, `preview_index_plan()` |
| `documents.py` | Загрузка/удаление файлов | `save_upload()`, `delete_document()` |
| `prompts.py` | Системные промпты для LLM | `RAG_SYSTEM_PROMPT`, `RAG_USER_TEMPLATE` |

---

## Настройки (`src/factory_assistant/config.py`)

```python
chunk_size: int = 1500          # размер одного чанка (символов)
chunk_overlap: int = 200        # перекрытие между чанками
top_k: int = 5                  # сколько чанков отдавать LLM
fetch_k: int = 20               # сколько чанков рассматривать при MMR
relevance_threshold: float = 0.35  # порог релевантности
llm_model: str = "deepseek/deepseek-v4-flash"
embedding_model: str = "nomic-embed-text:latest"
```

Меняются через `.env` или через API `PATCH /api/admin/config`.

---

## Эндпоинты API

| Метод | Путь | Тело запроса | Описание |
|---|---|---|---|
| GET | `/health` | — | Проверка сервиса |
| GET | `/api/config` | — | Текущая конфигурация |
| PATCH | `/api/admin/config` | `{llm_model?, embedding_model?, chunk_size?, ...}` | Обновить настройки |
| GET | `/api/admin/models` | — | Список моделей Ollama |
| GET | `/api/admin/index/status` | — | Статус индексации |
| GET | `/api/admin/index/preview` | — | План индексации |
| POST | `/api/ask` | `{question}` | Задать вопрос |
| POST | `/api/admin/reindex` | `{recreate: true}` | Переиндексация |
| POST | `/api/admin/documents/upload` | file + `?auto_reindex=true/false` | Загрузить документ |
| DELETE | `/api/admin/documents/{filename}` | `?auto_reindex=true/false` | Удалить документ |
| POST | `/api/onboarding/start` | `{session_id?}` | Начать онбординг |
| POST | `/api/onboarding/message` | `{session_id, message}` | Шаг онбординга |

**Legacy эндпоинты** (без `/api/`): `/ask`, `/onboarding/start`, `/onboarding/message`, `/admin/reindex`.

---

## Поток обработки вопроса (RAG)

```
Вопрос пользователя
    │
    ▼
MMR-поиск (self.retriever.invoke)
  — search_type="mmr"
  — k=top_k, fetch_k=fetch_k
  — возвращает top_k документов
    │
    ▼
Relevance scoring
  — similarity_search_with_relevance_scores(question, k=top_k)
  — max_score < threshold → "Информация не найдена"
    │
    ▼
LLM chain
  — context = format_docs(retrieved_docs)
  — prompt: RAG_SYSTEM_PROMPT + RAG_USER_TEMPLATE
  — model: ChatOpenAI(deepseek-v4-flash, temperature=0.1)
  — если LLM ответила "Информация не найдена" → финальный ответ "не найдено"
    │
    ▼
Ответ с источниками (RagAnswer)
```

---

## Поток индексации

```
data/documents/ (PDF/DOCX/TXT/MD)
    │
    ▼
_loader_for(path) → PyMuPDFLoader | Docx2txtLoader | TextLoader
    │
    ▼
_enrich_metadata() + _clean_pdf_text() (для PDF)
  — удаление артефактов: /emdash.cyr → —, \\команды
  — удаление image-референсов: ![](image.png), Рис. 1, [image]
  — схлопывание пробелов/пустых строк
    │
    ▼
RecursiveCharacterTextSplitter
  — separators: ["\nЗадача ", "\nПодзадача ", "\n\n", "\n", " ", ""]
  — chunk_size=1500, chunk_overlap=200
    │
    ▼
ChromaDB.add_documents(chunks)
  — embedding: nomic-embed-text (Ollama)
  — collection: factory_docs
```

---

## Известные особенности

### ChromaDB readonly error
При переиндексации через `service.reindex()` не используется `shutil.rmtree` + `mkdir`.
Чистка делается через `collection.delete(ids=all_ids)` внутри открытой сессии Chroma,
либо через удаление и пересоздание коллекции через `chromadb.PersistentClient`.

### RagEngine не падает при пустой chroma_db
`RagEngine.__init__` вызывает `_try_load_vectorstore()` вместо `_load_vectorstore()`.
Если база пуста — `vectorstore = None`, `ask()` возвращает "Информация не найдена",
сервер не падает. После `reindex()` происходит `_try_load_vectorstore()` повторно.

### Сервер без chroma_db
Сервер (FastAPI + uvicorn) запускается даже если `chroma_db/` не существует.
При старте `FactoryAssistantService.__init__` → `RagEngine.__init__` → `_try_load_vectorstore()` → `False`.

### PDF-парсинг
Используется `PyMuPDFLoader` (pymupdf). Если PDF содержит отсканированные страницы
(изображения вместо текста), текст не извлекается — OCR не подключён.
Для таких PDF нужно добавлять OCR-обработку (pytesseract, pdf2image).

---

## Команды для разработки

```bash
# Установка в режиме editable
source .venv/bin/activate
pip install -e .

# Запуск сервера
python -m factory_assistant.cli serve

# Индексация (с пересозданием chroma_db)
python -m factory_assistant.cli ingest

# Индексация (без пересоздания)
python -m factory_assistant.cli ingest --no-recreate

# Один вопрос
python -m factory_assistant.cli ask "ваш вопрос"

# Запуск через uvicorn напрямую
uvicorn factory_assistant.api:app --host 127.0.0.1 --port 8000

---

## opencode: двухмодельная конфигурация

Настроено в `opencode.json`:

| Модель | Где используется |
|---|---|
| `deepseek/deepseek-v4-flash-free` | **По умолчанию** — вся рутинная работа: чтение кода, рефакторинг, вёрстка, коммиты |
| `anthropic/claude-opus-4-6` | **Только через @opus** — сложные архитектурные решения, планы, критический ревью |

Правила работы:
1. Рутинные задачи делает DeepSeek (основной чат) — это безлимитно
2. Claude Opus (220 запросов) вызывается только через `@opus <задача>` — строго для сложного
3. Один `@opus` вызов = до 50 шагов инструментов (чтение файлов, написание кода) — всё в рамках одного запроса

Промпт Claude Opus: `.opencode/agent/opus.md`

> **Важно:** после изменения `opencode.json` — **перезапусти opencode** (выход и вход). Конфиг не hot-reload.
```
