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

Все модели подключаются через **один API-ключ Polza.ai** (OpenAI-совместимый endpoint) — локально ставить нечего. Модели настраиваются через `.env` и панель администратора.

| Компонент | Модель по умолчанию | Назначение | Где меняется |
|---|---|---|---|
| LLM (генерация ответа) | `deepseek/deepseek-v4-flash` | Формирует ответ по контексту | `LLM_MODEL`, панель администратора |
| Эмбеддинги (векторный поиск) | `qwen/qwen3-embedding-4b` | Превращает текст в векторы | `EMBEDDING_MODEL`* |
| Vision (описание картинок) | `google/gemini-2.5-flash-lite` | Описывает извлечённые из PDF картинки | `GEMINI_MODEL` |
| STT (голос → текст) | `gigaam-v3` | Распознавание речи из браузера | `WHISPER_MODEL` |

\* Смена модели эмбеддинга **требует переиндексации** (старые векторы несовместимы с новой моделью) — панель администратора запускает её автоматически в фоне.

**Важно про эмбеддинги:** по умолчанию используется API Polza.ai (`qwen/qwen3-embedding-4b`). Если `LLM_API_BASE_URL` недоступен, код автоматически падает до локального Ollama (`nomic-embed-text`).

---

## Архитектура

### Общая схема

```
        ┌─────────────┐
        │   Браузер   │  web/index.html + app.js (SPA, vanilla JS)
        │  (SPA, PDF.js)│
        └──────┬──────┘
               │  fetch /api/...
               ▼
        ┌─────────────┐
        │  FastAPI    │  api.py — роуты, CORS, статические файлы, auth
        │   (Uvicorn) │
        └──────┬──────┘
               │
               ▼
        ┌──────────────────────────────┐
        │ FactoryAssistantService       │  service.py — оркестрация
        │  ├─ RagEngine (rag.py)        │  RAG-движок: поиск + LLM + фильтр картинок
        │  └─ ingest / reindex          │  ingest.py — парсинг, чанкинг, индексация
        └──────────┬───────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ ChromaDB     │        │ Polza.ai API │  LLM / Embeddings / Vision / STT
│ (векторы)    │        │ (OpenAI-like)│
└──────────────┘        └──────────────┘
```

### Поток обработки вопроса (`POST /api/ask`)

| Шаг | Где | Что происходит | Замер времени |
|---|---|---|---|
| 1. Приём запроса | `api.py` → `service.ask` | Валидация, передача в `RagEngine.ask` | — |
| 2. RAG-поиск | `rag.py:_retrieve` | MMR-ретривер (`search_type="mmr"`, `k=top_k`, `fetch_k`) | `retrieval_ms` (до вызова LLM) |
| 3. Скоринг релевантности | `rag.py:ask` | `similarity_search_with_relevance_scores`, отсев по `relevance_threshold` | входит в `retrieval_ms` |
| 4. Сборка контекста | `format_docs` | Объединение чанков в промпт | входит в `retrieval_ms` |
| 5. Генерация ответа | `rag.py` → Polza.ai | `ChatOpenAI` / `ChatOllama`, `temperature=0.1` | `generation_ms` |
| 6. Фильтр картинок | `rag.py` | Только релевантные (см. ниже) | — |
| 7. Ответ + метрики | `RagAnswer` | `retrieval_ms + generation_ms = total_ms` | `total_ms` |

Итоговое время показывается в чате серой строкой: `⏱ RAG-поиск: X.XXс · генерация ИИ: Y.YYс · всего: Z.ZZс`.

### RAG-конвейер и фильтрация картинок

| Этап | Логика | Детали |
|---|---|---|
| Извлечение текста | `PyMuPDFLoader` и др. | PDF/DOCX/TXT/MD/XLSX/CSV/PPTX/HTML; очистка артефактов (`_clean_pdf_text`) |
| Чанкинг | `RecursiveCharacterTextSplitter` | `chunk_size=1500`, `chunk_overlap=200`, разделители по «Задача»/«Подзадача» |
| Индексация картинок | `gemini_describe.py` | Для каждой картинки PDF — описание через Gemini; вшито в текст чанка |
| MMR-поиск | Chroma retriever | Возвращает `top_k` наиболее релевантных чанков |
| Отсев по порогу | `relevance_threshold` | Если `max_score < threshold` → «Информация не найдена» |
| Отбор картинок | `rag.py` | Регэксп `.jpeg/.jpg/.png`; шум-фильтр (`логотип`, `герб`, …); **только те, что LLM назвал в секции «Картинки:»** (или прошли фильтр релевантности); дедуп по хэшу, мин. размер 3 КБ |

### Панель настроек (админ) — что реально применяется

Поле `PATCH /api/admin/config` **реально меняет работу движка**, а не только значение в памяти:

| Параметр | Эффект | Требует переиндексации? | Сохраняется в `.env`? |
|---|---|---|---|
| `llm_model` | Пересобирает цепочку LLM немедленно | Нет | Да |
| `llm_provider` / `llm_api_base_url` / `llm_api_key` | Переключает провайдера/эндпоинт | Нет | Да |
| `top_k` / `fetch_k` / `relevance_threshold` | Пересобирает ретривер немедленно | Нет | Да |
| `embedding_model` | **Запускает фоновую переиндексацию** (новые векторы) | **Да** | Да |
| `chunk_size` / `chunk_overlap` | **Запускает фоновую переиндексацию** (новые чанки) | **Да** | Да |
| `whisper_model` | Меняет STT-модель | Нет | Да |

Переиндексация при смене эмбеддинга/чанков реализована через полный сброс коллекции Chroma (`clear_system_cache()` + `delete_collection`) — это необходимо, т.к. Chroma не даёт переиспользовать коллекцию с другими настройками эмбеддинга. Статус виден в `GET /api/config` (`reindex_running`, `reindex_last`).

### Фронтенд

| Возможность | Реализация |
|---|---|
| SPA-чат | `web/app.js` (vanilla JS), без сборки |
| Предпросмотр PDF | **PDF.js 3.11.174** (в `web/vendor/pdfjs/`), рендер в `<canvas>` на нужной странице (`page_number` чанка) + навигация по страницам |
| Источники после ответа | Чипы `файл · стр. N` под ответом бота (как в Open WebUI); клик — предпросмотр |
| Тултип оригинального текста | Наведение на синий фрагмент-источник →原始 текст чанка |
| Метрики времени | Серая строка `retrieval_ms / generation_ms / total_ms` под ответом |
| Категории документов | Автоопределение (техкарты, инструкции, регламенты, SOP, нормативы, охрана труда, адаптация) |

### API-эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Проверка живости |
| GET | `/api/config` | Текущая конфигурация (+ статус переиндексации) |
| PATCH | `/api/admin/config` | Обновить настройки (применяются реально) |
| GET | `/api/admin/models` | Список моделей провайдера |
| GET | `/api/admin/index/status` | Статус индекса |
| GET | `/api/admin/index/preview` | План индексации |
| POST | `/api/ask` | Задать вопрос (основной) |
| POST | `/api/admin/reindex` | Ручная переиндексация (`{recreate: true}`) |
| POST | `/api/admin/documents/upload` | Загрузка документа |
| DELETE | `/api/admin/documents/{filename}` | Удаление документа |
| POST | `/api/onboarding/start` · `/message` | Онбординг нового сотрудника |
| GET | `/api/media/documents/{path}` | Отдача файлов (PDF отдаётся с `Content-Disposition: inline`) |
| GET | `/api/media/images/{path}` | Отдача извлечённых картинок |

### Стек технологий

| Слой | Технология |
|---|---|
| Язык | Python 3.10+ |
| Web-фреймворк | FastAPI + Uvicorn |
| RAG | LangChain |
| Векторная БД | ChromaDB 1.5.9 |
| LLM / Embeddings / Vision / STT | Polza.ai (OpenAI-совместимый API) |
| PDF-парсинг | PyMuPDF (`PyMuPDFLoader`) |
| DOCX | `Docx2txtLoader` |
| XLSX | `openpyxl` (`SimpleExcelLoader`) |
| Фронтенд | HTML + CSS + vanilla JS, PDF.js (вендор) |
| Конфиг | `pydantic-settings` (`.env`) |

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
│   ├── rag.py               # RAG-движок (поиск + LLM + фильтр картинок + метрики)
│   ├── ingest.py            # Парсинг/индексация, SimpleExcelLoader, build_embeddings
│   ├── service.py           # Сервис-слой, reindex, применение настроек
│   ├── config.py            # Настройки (pydantic-settings)
│   ├── gemini_describe.py   # Описание картинок из PDF через Vision
│   ├── documents.py         # Загрузка/удаление файлов
│   ├── admin.py             # Статус индекса, список моделей
│   └── prompts.py           # Промпты для LLM
├── web/                     # Фронтенд (SPA)
│   ├── index.html
│   ├── app.js               # Логика чата, предпросмотр PDF.js, метрики, тултипы
│   ├── style.css
│   └── vendor/pdfjs/        # PDF.js 3.11.174 (локально, для встроенного просмотра PDF)
├── data/documents/          # Твои документы
├── chroma_db/               # Векторная БД (авто)
└── scripts/
    ├── setup.sh / setup.ps1 # Скрипты установки
    └── benchmark_llm_latency.py  # Бенчмарк скорости LLM (имитация нескольких файлов)
```
