# ИИ-Помощник специалиста на заводе

RAG-бот для ответов на вопросы по производственной документации.
Отвечает на русском языке с указанием источников.

## Быстрый старт

### 1. Установка зависимостей

**Mac / Linux:**
```bash
git clone https://github.com/HolmesCodes/Gazprom_chatbot.git
cd Gazprom_chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows:**
```cmd
git clone https://github.com/HolmesCodes/Gazprom_chatbot.git
cd Gazprom_chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. Установка Ollama (только для эмбеддингов)

Ollama нужен только для генерации эмбеддингов (~200MB). LLM и распознавание речи работают через API.

**Mac:**
```bash
brew install ollama
ollama serve &
ollama pull nomic-embed-text
```

**Windows:**
1. Скачай с https://ollama.com/download
2. Установи и запусти
3. В терминале выполни:
```cmd
ollama pull nomic-embed-text
```

### 3. Настройка API-ключа

```bash
cp .env.example .env
```

Отредактируй `.env` — впиши свой API-ключ от Polza.ai:

```
LLM_API_KEY=pza_твой_ключ_здесь
```

Ключ получаешь на https://polza.ai после регистрации.

### 4. Индексация документов

Положи документы (PDF, DOCX, TXT, MD, XLSX, CSV, PPTX, HTML) в папку `data/documents/`.

```bash
python -m factory_assistant.cli ingest
```

### 5. Запуск сервера

```bash
python -m factory_assistant.cli serve
```

Открой в браузере: http://127.0.0.1:8000

---

## Команды

| Команда | Описание |
|---|---|
| `python -m factory_assistant.cli serve` | Запуск веб-сервера |
| `python -m factory_assistant.cli ingest` | Индексация документов |
| `python -m factory_assistant.cli ask "вопрос"` | Вопрос из терминала |

---

## Что поддерживается

- **Документы**: PDF, DOCX, TXT, MD, XLSX, CSV, PPTX, HTML, изображения (JPG, PNG)
- **Язык**: русский
- **LLM**: deepseek-v4-flash через Polza.ai (бесплатная)
- **STT**: whisper-1 через Polza.ai (микрофон в браузере)
- **Эмбеддинги**: nomic-embed-text через Ollama (локально)
- **Векторная БД**: ChromaDB

---

## Админ-панель

После запуска сервера открой http://127.0.0.1:8000 и нажми на иконку шестерёнки.

Ты можешь:
- Менять LLM модель и провайдера
- Настроить параметры поиска (top_k, chunk_size)
- Загружать/удалять документы
- Смотреть статус индексации
