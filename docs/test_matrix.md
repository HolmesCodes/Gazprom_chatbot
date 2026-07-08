# Матрица автоматизированных тестов

| Категория | Группа | Тест | Что проверяет | Критерий | Тип |
|---|---|---|---|---|---|
| **Smoke** | test_smoke.py | health | Сервер отвечает на /health | status=ok | smoke |
| | test_smoke.py | answer_not_empty | Ответ не пустой и есть структура | answer, sources, found_in_kb | smoke |
| | test_smoke.py | answer_structure | Все поля в ответе | answer, sources, retrieval_ms, generation_ms, total_ms | smoke |
| | test_smoke.py | source_structure | Источник содержит source_file, page_number, excerpt | — | smoke |
| | test_smoke.py | empty_question | Пустой вопрос → 422 | 422 Unprocessable | smoke |
| | test_smoke.py | not_found_question | Вопрос вне базы → «не найдена» | found_in_kb=False | smoke |
| | test_smoke.py | answer_has_russian | Ответ на русском | >10 русских букв | smoke |
| | test_smoke.py | all_fields_present | Все поля в ответе | question, answer, sources, found_in_kb, image_descriptions, тайминги | smoke |
| **Скорость** | test_latency.py | response_under_timeout | Полное время ответа | <30s | slow |
| | test_latency.py | generation_not_extreme | Среднее время генерации | <25s | slow |
| | test_latency.py | retrieval_is_fast | Время поиска в ChromaDB | <5s | slow |
| | test_latency.py | consecutive_questions_stable | Разброс времени между 5 вопросами | <15s | slow |
| **Достоверность** | test_factual_accuracy.py | answer_contains_keywords | В ответе есть ожидаемые ключевые слова | Все keywords из эталона | regression |
| | test_factual_accuracy.py | answer_cites_sources | Ответ ссылается на файл или страницу | Имя файла или «стр.» в тексте | regression |
| | test_factual_accuracy.py | no_hallucination_on_invented_query | Нет выдумок на несуществующий вопрос | Нет ложных утверждений | regression |
| | test_factual_accuracy.py | sources_match_excerpt | У каждого источника есть непустой excerpt | >10 символов | regression |
| **Конфигурация** | test_config.py | get_config | Чтение всех полей конфига | llm_model, top_k, fetch_k и т.д. | config |
| | test_config.py | patch_config_top_k | Изменение top_k | Новое значение применяется | config |
| | test_config.py | patch_config_chunk_size | Изменение chunk_size | Новое значение применяется | config |
| | test_config.py | patch_config_full | Массовое изменение 3 полей | Все три применились | config |
| | test_config.py | patch_config_invalid_values | Валидация граничных значений | -1 → 422, 50 → 422 | config |
| | test_config.py | index_status_endpoint | Статус индексации | chroma_ready, documents_count, files | config |
| **Качество поиска** | test_search_quality.py | mrr | Средняя обратная ранговая позиция | MRR > 0.3 | search |
| | test_search_quality.py | recall_at_3 | Доля найденных ключевых слов в top-3 | Recall@3 > 0.4 | search |
| | test_search_quality.py | precision_at_3 | Доля релевантных документов в top-3 | Precision@3 > 0.5 | search |
| | test_search_quality.py | document_coverage | Сколько разных файлов затронуто | >0 | search |
| | test_search_quality.py | consistent_retrieval | Стабильность поиска (2 запуска) | Пересечение >1 | search |
| | test_search_quality.py | retrieval_time | Время поиска | <3s | search |
| **Регресс** | test_regression.py | all_basic_questions_return_ok | Все 15 вопросов → 200 OK | 0 HTTP ошибок | regression |
| | test_regression.py | all_questions_have_valid_structure | Структура ответа у всех вопросов | answer=str, sources=list | regression |
| | test_regression.py | total_time_all_questions | Ни один вопрос не превысил 25s | Все total_ms < 25000 | regression |
| | test_regression.py | config_after_questions | Конфиг не сломался после вопросов | top_k=5, fetch_k=15 | regression |

---

## Запуск

```bash
# Быстрый прогон (smoke + config, без медленных и поиска)
pytest tests/ -m "not slow and not search" -v

# Полный прогон (все категории)
pytest tests/ -v

# Только регресс
pytest tests/ -m regression -v

# Только скорость
pytest tests/ -m slow -v

# Только поиск
pytest tests/ -m search -v

# Только smoke
pytest tests/ -m smoke -v
```

## Структура файлов

```
tests/
├── conftest.py              # Фикстуры: TestClient, Settings, 15 вопросов
├── pytest.ini               # Маркеры slow, smoke, search, config, regression, chroma
├── test_smoke.py            # Базовые проверки (8 тестов)
├── test_latency.py          # Бенчмарк скорости (4 теста, маркер slow)
├── test_factual_accuracy.py # Достоверность и источники (4 теста)
├── test_config.py           # CRUD конфигурации (7 тестов)
├── test_search_quality.py   # MRR, Precision@3, Recall@3 (6 тестов)
└── test_regression.py       # Полный прогон (4 теста)
```

## Зависимости

```bash
pip install pytest httpx
```
