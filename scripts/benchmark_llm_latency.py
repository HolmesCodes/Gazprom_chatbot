"""Бенчмарк латентности LLM при генерации ответа с «несколькими прикреплёнными файлами».

Сценарий имитирует RAG-выдачу бота: в контексте несколько извлечённых
кусков документов (имитация прикреплённых файлов), модель должна ответить
на вопрос. Для каждой модели из списка делается один запрос, замеряется
полное время генерации ответа.

Запуск:
    python scripts/benchmark_llm_latency.py
    python scripts/benchmark_llm_latency.py --models deepseek/deepseek-v4-flash google/gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


# Контекст «несколько прикреплённых файлов» — имитация RAG (6 кусков документа).
ATTACHED_FILES_CONTEXT = """\
Ниже приведены извлечённые фрагменты из внутренней документации предприятия
(каждый блок — отдельный прикреплённый файл/страница):

[файл 1, стр. 2] Технологическая карта: подготовка емкости к подъему.
Перед строповкой проверить отсутствие остатков продукта, заземлить ёмкость,
установить оттяжки. Стропы канатные (СК) или цепные (СЦ) грузоподъёмностью
не менее 4 т. Угол между ветвями стропа не более 90°.

[файл 2, стр. 5] Инструкция по охране труда: работа на высоте.
Применять страховочную привязь, закреплять на анкерных линиях. Запрещается
находиться под грузом. Работы проводить по наряду-допуску.

[файл 3, стр. 1] Регламент ТО оборудования: перечень узлов.
Насосный блок, трубопроводы, запорная арматура, КИПиА. Периодичность
обслуживания: ежесменно, еженедельно, ежемесячно.

[файл 4, стр. 3] Положение о промышленной безопасности.
Ответственность за нарушение требований несёт руководитель участка.
Плановые аудиты раз в квартал. Расследование инцидентов в течение 72 часов.

[файл 5, стр. 7] Справочник материалов: марки стали.
Сталь 09Г2С — для сварных конструкций при -40°C. Сталь 12Х18Н10Т —
нержавеющая, пищевая и химическая отрасль.

[файл 6, стр. 2] Чек-лист адаптации нового сотрудника.
Изучение инструкций, назначение наставника, прохождение инструктажа,
допуск к самостоятельной работе после аттестации.
"""

QUESTION = (
    "Кратко ответь по существу: в чём польза внедрения ИИ на производстве "
    "и какие сейчас существуют быстрые LLM? Опирайся на прикреплённые файлы."
)

SYSTEM_PROMPT = (
    "Ты — ИИ-помощник специалиста на производственном предприятии. "
    "Отвечай кратко, по делу, на русском языке, только на основе контекста."
)

# Кандидаты: быстрые chat-модели на Polza.ai (flash/lite/mini/nano/haiku/micro).
DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "amazon/nova-micro-v1",
    "amazon/nova-lite-v1",
    "bytedance-seed/seed-2.0-mini",
    "bytedance-seed/seed-2.0-lite",
    "bytedance-seed/seed-1.6-flash",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3-haiku",
    "openai/gpt-5-nano",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o-mini",
    "qwen/qwen3.6-flash",
    "z-ai/glm-4.7-flash",
    "mistralai/ministral-3b-2512",
]


def run_one(base_url: str, api_key: str, model: str, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ATTACHED_FILES_CONTEXT + "\n\nВопрос: " + QUESTION,
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - start
        if resp.status_code != 200:
            return {
                "model": model,
                "ok": False,
                "status": resp.status_code,
                "seconds": round(elapsed, 2),
                "error": resp.text[:300],
                "chars": 0,
            }
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "model": model,
            "ok": True,
            "status": 200,
            "seconds": round(elapsed, 2),
            "chars": len(content),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return {
            "model": model,
            "ok": False,
            "status": 0,
            "seconds": round(elapsed, 2),
            "error": str(exc)[:300],
            "chars": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк латентности LLM (multi-file context)")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="Список моделей для прогона")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Таймаут на один запрос, сек")
    parser.add_argument("--runs", type=int, default=1,
                        help="Сколько прогонов на модель (для усреднения)")
    parser.add_argument("--out", default="scripts/benchmark_results.json",
                        help="Куда сохранить результаты")
    args = parser.parse_args()

    load_env(ENV_PATH)
    base_url = os.environ.get("LLM_API_BASE_URL", "https://polza.ai/api/v1")
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise SystemExit("LLM_API_KEY не найден ни в .env, ни в окружении")

    print(f"Базовый URL: {base_url}")
    print(f"Моделей: {len(args.models)}, прогонов на модель: {args.runs}, таймаут: {args.timeout}с")
    print(f"Контекст: {len(ATTACHED_FILES_CONTEXT)} символов (имитация нескольких файлов)\n")

    aggregated: list[dict] = []
    for model in args.models:
        times: list[float] = []
        last: dict = {}
        for run in range(args.runs):
            res = run_one(base_url, api_key, model, args.timeout)
            last = res
            if res["ok"]:
                times.append(res["seconds"])
            status = "OK" if res["ok"] else f"ERR({res.get('status')})"
            print(f"  {model:<40} {status:<8} {res['seconds']}с  chars={res.get('chars', 0)}")
        entry = {
            "model": model,
            "ok": bool(times),
            "runs": args.runs,
            "min_s": round(min(times), 2) if times else None,
            "mean_s": round(statistics.mean(times), 2) if times else None,
            "max_s": round(max(times), 2) if times else None,
            "chars": last.get("chars"),
            "error": last.get("error"),
        }
        aggregated.append(entry)

    succeeded = [e for e in aggregated if e["ok"]]
    succeeded.sort(key=lambda e: e["mean_s"])
    ranking = succeeded[:15]

    print("\n=== ТОП по скорости ответа (время, сек) ===")
    print(f"{'#':<3} {'model':<40} {'mean':<7} {'min':<7} {'max':<7} {'chars':<6}")
    for i, e in enumerate(ranking, 1):
        print(f"{i:<3} {e['model']:<40} {str(e['mean_s']):<7} {str(e['min_s']):<7} "
              f"{str(e['max_s']):<7} {str(e['chars']):<6}")

    failed = [e for e in aggregated if not e["ok"]]
    if failed:
        print("\n=== Не прошли / ошибки ===")
        for e in failed:
            print(f"  {e['model']:<40} {e.get('error', '')[:120]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"ranking": ranking, "all": aggregated,
             "context_chars": len(ATTACHED_FILES_CONTEXT)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
