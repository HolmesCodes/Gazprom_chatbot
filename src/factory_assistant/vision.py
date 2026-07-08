from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_FILE = Path("data/vision_cache.json")
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def describe_image(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    api_base: str = "https://polza.ai/api/v1",
    prompt: str = "Опиши это изображение подробно как технический специалист. Укажи: тип изображения (чертёж/схема/фото/таблица), что изображено, ключевые размеры, номера, маркировку, стандарты. Отвечай на русском, 3-5 предложений.",
) -> str | None:
    image_path = Path(image_path)
    if not image_path.exists():
        return None

    cache_key = str(image_path.resolve())
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    try:
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            desc = resp.json()["choices"][0]["message"]["content"].strip()
            if desc:
                cache[cache_key] = desc
                _save_cache(cache)
                return desc
    except Exception:
        logger.debug("Vision failed for %s", image_path, exc_info=True)
    return None


def describe_images(
    image_paths: list[str],
    source_file: str = "",
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    api_base: str = "https://polza.ai/api/v1",
    base_dir: str = "data/images",
) -> dict[str, str]:
    results: dict[str, str] = {}
    base = Path(base_dir)
    if source_file:
        pdf_stem = Path(source_file).stem
        img_base = base / pdf_stem
    else:
        img_base = base
    for rel_path in image_paths:
        full = img_base / rel_path
        if not full.exists():
            full = base / rel_path
        desc = describe_image(full, model=model, api_key=api_key, api_base=api_base)
        if desc:
            results[rel_path] = desc
    return results
