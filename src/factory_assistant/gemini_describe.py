"""Описание страниц и картинок через Gemini 2.5 (Polza.ai API)."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = """\
Ты — ассистент, анализирующий документы по игре роботов (Лига Технологий).

Проанализируй страницу {page_num} из документа.

Текст страницы:
{page_text}

На странице {image_count} изображений. Опиши КАЖДОЕ изображение кратко (1-2 предложения).
Для каждого изображения укажи:
- Что изображено (объект, детали)
- Ключевые характеристики (размеры, формулы, номера, названия)
- Связь с текстом страницы (если есть)

Файлы изображений (используй ТОЛЬКО эти имена):
{image_list}

Формат — строго построчно:
- [имя_файла]: [описание]

Не добавляй лишнего. Только описания."""


def describe_page(
    client: OpenAI,
    model: str,
    page_text: str,
    image_paths: list[str],
    page_num: int,
    images_dir: Path,
) -> str:
    """Обогатить текст страницы описаниями картинок через Gemini 2.5.

    Args:
        client: OpenAI-совместимый клиент (Polza.ai)
        model: модель для описания (gemini-2.5-flash)
        page_text: текст страницы из PDF
        image_paths: пути к картинкам (относительно images_dir)
        page_num: номер страницы
        images_dir: корневая папка с картинками

    Returns:
        Обогащённый текст: оригинал + описания картинок
    """
    if not image_paths or not page_text.strip():
        return page_text

    # Собираем base64 изображений
    image_contents = []
    valid_paths = []
    for img_path in image_paths:
        full = images_dir / img_path
        if not full.exists():
            continue
        if full.stat().st_size < 500:  # пропускаем мусор
            continue
        try:
            import base64

            data = full.read_bytes()
            b64 = base64.b64encode(data).decode()
            ext = full.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/png")
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
            valid_paths.append(img_path)
        except Exception:
            continue

    if not image_contents:
        return page_text

    image_list = "\n".join(f"- {p}" for p in valid_paths)
    prompt = DESCRIBE_PROMPT.format(
        page_num=page_num,
        page_text=page_text[:3000],
        image_count=len(valid_paths),
        image_list=image_list,
    )

    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        descriptions = response.choices[0].message.content.strip()

        # Формируем обогащённый текст
        enriched = page_text
        if descriptions:
            enriched += f"\n\nИзображения на странице {page_num}:\n{descriptions}"

        return enriched

    except Exception as exc:
        logger.warning("Gemini describe page %d failed: %s", page_num, exc)
        return page_text


def build_gemini_client(api_base: str, api_key: str) -> OpenAI:
    """Создать OpenAI-совместимый клиент для Polza.ai."""
    return OpenAI(
        base_url=api_base.strip().rstrip("/"),
        api_key=api_key or "sk-placeholder",
    )
