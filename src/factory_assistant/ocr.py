"""OCR через Polza.ai Vision API (gemini-2.5-flash-lite).

Не требует системных зависимостей (Tesseract, Apple Vision).
Работает на любой платформе через API.
"""

from __future__ import annotations

import base64
import logging
import re
import tempfile
from pathlib import Path

import requests

from factory_assistant.config import settings
from factory_assistant.pdf_images import render_page_as_image

logger = logging.getLogger(__name__)

OCR_CACHE_DIR = Path("data/ocr_cache")
VISION_MODEL = "google/gemini-2.5-flash-lite"

# Буквы (Cyrillic + Latin)
ALPHA_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]")


def _is_garbled_text(text: str, threshold: float = 0.15) -> bool:
    """Определяет, является ли текст «битым» по доле буквенных символов.

    Если доля букв (Cyrillic + Latin) от общего числа не-пробельных символов
    ниже threshold — текст считается битым (PDF с кривым ToUnicode CMap).
    """
    if not text.strip():
        return True
    non_space = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(non_space) < 10:
        return True

    alpha_chars = len(ALPHA_RE.findall(non_space))
    ratio = alpha_chars / max(len(non_space), 1)
    logger.debug(
        "garbled check: alpha=%d/%d, ratio=%.2f, threshold=%.2f",
        alpha_chars, len(non_space), ratio, threshold,
    )
    return ratio < threshold


def _ocr_via_vision_api(
    image_path: Path,
    api_key: str = "",
    api_base: str = "https://polza.ai/api/v1",
) -> str:
    """Распознать текст на изображении через Polza.ai Vision API."""
    if not image_path.exists():
        return ""

    try:
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Распознай и верни весь текст на этом изображении. "
                                    "Сохрани структуру: абзацы, списки, заголовки. "
                                    "Отвечай ТОЛЬКО распознанным текстом, без комментариев."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.05,
                "max_tokens": 4096,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.warning("Vision API error %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Vision API request failed: %s", exc)
    return ""


def ocr_page(
    pdf_path: str | Path,
    page_number: int,
    cache_dir: str | Path | None = None,
) -> str:
    """Выполнить OCR для одной страницы PDF через Vision API.

    1. Рендерит страницу в изображение
    2. Отправляет на распознавание через Polza.ai Vision
    3. Кэширует результат

    Возвращает распознанный текст или пустую строку.
    """
    cache_dir = Path(cache_dir) if cache_dir else OCR_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(pdf_path)
    cache_key = f"{pdf_path.stem}_p{page_number}"
    cache_file = cache_dir / f"{cache_key}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / f"page_{page_number}.png"
        rendered = render_page_as_image(pdf_path, page_number, img_path, dpi=200)
        if not rendered:
            return ""

        text = _ocr_via_vision_api(
            Path(rendered),
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base_url,
        )

    if text.strip():
        cache_file.write_text(text, encoding="utf-8")
        logger.info("OCR page %d: %d chars", page_number, len(text))

    return text


def has_text_content(text: str, threshold: int = 20) -> bool:
    if not text:
        return False
    cleaned = text.strip().replace("\n", "").replace(" ", "")
    return len(cleaned) >= threshold


def is_page_scanned(pdf_path: str | Path, page_number: int) -> bool:
    """Проверяет, является ли страница отсканированной (пустой текст или битый)."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            if page_number < 1 or page_number > len(doc):
                return True
            page = doc[page_number - 1]
            text = page.get_text("text")
            return not has_text_content(text) or _is_garbled_text(text)
        finally:
            doc.close()
    except Exception:
        return True
