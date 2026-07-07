"""OCR через Apple Vision (нативный на macOS).

Использует Vision.VNRecognizeTextRequest для распознавания текста
на отсканированных страницах PDF. Работает быстро на Apple Silicon.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .pdf_images import render_page_as_image

logger = logging.getLogger(__name__)

# Проверяем доступность Apple Vision
_VISION_AVAILABLE = False
try:
    import Vision
    import Cocoa
    _VISION_AVAILABLE = True
    logger.info("Apple Vision OCR доступен")
except ImportError:
    logger.info("Apple Vision недоступен, OCR отключён")


def ocr_image(image_path: str | Path, lang: str = "ru-RU") -> str:
    """Распознать текст на изображении через Apple Vision.

    Args:
        image_path: путь к изображению (PNG/JPG)
        lang: язык распознавания (по умолчанию русский)

    Возвращает распознанный текст.
    """
    if not _VISION_AVAILABLE:
        return ""

    try:
        image_path = Path(image_path)
        if not image_path.exists():
            return ""

        # Загружаем изображение
        from Foundation import NSURL

        image_url = NSURL.fileURLWithPath_(str(image_path))
        source = Vision.VNImageSourceHandler.alloc().initWithURL_error_(image_url, None)
        if source is None:
            return ""

        # Создаём запрос на распознавание текста
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        request.setRecognitionLanguages_([lang, "en-US"])

        # Выполняем запрос
        handler = Vision.VNImageRequestHandler.alloc().initWithSource_options_(
            source, {}
        )
        success = handler.performRequests_error_([request], None)

        if not success[0]:
            return ""

        results = request.results()
        if not results:
            return ""

        # Собираем текст
        lines = []
        for observation in results:
            candidate = observation.topCandidates_(1)
            if candidate and len(candidate) > 0:
                text = candidate[0].string()
                if text.strip():
                    lines.append(text.strip())

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Ошибка Apple Vision OCR: %s", exc)
        return ""


def ocr_page(
    pdf_path: str | Path,
    page_number: int,
    cache_dir: str | Path,
    lang: str = "ru-RU",
) -> str:
    """Выполнить OCR для одной страницы PDF.

    1. Рендерит страницу в изображение
    2. Распознаёт текст через Apple Vision
    3. Кэширует результат

    Возвращает распознанный текст.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(pdf_path)
    cache_key = f"{pdf_path.stem}_p{page_number}"
    cache_file = cache_dir / f"{cache_key}.txt"

    # Проверяем кэш
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # Рендерим страницу
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / f"page_{page_number}.png"
        rendered = render_page_as_image(pdf_path, page_number, img_path, dpi=200)

        if not rendered:
            return ""

        # OCR
        text = ocr_image(img_path, lang=lang)

    # Кэшируем
    if text.strip():
        cache_file.write_text(text, encoding="utf-8")

    return text


def has_text_content(text: str, threshold: int = 20) -> bool:
    """Проверяет, есть ли осмысленный текст (не пустой/мусор).

    Args:
        text: текст страницы
        threshold: минимальное количество символов для «нормального» текста

    Возвращает True если текст достаточный.
    """
    if not text:
        return False
    # Убираем пробелы и переносы
    cleaned = text.strip().replace("\n", "").replace(" ", "")
    return len(cleaned) >= threshold


def is_page_scanned(pdf_path: str | Path, page_number: int) -> bool:
    """Проверяет, является ли страница отсканированной (пустой текст).

    Использует PyMuPDF для извлечения текста.
    """
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            if page_number < 1 or page_number > len(doc):
                return True
            page = doc[page_number - 1]
            text = page.get_text("text")
            return not has_text_content(text)
        finally:
            doc.close()
    except Exception:
        return True
