"""Извлечение встроенных изображений из PDF через PyMuPDF."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Минимальный размер картинки (пикселей) — игнорируем иконки, логотипы
MIN_IMAGE_SIZE = 80


def _image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def extract_page_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    page_number: int,
) -> list[dict]:
    """Извлечь все изображения со страницы PDF.

    Возвращает список dict:
        { "path": str, "width": int, "height": int, "xref": int }
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("Не удалось открыть PDF %s: %s", pdf_path, exc)
        return results

    try:
        if page_number < 1 or page_number > len(doc):
            return results

        page = doc[page_number - 1]  # 0-indexed
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image.get("image")
            if not image_bytes:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Пропускаем мелкие картинки (иконки, логотипы)
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                continue

            # Определяем расширение
            ext = base_image.get("ext", "png")
            if ext not in ("png", "jpg", "jpeg", "webp"):
                ext = "png"

            img_hash = _image_hash(image_bytes)
            filename = f"p{page_number}_img{img_index}_{img_hash}.{ext}"
            filepath = output_dir / filename

            # Не перезаписываем если уже есть
            if not filepath.exists():
                try:
                    # Конвертируем через Pillow для стабильного формата
                    import io

                    pil_img = Image.open(io.BytesIO(image_bytes))
                    pil_img.save(str(filepath), quality=95)
                except Exception:
                    # Fallback: записываем сырые байты
                    filepath.write_bytes(image_bytes)

            results.append(
                {
                    "path": str(filepath),
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "xref": xref,
                }
            )
    finally:
        doc.close()

    return results


def extract_pdf_images(
    pdf_path: str | Path,
    images_base_dir: str | Path,
) -> dict[int, list[dict]]:
    """Извлечь все изображения из всего PDF.

    Возвращает dict: { page_number: [image_info, ...] }
    """
    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    output_dir = Path(images_base_dir) / stem

    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
    except Exception as exc:
        logger.warning("Не удалось открыть PDF %s: %s", pdf_path, exc)
        return {}
    finally:
        if doc:
            doc.close()

    result: dict[int, list[dict]] = {}
    for page_num in range(1, page_count + 1):
        images = extract_page_images(pdf_path, output_dir, page_num)
        if images:
            result[page_num] = images

    total = sum(len(imgs) for imgs in result.values())
    logger.info("Извлечено %d изображений из %s (%d стр.)", total, pdf_path.name, page_count)
    return result


def render_page_as_image(
    pdf_path: str | Path,
    page_number: int,
    output_path: str | Path,
    dpi: int = 200,
) -> str | None:
    """Отрисовать страницу PDF как изображение (для OCR).

    Возвращает путь к сохранённому PNG или None.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None

    try:
        if page_number < 1 or page_number > len(doc):
            return None

        page = doc[page_number - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_path))
        return str(output_path)
    except Exception as exc:
        logger.warning("Ошибка рендера страницы %d: %s", page_number, exc)
        return None
    finally:
        doc.close()
