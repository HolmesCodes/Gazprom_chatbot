from __future__ import annotations

from pathlib import Path

from factory_assistant.admin import SUPPORTED_EXTENSIONS
from factory_assistant.config import Settings, settings


def save_upload(filename: str, content: bytes, documents_dir: Path | None = None) -> dict:
    root = (documents_dir or settings.documents_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith("."):
        raise ValueError("Некорректное имя файла.")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Формат не поддерживается: {suffix}")

    target = root / safe_name
    target.write_bytes(content)
    stat = target.stat()
    return {
        "name": safe_name,
        "size_bytes": stat.st_size,
        "path": str(target),
    }


def delete_document(filename: str, documents_dir: Path | None = None) -> dict:
    root = (documents_dir or settings.documents_dir).resolve()
    target = (root / Path(filename).name).resolve()

    if root not in target.parents and target != root:
        raise ValueError("Недопустимый путь к файлу.")
    if not target.exists():
        raise FileNotFoundError(f"Файл не найден: {filename}")

    target.unlink()
    return {"deleted": target.name}
