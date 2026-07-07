from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from factory_assistant.config import Settings, settings
from factory_assistant.ingest import build_vectorstore, detect_category, load_documents, split_documents

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".xlsx", ".xls", ".csv", ".pptx", ".html", ".htm", ".jpg", ".jpeg", ".png", ".gif", ".webp"}


def list_ollama_models(base_url: str | None = None) -> list[dict]:
    base_url = (base_url or settings.ollama_base_url).rstrip("/")
    try:
        with urlopen(f"{base_url}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode())
    except (URLError, TimeoutError, json.JSONDecodeError):
        return []

    models = []
    for item in payload.get("models", []):
        name = item.get("name", "")
        size_bytes = item.get("size", 0)
        models.append(
            {
                "name": name,
                "size_gb": round(size_bytes / (1024**3), 2) if size_bytes else None,
                "modified_at": item.get("modified_at"),
            }
        )
    return sorted(models, key=lambda m: m["name"])


def list_document_files(documents_dir: Path | None = None) -> list[dict]:
    root = (documents_dir or settings.documents_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "relative_path": str(path.relative_to(root)),
                "extension": path.suffix.lower(),
                "category": detect_category(path.name),
                "size_bytes": stat.st_size,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_at": stat.st_mtime,
            }
        )
    return files


def preview_index_plan(cfg: Settings | None = None) -> dict:
    cfg = cfg or settings
    documents = load_documents(cfg.documents_dir)
    chunks = split_documents(documents, cfg) if documents else []

    by_file: dict[str, int] = {}
    for chunk in chunks:
        name = chunk.metadata.get("source_file", "unknown")
        by_file[name] = by_file.get(name, 0) + 1

    return {
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "files": [
            {
                "name": item["name"],
                "size_kb": item["size_kb"],
                "planned_chunks": by_file.get(item["name"], 0),
            }
            for item in list_document_files(cfg.documents_dir)
        ],
    }


def _count_indexed_chunks(cfg: Settings) -> tuple[int, dict[str, int], bool]:
    try:
        vectorstore = build_vectorstore(cfg, recreate=False)
        collection = vectorstore._collection
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        indexed_by_file: dict[str, int] = {}
        for meta in metadatas:
            if not meta:
                continue
            name = meta.get("source_file", "unknown")
            indexed_by_file[name] = indexed_by_file.get(name, 0) + 1
        return len(metadatas), indexed_by_file, len(metadatas) > 0
    except Exception:
        return 0, {}, False


def get_index_status(cfg: Settings | None = None) -> dict:
    cfg = cfg or settings
    files = list_document_files(cfg.documents_dir)
    plan = preview_index_plan(cfg)

    persist_dir = cfg.chroma_dir.resolve()
    if persist_dir.exists() and any(persist_dir.iterdir()):
        indexed_chunks, indexed_by_file, chroma_ready = _count_indexed_chunks(cfg)
    else:
        indexed_chunks, indexed_by_file, chroma_ready = 0, {}, False

    file_rows = []
    for item in files:
        file_rows.append(
            {
                **item,
                "indexed_chunks": indexed_by_file.get(item["name"], 0),
                "planned_chunks": next(
                    (f["planned_chunks"] for f in plan["files"] if f["name"] == item["name"]),
                    0,
                ),
            }
        )

    return {
        "chroma_ready": chroma_ready,
        "documents_count": len(files),
        "indexed_chunks": indexed_chunks,
        "planned_chunks": plan["chunks_count"],
        "collection_name": cfg.collection_name,
        "chroma_dir": str(persist_dir),
        "documents_dir": str(cfg.documents_dir.resolve()),
        "embedding_model": cfg.embedding_model,
        "index_settings": {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
            "fetch_k": cfg.fetch_k,
            "relevance_threshold": cfg.relevance_threshold,
        },
        "files": file_rows,
    }
