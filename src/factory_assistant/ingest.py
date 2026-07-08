from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from factory_assistant.config import Settings, settings

logger = logging.getLogger(__name__)

MANIFEST_FILE = Path("data/ingest_manifest.json")
ENRICHED_CACHE = Path("data/enriched_cache.json")


def _load_manifest() -> dict[str, str]:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def _load_enriched_cache() -> dict[str, str]:
    if ENRICHED_CACHE.exists():
        try:
            return json.loads(ENRICHED_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_enriched_cache(cache: dict[str, str]) -> None:
    ENRICHED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ENRICHED_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

IMAGE_REF_RE = re.compile(
    r"!\s*\[.*?\]\s*\(.*?\)|Рис\.?\s*\d+|Figure\s*\d+|\[image\]|\[рисунок\]",
    re.IGNORECASE,
)

SUPPORTED_TEXT = {".pdf", ".docx", ".txt", ".md", ".xlsx", ".xls", ".csv", ".pptx", ".html"}
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _fix_cyrillic_encoding(text: str) -> str:
    """Fix Cyrillic text encoded as CP1251 but misread as Latin-1."""
    lines = []
    for line in text.split("\n"):
        try:
            fixed = line.encode("latin-1").decode("cp1251")
            lines.append(fixed)
        except (UnicodeEncodeError, UnicodeDecodeError):
            lines.append(line)
    return "\n".join(lines)


def _clean_pdf_text(text: str) -> str:
    text = _fix_cyrillic_encoding(text)
    text = text.replace("/emdash.cyr", "—")
    text = text.replace("/cyr", "")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = IMAGE_REF_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class SimpleExcelLoader:
    """Лёгкий загрузчик Excel на openpyxl (без зависимости от unstructured)."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        from openpyxl import load_workbook

        wb = load_workbook(self.file_path, read_only=True, data_only=True)
        docs: list[Document] = []
        try:
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                rows: list[str] = []
                for row in ws.iter_rows(values_only=True):
                    cells = ["" if c is None else str(c) for c in row]
                    line = "\t".join(cells).strip()
                    if line:
                        rows.append(line)
                text = "\n".join(rows)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"sheet": sheet}))
        finally:
            wb.close()
        return docs


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyMuPDFLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    if suffix in {".xlsx", ".xls"}:
        return SimpleExcelLoader(str(path))
    if suffix == ".csv":
        return CSVLoader(str(path), encoding="utf-8")
    if suffix == ".pptx":
        return UnstructuredPowerPointLoader(str(path))
    if suffix == ".html":
        return UnstructuredHTMLLoader(str(path))
    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("tech_cards", ["техкарт", "техническая карт", "технологическая карт"]),
    ("instructions", ["инструкц", "руководств"]),
    ("regulations", ["регламент", "положени"]),
    ("sop", ["sop", "справоч", "стандарт"]),
    ("normative", ["норматив", "норма", "гост", "снип"]),
    ("safety", ["охрана", "безопас", "промбез"]),
    ("onboarding", ["onboard", "адаптац", "чек-лист", "checklist"]),
    ("games", ["игр", "робот", "соревнован"]),
]


def detect_category(filename: str) -> str:
    lower = filename.lower()
    for cat, keywords in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in lower:
                return cat
    return "other"


def _image_document(path: Path) -> Document:
    meta = {
        "source_file": path.name,
        "source_path": str(path.resolve()),
        "type": "image",
        "page_number": None,
        "category": detect_category(path.name),
    }
    return Document(page_content=f"[Изображение: {path.name}]", metadata=meta)


def _process_pdf_with_images(
    path: Path,
    images_base_dir: Path,
    ocr_cache_dir: Path,
    gemini_client=None,
    gemini_model: str = "",
) -> list[Document]:
    """Загрузить PDF с извлечением картинок и описанием через Gemini 2.5."""
    from .ocr import is_page_scanned, ocr_page
    from .pdf_images import extract_pdf_images

    documents: list[Document] = []

    # Загружаем кэш Gemini описаний
    enriched_cache = _load_enriched_cache()

    # Извлекаем все картинки из PDF
    page_images = extract_pdf_images(path, images_base_dir)

    # Загружаем текстовые страницы через PyMuPDFLoader
    loader = PyMuPDFLoader(str(path))
    raw_docs = loader.load()

    # Папка с картинками этой страницы
    page_images_dir = images_base_dir / path.stem

    for doc in raw_docs:
        page_num = doc.metadata.get("page", 0) + 1  # 1-indexed
        cache_key = f"{path.stem}|{page_num}"
        content = doc.page_content.strip()

        # Проверяем, отсканирована ли страница
        if not content or len(content) < 20:
            if is_page_scanned(str(path), page_num):
                logger.info("OCR страницы %d из %s...", page_num, path.name)
                ocr_text = ocr_page(path, page_num, ocr_cache_dir)
                if ocr_text.strip():
                    content = ocr_text
                    logger.info(
                        "  → OCR: %d символов со стр. %d", len(ocr_text), page_num
                    )

        # Собираем пути к картинкам этой страницы
        image_paths = []
        if page_num in page_images:
            for img_info in page_images[page_num]:
                image_paths.append(img_info["filename"])

        # Обогащаем метаданные
        meta = dict(doc.metadata)
        meta["source_file"] = path.name
        meta["source_path"] = str(path.resolve())
        meta["category"] = detect_category(path.name)
        meta["page_number"] = page_num
        if image_paths:
            meta["image_paths"] = image_paths

        content = _clean_pdf_text(content) if content else content

        # Описываем картинки через Gemini 2.5 (с кэшем)
        if gemini_client and gemini_model and image_paths and content:
            cached = enriched_cache.get(cache_key)
            if cached:
                content = cached
                logger.info("Кэш: стр. %d из %s", page_num, path.name)
            else:
                try:
                    from .gemini_describe import describe_page

                    content = describe_page(
                        client=gemini_client,
                        model=gemini_model,
                        page_text=content,
                        image_paths=image_paths,
                        page_num=page_num,
                        images_dir=page_images_dir,
                    )
                    enriched_cache[cache_key] = content
                    _save_enriched_cache(enriched_cache)
                    logger.info(
                        "Gemini: стр. %d из %s — %d картинок описано",
                        page_num,
                        path.name,
                        len(image_paths),
                    )
                except Exception as exc:
                    logger.warning("Gemini describe failed for page %d: %s", page_num, exc)

        if content:
            documents.append(Document(page_content=content, metadata=meta))
        elif image_paths:
            # Страница только с картинками, без текста — создаём заглушку
            meta["type"] = "image_only"
            documents.append(
                Document(
                    page_content=f"[Страница {page_num} — изображения: {', '.join(image_paths)}]",
                    metadata=meta,
                )
            )

    return documents


def _enrich_metadata(doc: Document, source_path: Path) -> Document:
    meta = dict(doc.metadata)
    meta["source_file"] = source_path.name
    meta["source_path"] = str(source_path.resolve())
    meta["category"] = detect_category(source_path.name)
    if "page" in meta and meta["page"] is not None:
        meta["page_number"] = int(meta["page"]) + 1
    content = (
        _clean_pdf_text(doc.page_content)
        if source_path.suffix.lower() == ".pdf"
        else doc.page_content
    )
    return Document(page_content=content, metadata=meta)


def load_documents(
    documents_dir: Path | None = None,
    files: list[Path] | None = None,
) -> list[Document]:
    root = documents_dir or settings.documents_dir
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Каталог документов не найден: {root}")

    images_base_dir = root.parent / "images"
    ocr_cache_dir = root.parent / "ocr_cache"
    images_base_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)

    # НОВОЕ: Создаём Gemini клиент для описания картинок
    gemini_client = None
    gemini_model = ""
    if settings.llm_api_base_url and settings.llm_api_key:
        try:
            from .gemini_describe import build_gemini_client

            gemini_client = build_gemini_client(
                settings.llm_api_base_url, settings.llm_api_key
            )
            gemini_model = settings.gemini_model
            logger.info("Gemini client создан: %s", gemini_model)
        except Exception as exc:
            logger.warning("Не удалось создать Gemini клиент: %s", exc)

    if files is not None:
        paths = files
    else:
        paths = sorted(root.rglob("*"))

    documents: list[Document] = []
    for path in paths:
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            # PDF: извлечение картинок + OCR + Gemini описания
            pdf_docs = _process_pdf_with_images(
                path, images_base_dir, ocr_cache_dir,
                gemini_client=gemini_client,
                gemini_model=gemini_model,
            )
            documents.extend(pdf_docs)
        elif suffix in SUPPORTED_TEXT:
            loader = _loader_for(path)
            for doc in loader.load():
                documents.append(_enrich_metadata(doc, path))
        elif suffix in SUPPORTED_IMAGES:
            documents.append(_image_document(path))

    logger.info("Загружено %d документов", len(documents))
    return documents


def split_documents(
    documents: list[Document], cfg: Settings | None = None
) -> list[Document]:
    cfg = cfg or settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\nЗадача ", "\nПодзадача ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["content_hash"] = hashlib.md5(
            chunk.page_content.encode()
        ).hexdigest()
    return chunks


class _SafeEmbeddings:
    """Wrapper around OpenAI embeddings with retry on empty response."""

    def __init__(self, base: "OpenAIEmbeddings"):
        self._base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 3:
            return self._base.embed_documents(texts)
        for attempt in range(3):
            try:
                return self._base.embed_documents(texts)
            except ValueError:
                if attempt < 2:
                    mid = len(texts) // 2
                    left = self.embed_documents(texts[:mid])
                    right = self.embed_documents(texts[mid:])
                    return left + right
                raise
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(text)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _build_api_embeddings(cfg: Settings):
    """Build embeddings using raw OpenAI client (bypasses langchain tokenization)."""
    import time
    from openai import OpenAI

    client = OpenAI(
        base_url=cfg.llm_api_base_url.strip().rstrip("/"),
        api_key=cfg.llm_api_key,
    )

    class RawEmbeddings:
        def __init__(self, client, model):
            self._client = client
            self._model = model

        def _embed_with_retry(self, input_data, max_retries=3):
            for attempt in range(max_retries):
                try:
                    resp = self._client.embeddings.create(model=self._model, input=input_data)
                    if not resp.data:
                        raise ValueError("No embedding data received")
                    return resp.data
                except (ValueError, Exception) as e:
                    if attempt < max_retries - 1:
                        wait = 1.5 ** attempt
                        logger.warning("Embedding attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, e, wait)
                        time.sleep(wait)
                    else:
                        raise

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            all_embeds = []
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                data = self._embed_with_retry(batch)
                all_embeds.extend([d.embedding for d in data])
            return all_embeds

        def embed_query(self, text: str) -> list[float]:
            data = self._embed_with_retry([text])
            return data[0].embedding

    return RawEmbeddings(client, cfg.embedding_model)


def build_embeddings(cfg: Settings | None = None):
    cfg = cfg or settings
    # API эмбеддинги через Polza.ai (raw клиент — обходит баг langchain tokenization)
    if cfg.llm_api_base_url and cfg.llm_api_key:
        try:
            return _build_api_embeddings(cfg)
        except Exception as exc:
            logger.warning("API embeddings недоступны, fallback на Ollama: %s", exc)
    # Fallback на Ollama если нет API
    return OllamaEmbeddings(
        model=cfg.embedding_model,
        base_url=cfg.ollama_base_url,
    )


def build_vectorstore(
    cfg: Settings | None = None,
    *,
    recreate: bool = False,
) -> Chroma:
    cfg = cfg or settings
    persist_dir = str(cfg.chroma_dir.resolve())
    embeddings = build_embeddings(cfg)

    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)

    if recreate:
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        try:
            client.delete_collection(cfg.collection_name)
        except Exception:
            pass

    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def ingest_documents(
    documents_dir: Path | None = None,
    cfg: Settings | None = None,
    *,
    recreate: bool = False,
) -> dict:
    cfg = cfg or settings
    root = (documents_dir or cfg.documents_dir).resolve()

    chroma_empty = not cfg.chroma_dir.exists() or not any(cfg.chroma_dir.iterdir())
    if recreate or chroma_empty:
        manifest: dict[str, str] = {}
    else:
        manifest = _load_manifest()

    all_files = sorted(root.rglob("*"))
    changed_files: list[Path] = []
    skipped = 0
    for path in all_files:
        if not path.is_file() or path.name.startswith("."):
            continue
        rel = path.relative_to(root)
        h = _file_hash(path)
        if manifest.get(str(rel)) == h:
            skipped += 1
        else:
            changed_files.append(path)
            manifest[str(rel)] = h

    if not changed_files:
        return {
            "documents": 0,
            "chunks": 0,
            "new_files": 0,
            "skipped": skipped,
            "collection": cfg.collection_name,
            "persist_dir": str(cfg.chroma_dir.resolve()),
        }

    documents = load_documents(documents_dir, files=changed_files)
    vectorstore = build_vectorstore(cfg, recreate=recreate)

    if not documents:
        return {
            "documents": 0,
            "chunks": 0,
            "new_files": len(changed_files),
            "skipped": skipped,
            "collection": cfg.collection_name,
            "persist_dir": str(cfg.chroma_dir.resolve()),
        }

    chunks = split_documents(documents, cfg)
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
    _save_manifest(manifest)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "new_files": len(changed_files),
        "skipped": skipped,
        "collection": cfg.collection_name,
        "persist_dir": str(cfg.chroma_dir.resolve()),
    }
