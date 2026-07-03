from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from factory_assistant.config import Settings, settings

# Регулярка для удаления ссылок на изображения в тексте PDF
IMAGE_REF_RE = re.compile(r'!\s*\[.*?\]\s*\(.*?\)|Рис\.?\s*\d+|Figure\s*\d+|\[image\]|\[рисунок\]', re.IGNORECASE)


def _clean_pdf_text(text: str) -> str:
    text = text.replace("/emdash.cyr", "—")
    text = text.replace("/cyr", "")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = IMAGE_REF_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyMuPDFLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


def _enrich_metadata(doc: Document, source_path: Path) -> Document:
    meta = dict(doc.metadata)
    meta["source_file"] = source_path.name
    meta["source_path"] = str(source_path.resolve())
    if "page" in meta and meta["page"] is not None:
        meta["page_number"] = int(meta["page"]) + 1
    content = _clean_pdf_text(doc.page_content) if source_path.suffix.lower() == ".pdf" else doc.page_content
    return Document(page_content=content, metadata=meta)


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    root = documents_dir or settings.documents_dir
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Каталог документов не найден: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            continue
        loader = _loader_for(path)
        for doc in loader.load():
            documents.append(_enrich_metadata(doc, path))
    return documents


def split_documents(documents: list[Document], cfg: Settings | None = None) -> list[Document]:
    cfg = cfg or settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\nЗадача ", "\nПодзадача ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["content_hash"] = hashlib.md5(chunk.page_content.encode()).hexdigest()
    return chunks


def build_embeddings(cfg: Settings | None = None) -> OllamaEmbeddings:
    cfg = cfg or settings
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
    documents = load_documents(documents_dir)
    vectorstore = build_vectorstore(cfg, recreate=recreate)

    if not documents:
        return {
            "documents": 0,
            "chunks": 0,
            "collection": cfg.collection_name,
            "persist_dir": str(cfg.chroma_dir.resolve()),
        }

    chunks = split_documents(documents, cfg)
    vectorstore.add_documents(chunks)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "collection": cfg.collection_name,
        "persist_dir": str(cfg.chroma_dir.resolve()),
    }
