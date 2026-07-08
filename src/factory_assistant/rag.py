from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from factory_assistant.config import Settings, settings
from factory_assistant.ingest import build_vectorstore
from factory_assistant.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "Информация не найдена в базе знаний"

NOISE_KEYWORDS = {
    "логотип", "логотипы", "emblem", "logo", "brand", "шапка", "герб",
    "пуст", "бел", "фон", "заглуш", "placeholder", "пустое",
}

_image_cache: dict[str, tuple[bool, int]] = {}


@dataclass
class SourceReference:
    source_file: str
    page_number: int | None = None
    chunk_id: int | None = None
    excerpt: str = ""
    image_paths: list[str] = field(default_factory=list)
    image_descriptions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "image_paths": self.image_paths,
            "image_descriptions": self.image_descriptions,
        }


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    found_in_kb: bool = True
    image_descriptions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "found_in_kb": self.found_in_kb,
            "image_descriptions": self.image_descriptions,
        }


def _extract_image_descriptions(docs: list[Document]) -> dict[str, str]:
    descriptions = {}
    for doc in docs:
        text = doc.page_content
        for m in re.finditer(r"(image_?\d+\.png|p\d+_img\d+_[a-f0-9]+\.png):\s*(.+)", text):
            img_name = m.group(1)
            desc = m.group(2).strip()
            if img_name not in descriptions:
                descriptions[img_name] = desc
    return descriptions


def _is_noise_image(desc: str) -> bool:
    lower = desc.lower()
    return any(kw in lower for kw in NOISE_KEYWORDS)


def _check_image(full: Path) -> tuple[bool, int]:
    """Check if image exists and get size (cached)."""
    key = str(full)
    if key in _image_cache:
        return _image_cache[key]
    exists = full.exists()
    size = full.stat().st_size if exists else 0
    _image_cache[key] = (exists, size)
    return (exists, size)


def _is_relevant_image(desc: str, question: str, answer: str) -> bool:
    if _is_noise_image(desc):
        return False
    if len(desc) < 20:
        return False
    combined = f"{question} {answer}".lower()
    desc_lower = desc.lower()
    words = set(re.findall(r"[а-яёa-z]{4,}", combined))
    desc_words = set(re.findall(r"[а-яёa-z]{4,}", desc_lower))
    overlap = words & desc_words
    return len(overlap) >= 2


def format_source(doc: Document) -> SourceReference:
    meta = doc.metadata
    image_paths = meta.get("image_paths", [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    return SourceReference(
        source_file=meta.get("source_file", meta.get("source", "unknown")),
        page_number=meta.get("page_number"),
        chunk_id=meta.get("chunk_id"),
        excerpt=doc.page_content[:240].strip(),
        image_paths=image_paths,
    )


def format_docs(docs: list[Document]) -> str:
    blocks: list[str] = []
    for doc in docs:
        src = format_source(doc)
        header = f"[{src.source_file}"
        if src.page_number:
            header += f", стр. {src.page_number}"
        header += "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


class RagEngine:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.vectorstore: Chroma | None = None
        self.retriever = None
        self.llm = self._create_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_USER_TEMPLATE),
            ]
        )
        self.chain = None
        self._try_load_vectorstore()

    def _create_llm(self):
        if self.cfg.llm_provider == "openai" and self.cfg.llm_api_base_url:
            base_url = self.cfg.llm_api_base_url.strip().rstrip("/")
            return ChatOpenAI(
                model=self.cfg.llm_model,
                base_url=base_url,
                api_key=self.cfg.llm_api_key or "sk-placeholder",
                temperature=0.1,
                max_tokens=2048,
            )
        return ChatOllama(
            model=self.cfg.llm_model,
            base_url=self.cfg.ollama_base_url,
            temperature=0.1,
            num_ctx=8192,
            keep_alive="24h",
        )

    def _try_load_vectorstore(self) -> bool:
        persist_dir = self.cfg.chroma_dir.resolve()
        if not persist_dir.exists() or not any(persist_dir.iterdir()):
            return False
        try:
            self.vectorstore = build_vectorstore(self.cfg, recreate=False)
            self._rebuild_retriever()
            self._rebuild_chain()
            return True
        except Exception:
            self.vectorstore = None
            return False

    def _retrieve(self, question: str) -> list[Document]:
        if self.vectorstore is None:
            return []
        for attempt in range(3):
            try:
                return self.retriever.invoke(question)
            except Exception as exc:
                if attempt < 2:
                    logger.warning("Retrieve attempt %d failed: %s. Retrying...", attempt + 1, exc)
                    import time
                    time.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning("All retrieve attempts failed: %s", exc)
                    return []
        return []

    def _build_retriever(self, vectorstore: Chroma):
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.cfg.top_k, "fetch_k": 20},
        )

    def ask(self, question: str) -> RagAnswer:
        question = question.strip()
        if not question:
            return RagAnswer(
                question=question,
                answer="Задайте вопрос по производственной документации.",
                sources=[],
                found_in_kb=False,
            )

        if self.vectorstore is None:
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

        docs = self._retrieve(question)

        if not docs:
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

        try:
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question,
                k=self.cfg.top_k,
            )
            max_score = scored[0][1] if scored else 0.0
            best_page = scored[0][0].metadata.get("page_number") if scored else None
            best_file = scored[0][0].metadata.get("source_file") if scored else None
        except Exception:
            max_score = 0.0
            best_page = None
            best_file = None

        if max_score < self.cfg.relevance_threshold:
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

        sources = [format_source(doc) for doc in docs]
        all_image_descriptions = _extract_image_descriptions(docs)

        context = format_docs(docs)
        prompt_msg = self.prompt.invoke(
            {"context": context, "question": question}
        )
        answer = self.llm.invoke(prompt_msg).content

        if NOT_FOUND_ANSWER.lower() in answer.lower():
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

        cited_pages: set[int] = set()
        for m in re.finditer(r"стр\.?\s*(\d+)", answer):
            cited_pages.add(int(m.group(1)))

        if cited_pages:
            filtered = [s for s in sources if s.page_number in cited_pages]
            if filtered:
                sources = filtered

        for src in sources:
            if best_file and best_page:
                if src.source_file != best_file or src.page_number != best_page:
                    src.image_paths = []

        mentioned_images: set[str] = set()
        for m in re.finditer(r"(p\d+_img\d+_[a-f0-9]+\.png|image_\d+\.png)", answer):
            mentioned_images.add(m.group(1))

        cited_files = {s.source_file for s in sources}
        cited_pages_int = set(int(m.group(1)) for m in re.finditer(r"стр\.?\s*(\d+)", answer))

        image_sources = [
            format_source(doc) for doc in docs
            if doc.metadata.get("source_file") in cited_files
            and doc.metadata.get("page_number") in cited_pages_int
        ]
        if not image_sources and best_file and best_page:
            image_sources = [
                format_source(doc) for doc in docs
                if doc.metadata.get("source_file") == best_file
                and doc.metadata.get("page_number") == best_page
            ]

        seen_hashes: dict[str, str] = {}
        min_size = 3000

        merged_images: dict[tuple[str, int | None], dict[str, list[str]]] = {}
        for src in image_sources:
            key = (src.source_file, src.page_number)
            if key not in merged_images:
                merged_images[key] = {"paths": [], "descs": {}}
            merged_images[key]["paths"].extend(src.image_paths)
            merged_images[key]["descs"].update(src.image_descriptions)

        for (src_file, src_page), img_data in merged_images.items():
            unique: list[str] = []
            for img_path in img_data["paths"]:
                full = (
                    Path("data/images")
                    / Path(src_file).stem
                    / img_path
                )
                exists, size = _check_image(full)
                if not exists:
                    continue
                if size < min_size:
                    continue

                desc = all_image_descriptions.get(img_path, "")
                if not _is_relevant_image(desc, question, answer):
                    logger.info("Image filtered (relevance): %s", img_path)
                    continue
                if mentioned_images and img_path not in mentioned_images:
                    logger.info("Image filtered (not mentioned): %s", img_path)
                    continue

                h = hashlib.md5(full.read_bytes()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes[h] = img_path
                unique.append(img_path)
            if unique:
                existing = next(
                    (s for s in sources if s.source_file == src_file and s.page_number == src_page),
                    None,
                )
                if existing:
                    existing.image_paths = unique
                    existing.image_descriptions = {
                        img: all_image_descriptions.get(img, "")
                        for img in unique
                    }
                else:
                    sources.append(SourceReference(
                        source_file=src_file,
                        page_number=src_page,
                        image_paths=unique,
                        image_descriptions={
                            img: all_image_descriptions.get(img, "")
                            for img in unique
                        },
                    ))

        flat_descriptions = {}
        for src in sources:
            for img, desc in src.image_descriptions.items():
                flat_descriptions[img] = desc

        return RagAnswer(
            question=question,
            answer=answer,
            sources=sources,
            found_in_kb=True,
            image_descriptions=flat_descriptions,
        )

    def reload_vectorstore(self) -> bool:
        old = self.vectorstore
        self.vectorstore = None
        self.retriever = None
        self.chain = None

        ok = self._try_load_vectorstore()
        if not ok:
            self.vectorstore = old
            if old is not None:
                self._rebuild_retriever()
                self._rebuild_chain()
        return ok

    def update_llm_model(self, model: str) -> None:
        self.cfg.llm_model = model
        self.llm = self._create_llm()
        self._rebuild_chain()

    def update_provider(self, provider: str, api_base: str = "", api_key: str = "") -> None:
        self.cfg.llm_provider = provider
        if api_base:
            self.cfg.llm_api_base_url = api_base.strip().rstrip("/")
        if api_key:
            self.cfg.llm_api_key = api_key
        self.llm = self._create_llm()
        self._rebuild_chain()

    def update_retrieval_settings(
        self,
        *,
        top_k: int | None = None,
        fetch_k: int | None = None,
        relevance_threshold: float | None = None,
    ) -> None:
        if top_k is not None:
            self.cfg.top_k = top_k
        if fetch_k is not None:
            self.cfg.fetch_k = fetch_k
        if relevance_threshold is not None:
            self.cfg.relevance_threshold = relevance_threshold
        self._rebuild_retriever()
        self._rebuild_chain()

    def _rebuild_retriever(self) -> None:
        if self.vectorstore is None:
            self.retriever = None
            return
        self.retriever = self._build_retriever(self.vectorstore)

    def _rebuild_chain(self) -> None:
        if self.retriever is None:
            self.chain = None
            return
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
