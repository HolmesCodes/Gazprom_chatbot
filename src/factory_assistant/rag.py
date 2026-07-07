from __future__ import annotations

import hashlib
import json
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

NOT_FOUND_ANSWER = "Информация не найдена в базе знаний"


@dataclass
class SourceReference:
    source_file: str
    page_number: int | None = None
    chunk_id: int | None = None
    excerpt: str = ""
    image_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "image_paths": self.image_paths,
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
        if self.retriever is None:
            return []
        try:
            emb_results = self.retriever.invoke(question)
        except Exception:
            emb_results = []

        keywords = [w for w in question.split() if len(w) > 3]
        kw_results: list[Document] = []
        if keywords:
            try:
                kw_results = self.vectorstore.similarity_search(
                    " ".join(keywords), k=self.cfg.top_k
                )
            except Exception:
                pass

        image_results: list[Document] = []
        lower_q = question.lower()
        if any(kw in lower_q for kw in ("картинк", "изображени", "рисунк", "рис.", "фото", "иллюстрац", "покаж", "чертеж", "схем")):
            for term in ("Рис.", "Приложение", "игровое поле", "размеры", "элементы"):
                try:
                    more = self.vectorstore.similarity_search(term, k=self.cfg.top_k)
                    image_results.extend(more)
                except Exception:
                    pass

        seen_ids: set[str] = set()
        merged: list[Document] = []
        for doc in emb_results + kw_results + image_results:
            doc_id = f"{doc.metadata.get('source_file')}:{doc.metadata.get('page_number')}:{doc.page_content[:100]}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)
        return merged[: self.cfg.top_k]

    def _build_retriever(self, vectorstore: Chroma):
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.cfg.top_k, "fetch_k": 50},
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

        vision_cache: dict[str, str] = {}
        try:
            vision_cache = json.loads(
                Path("data/vision_cache.json").read_text()
            )
        except Exception:
            pass

        if vision_cache:
            for doc in docs:
                img_paths = doc.metadata.get("image_paths", []) or []
                if not img_paths:
                    continue
                source_file = doc.metadata.get("source_file", "")
                page_number = doc.metadata.get("page_number", "")
                descs: list[str] = []
                for img_path in img_paths:
                    full = (
                        Path("data/images")
                        / Path(source_file).stem
                        / img_path
                    )
                    cache_key = str(full.resolve())
                    desc = vision_cache.get(cache_key, "")
                    if desc:
                        descs.append(desc)
                if descs:
                    extra = "\n\nИзображения на этой странице:\n" + "\n".join(
                        f"- {d}" for d in descs
                    )
                    doc.page_content += extra
        # Check relevance by matching retrieved docs to question terms
        try:
            question_terms = set(
                w.lower() for w in question.split() if len(w) > 2
            )
            doc_terms: set[str] = set()
            for doc in docs:
                terms = set(
                    w.lower()
                    for w in doc.page_content.split()
                    if len(w) > 2
                )
                doc_terms |= terms

            overlap = question_terms & doc_terms
            # If any key term overlaps, docs are relevant
            # Otherwise, run similarity fallback
            if not overlap:
                scored = self.vectorstore.similarity_search_with_relevance_scores(
                    question, k=1,
                )
                max_score = scored[0][1] if scored else 0.0

                if max_score < self.cfg.relevance_threshold:
                    keywords = " ".join(
                        w for w in question.split() if len(w) > 3
                    )
                    if keywords:
                        kw_scored = self.vectorstore.similarity_search_with_relevance_scores(
                            keywords, k=1,
                        )
                        max_score = kw_scored[0][1] if kw_scored else max_score

                    if max_score < self.cfg.relevance_threshold:
                        return RagAnswer(
                            question=question,
                            answer=NOT_FOUND_ANSWER,
                            sources=[],
                            found_in_kb=False,
                        )
        except Exception:
            pass
        sources = [format_source(doc) for doc in docs]

        if not docs:
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

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

        # Edge case: изображения на соседних страницах (±1)
        expanded_pages: set[int] = set()
        for p in cited_pages:
            expanded_pages.add(p)
            if p > 1:
                expanded_pages.add(p - 1)
            expanded_pages.add(p + 1)
        cited_pages = expanded_pages

        if cited_pages:
            filtered_sources = [s for s in sources if s.page_number in cited_pages]
            if filtered_sources:
                sources = filtered_sources

        query_words = {
            w.lower()
            for w in re.findall(r"[а-яёa-z]{4,}", question.lower())
        }

        vision_cache: dict[str, str] = {}
        try:
            vision_cache = json.loads(
                Path("data/vision_cache.json").read_text()
            )
        except Exception:
            pass

        seen_hashes: set[str] = set()
        min_size = 3000
        for src in sources:
            unique: list[str] = []
            for img_path in src.image_paths:
                full = (
                    Path("data/images")
                    / Path(src.source_file).stem
                    / img_path
                )
                if full.exists():
                    if full.stat().st_size < min_size:
                        continue
                    h = hashlib.md5(full.read_bytes()).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    cache_key = str(full.resolve())
                    desc = vision_cache.get(cache_key, "")
                    if desc and query_words:
                        desc_words = set(re.findall(r"[а-яёa-z]{4,}", desc.lower()))
                        overlap = query_words & desc_words
                        if not overlap:
                            continue
                unique.append(img_path)
            src.image_paths = unique

        return RagAnswer(
            question=question,
            answer=answer,
            sources=sources,
            found_in_kb=True,
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
