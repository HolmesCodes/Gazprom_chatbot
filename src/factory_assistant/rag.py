from __future__ import annotations

from dataclasses import dataclass, field

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

    def as_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
        }


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    found_in_kb: bool = True

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "found_in_kb": self.found_in_kb,
        }


def format_source(doc: Document) -> SourceReference:
    meta = doc.metadata
    return SourceReference(
        source_file=meta.get("source_file", meta.get("source", "unknown")),
        page_number=meta.get("page_number"),
        chunk_id=meta.get("chunk_id"),
        excerpt=doc.page_content[:240].strip(),
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
        return self.retriever.invoke(question)

    def _build_retriever(self, vectorstore: Chroma):
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.cfg.top_k, "fetch_k": self.cfg.fetch_k},
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
        scored = self.vectorstore.similarity_search_with_relevance_scores(
            question,
            k=self.cfg.top_k,
        )
        max_score = scored[0][1] if scored else 0.0
        sources = [format_source(doc) for doc in docs]

        if not docs or max_score < self.cfg.relevance_threshold:
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

        answer = self.chain.invoke(question)
        if NOT_FOUND_ANSWER.lower() in answer.lower():
            return RagAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found_in_kb=False,
            )

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
