from __future__ import annotations

from factory_assistant.admin import get_index_status, list_ollama_models, preview_index_plan
from factory_assistant.config import Settings, settings
from factory_assistant.documents import delete_document, save_upload
from factory_assistant.ingest import ingest_documents
from factory_assistant.onboarding import OnboardingManager
from factory_assistant.rag import RagEngine


class FactoryAssistantService:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.rag = RagEngine(self.cfg)
        self.onboarding = OnboardingManager(self.cfg)

    def get_config(self) -> dict:
        return {
            "ollama_base_url": self.cfg.ollama_base_url,
            "llm_model": self.cfg.llm_model,
            "embedding_model": self.cfg.embedding_model,
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
            "top_k": self.cfg.top_k,
            "fetch_k": self.cfg.fetch_k,
            "relevance_threshold": self.cfg.relevance_threshold,
            "documents_dir": str(self.cfg.documents_dir.resolve()),
            "chroma_dir": str(self.cfg.chroma_dir.resolve()),
        }

    def update_config(self, payload: dict) -> dict:
        if "llm_model" in payload and payload["llm_model"]:
            self.cfg.llm_model = payload["llm_model"]
            self.rag.update_llm_model(payload["llm_model"])
            self.onboarding.update_llm_model(payload["llm_model"])

        if "embedding_model" in payload and payload["embedding_model"]:
            self.cfg.embedding_model = payload["embedding_model"]

        for field in ("chunk_size", "chunk_overlap", "top_k", "fetch_k", "relevance_threshold"):
            if field in payload and payload[field] is not None:
                setattr(self.cfg, field, payload[field])

        if self.rag.vectorstore is not None:
            self.rag.update_retrieval_settings(
                top_k=payload.get("top_k"),
                fetch_k=payload.get("fetch_k"),
                relevance_threshold=payload.get("relevance_threshold"),
            )
        return self.get_config()

    def list_models(self) -> list[dict]:
        return list_ollama_models(self.cfg.ollama_base_url)

    def get_index_status(self) -> dict:
        return get_index_status(self.cfg)

    def preview_index(self) -> dict:
        return preview_index_plan(self.cfg)

    def ask(self, question: str) -> dict:
        result = self.rag.ask(question).as_dict()
        result["llm_model"] = self.cfg.llm_model
        return result

    def start_onboarding(self, session_id: str | None = None) -> dict:
        session = self.onboarding.start_session(session_id)
        step = session.current_step
        return {
            **session.as_dict(),
            "reply": (
                f"Начинаем адаптацию. Шаг {step.number} из {session.total_steps}: "
                f"{step.title}\n\n{step.body}\n\n"
                "Команды: «далее», «назад», «статус»."
            ),
        }

    def onboarding_message(self, session_id: str, message: str) -> dict:
        return self.onboarding.handle_message(session_id, message)

    def upload_document(self, filename: str, content: bytes, *, auto_reindex: bool = True) -> dict:
        saved = save_upload(filename, content, self.cfg.documents_dir)
        result = {"upload": saved, "reindex": None}
        if auto_reindex:
            result["reindex"] = self.reindex(recreate=True)
        return result

    def remove_document(self, filename: str, *, auto_reindex: bool = True) -> dict:
        deleted = delete_document(filename, self.cfg.documents_dir)
        result = {"deleted": deleted, "reindex": None}
        if auto_reindex:
            result["reindex"] = self.reindex(recreate=True)
        return result

    def reindex(self, recreate: bool = True) -> dict:
        if recreate and self.rag.vectorstore is not None:
            try:
                existing = self.rag.vectorstore
                collection = existing._collection
                ids = collection.get(include=[])["ids"]
                if ids:
                    collection.delete(ids=ids)
            except Exception:
                pass
            self.rag.vectorstore = None
            self.rag.retriever = None
            self.rag.chain = None

        result = ingest_documents(cfg=self.cfg, recreate=False)
        self.rag._try_load_vectorstore()
        if not self.rag.vectorstore:
            import gc
            gc.collect()
            result = ingest_documents(cfg=self.cfg, recreate=False)
            self.rag._try_load_vectorstore()
        return result
