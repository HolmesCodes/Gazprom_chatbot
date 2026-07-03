from __future__ import annotations

from factory_assistant.admin import get_index_status, list_ollama_models, preview_index_plan
from factory_assistant.config import Settings, settings
from factory_assistant.documents import delete_document, save_upload
from factory_assistant.ingest import ingest_documents
from factory_assistant.rag import RagEngine


class FactoryAssistantService:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.rag = RagEngine(self.cfg)

    def get_config(self) -> dict:
        return {
            "ollama_base_url": self.cfg.ollama_base_url,
            "llm_model": self.cfg.llm_model,
            "embedding_model": self.cfg.embedding_model,
            "llm_provider": self.cfg.llm_provider,
            "llm_api_base_url": self.cfg.llm_api_base_url,
            "whisper_model": self.cfg.whisper_model,
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
            "top_k": self.cfg.top_k,
            "fetch_k": self.cfg.fetch_k,
            "relevance_threshold": self.cfg.relevance_threshold,
            "documents_dir": str(self.cfg.documents_dir.resolve()),
            "chroma_dir": str(self.cfg.chroma_dir.resolve()),
        }

    def update_config(self, payload: dict) -> dict:
        provider_changed = False
        if "llm_provider" in payload and payload["llm_provider"]:
            self.cfg.llm_provider = payload["llm_provider"]
            provider_changed = True

        if "llm_api_base_url" in payload:
            self.cfg.llm_api_base_url = (payload["llm_api_base_url"] or "").strip().rstrip("/")

        if "llm_api_key" in payload:
            self.cfg.llm_api_key = payload["llm_api_key"]

        if "whisper_model" in payload and payload["whisper_model"]:
            self.cfg.whisper_model = payload["whisper_model"]

        if "llm_model" in payload and payload["llm_model"]:
            self.cfg.llm_model = payload["llm_model"]
            if provider_changed:
                self.rag.update_provider(
                    self.cfg.llm_provider,
                    self.cfg.llm_api_base_url,
                    self.cfg.llm_api_key,
                )
            else:
                self.rag.update_llm_model(payload["llm_model"])

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

    def transcribe(self, audio_data: bytes) -> str:
        if self.cfg.llm_provider == "openai" and self.cfg.llm_api_base_url:
            return self._transcribe_openai(audio_data)
        return self._transcribe_local(audio_data)

    def _transcribe_openai(self, audio_data: bytes) -> str:
        from openai import OpenAI
        base_url = self.cfg.llm_api_base_url.strip().rstrip("/")
        client = OpenAI(base_url=base_url, api_key=self.cfg.llm_api_key or "sk-placeholder")
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.webm", audio_data, "audio/webm"),
            )
            return transcript.text.strip()
        except Exception as exc:
            raise RuntimeError(f"Ошибка распознавания через API: {exc}")

    def _transcribe_local(self, audio_data: bytes) -> str:
        import tempfile
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper не установлен. pip install faster-whisper")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        try:
            model = WhisperModel(self.cfg.whisper_model, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(tmp_path, beam_size=1, language="ru")
            text = " ".join(seg.text for seg in segments)
            return text.strip() or ""
        finally:
            import os
            os.unlink(tmp_path)

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
