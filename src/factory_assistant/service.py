from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from factory_assistant.admin import get_index_status, list_document_files, list_ollama_models, preview_index_plan
from factory_assistant.config import Settings, settings
from factory_assistant.documents import delete_document, save_upload
from factory_assistant.ingest import CATEGORY_KEYWORDS, build_vectorstore, ingest_documents
from factory_assistant.rag import RagEngine

logger = logging.getLogger(__name__)

# Карта поле конфига -> переменная окружения для сохранения в .env
_ENV_MAP = {
    "llm_provider": "LLM_PROVIDER",
    "llm_api_base_url": "LLM_API_BASE_URL",
    "llm_api_key": "LLM_API_KEY",
    "llm_model": "LLM_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "whisper_model": "WHISPER_MODEL",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "top_k": "TOP_K",
    "fetch_k": "FETCH_K",
    "relevance_threshold": "RELEVANCE_THRESHOLD",
}

CATEGORY_LABELS: dict[str, str] = {
    "tech_cards": "Технические карты",
    "instructions": "Инструкции",
    "regulations": "Регламенты",
    "sop": "Справочники (SOP)",
    "normative": "Нормативные документы",
    "safety": "Охрана труда",
    "onboarding": "Адаптация",
    "other": "Прочее",
}


class FactoryAssistantService:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.rag = RagEngine(self.cfg)
        self._reindex_running = False
        self._reindex_last: dict | None = None
        self._reindex_thread: threading.Thread | None = None

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
            "reindex_running": self._reindex_running,
            "reindex_last": self._reindex_last,
        }

    def update_config(self, payload: dict) -> dict:
        changed: dict = {}
        provider_changed = False
        if "llm_provider" in payload and payload["llm_provider"]:
            self.cfg.llm_provider = payload["llm_provider"]
            changed["llm_provider"] = payload["llm_provider"]
            provider_changed = True

        if "llm_api_base_url" in payload:
            self.cfg.llm_api_base_url = (payload["llm_api_base_url"] or "").strip().rstrip("/")
            changed["llm_api_base_url"] = self.cfg.llm_api_base_url

        if "llm_api_key" in payload:
            self.cfg.llm_api_key = payload["llm_api_key"]
            changed["llm_api_key"] = "***" if payload["llm_api_key"] else ""

        if "whisper_model" in payload and payload["whisper_model"]:
            self.cfg.whisper_model = payload["whisper_model"]
            changed["whisper_model"] = payload["whisper_model"]

        if "llm_model" in payload and payload["llm_model"]:
            self.cfg.llm_model = payload["llm_model"]
            changed["llm_model"] = payload["llm_model"]
            if provider_changed:
                self.rag.update_provider(
                    self.cfg.llm_provider,
                    self.cfg.llm_api_base_url,
                    self.cfg.llm_api_key,
                )
            else:
                self.rag.update_llm_model(payload["llm_model"])

        embedding_changed = False
        if "embedding_model" in payload and payload["embedding_model"]:
            self.cfg.embedding_model = payload["embedding_model"]
            changed["embedding_model"] = payload["embedding_model"]
            embedding_changed = True

        chunking_changed = False
        for field in ("chunk_size", "chunk_overlap", "top_k", "fetch_k", "relevance_threshold"):
            if field in payload and payload[field] is not None:
                setattr(self.cfg, field, payload[field])
                changed[field] = payload[field]
                if field in ("chunk_size", "chunk_overlap"):
                    chunking_changed = True

        if self.rag.vectorstore is not None:
            self.rag.update_retrieval_settings(
                top_k=payload.get("top_k"),
                fetch_k=payload.get("fetch_k"),
                relevance_threshold=payload.get("relevance_threshold"),
            )

        # Смена модели эмбеддинга или параметров чанкирования требует
        # переиндексации: старые векторы/чанки несовместимы с новыми.
        needs_reindex = embedding_changed or chunking_changed
        reindex_triggered = False
        if needs_reindex:
            reindex_triggered = self._start_reindex()

        # Сохраняем изменения в .env, чтобы пережили перезапуск сервера.
        self._persist_env(changed)

        result = self.get_config()
        result["changed"] = changed
        result["reindex_triggered"] = reindex_triggered
        if needs_reindex and not reindex_triggered:
            result["reindex_error"] = "Не удалось запустить переиндексацию (уже выполняется или ошибка)."
        return result

    def _start_reindex(self) -> bool:
        if self._reindex_running:
            return False
        self._reindex_running = True

        def _run():
            try:
                logger.info("Переиндексация запущена из панели настроек")
                res = self.reindex(recreate=True)
                self._reindex_last = {
                    "ok": True,
                    "documents": res.get("documents"),
                    "chunks": res.get("chunks"),
                    "error": res.get("error"),
                }
                logger.info("Переиндексация завершена: %s", self._reindex_last)
            except Exception as exc:  # noqa: BLE001
                self._reindex_last = {"ok": False, "error": str(exc)}
                logger.exception("Ошибка переиндексации из панели настроек")
            finally:
                self._reindex_running = False

        self._reindex_thread = threading.Thread(target=_run, daemon=True)
        self._reindex_thread.start()
        return True

    def _persist_env(self, changed: dict) -> None:
        if not changed:
            return
        env_path = Path(".env")
        if not env_path.exists():
            env_path = self.cfg.documents_dir.parent / ".env"
        if not env_path.exists():
            try:
                env_path.write_text("", encoding="utf-8")
            except Exception:
                return
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return

        seen = set()
        new_lines: list[str] = []
        for line in lines:
            key = line.split("=", 1)[0].strip()
            mapped = None
            for cfg_key, env_key in _ENV_MAP.items():
                if env_key == key:
                    mapped = cfg_key
                    break
            if mapped in changed:
                value = changed[mapped]
                if mapped == "llm_api_key" and value == "***":
                    new_lines.append(line)  # не перезаписываем секрет пустышкой/звёздочками
                else:
                    new_lines.append(f"{key}={value}")
                seen.add(mapped)
            else:
                new_lines.append(line)

        for cfg_key, env_key in _ENV_MAP.items():
            if cfg_key in changed and cfg_key not in seen:
                value = changed[cfg_key]
                if not (cfg_key == "llm_api_key" and value == "***"):
                    new_lines.append(f"{env_key}={value}")

        try:
            env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("Не удалось сохранить .env: %s", exc)

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
                model=self.cfg.whisper_model,
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

    def list_documents(self, type: str | None = None, search: str | None = None, recent: bool = False) -> dict:
        files = list_document_files(self.cfg.documents_dir)
        if type:
            files = [f for f in files if f.get("category") == type]
        if search:
            q = search.lower()
            files = [f for f in files if q in f["name"].lower()]
        if recent:
            files.sort(key=lambda f: f.get("modified_at", 0), reverse=True)
        return {"files": files, "total": len(files)}

    def get_document_categories(self) -> dict:
        files = list_document_files(self.cfg.documents_dir)
        cats: dict[str, list[dict]] = {}
        for f in files:
            cat = f.get("category", "other")
            cats.setdefault(cat, []).append(f)
        return {"categories": cats, "labels": CATEGORY_LABELS}

    def get_document_preview(self, filename: str, chunk_id: int | None = None) -> dict:
        from factory_assistant.ingest import load_documents, split_documents
        root = self.cfg.documents_dir.resolve()
        target = (root / filename).resolve()
        if root not in target.parents:
            raise FileNotFoundError("Доступ запрещён")
        if not target.exists():
            raise FileNotFoundError(f"Файл не найден: {filename}")

        ext = target.suffix.lower()
        is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        if is_image:
            return {
                "filename": filename,
                "type": "image",
                "media_url": f"/api/media/documents/{filename}",
            }

        if chunk_id is not None:
            docs = load_documents(self.cfg.documents_dir)
            chunks = split_documents(docs, self.cfg)
            chunk = next((c for c in chunks if c.metadata.get("chunk_id") == chunk_id and c.metadata.get("source_file") == filename), None)
            if not chunk:
                raise FileNotFoundError(f"Чанк {chunk_id} не найден")
            return {
                "filename": filename,
                "type": "chunk",
                "chunk_id": chunk_id,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }

        docs = load_documents(self.cfg.documents_dir)
        chunks = split_documents(docs, self.cfg)
        file_chunks = [c for c in chunks if c.metadata.get("source_file") == filename]
        return {
            "filename": filename,
            "type": "document",
            "chunks_count": len(file_chunks),
            "preview": file_chunks[0].page_content[:500] if file_chunks else "",
        }

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

    def _wipe_vectorstore(self) -> None:
        chroma_dir = self.cfg.chroma_dir.resolve()
        # Chroma кэширует клиента как singleton по пути. Чтобы пересоздать БД
        # с другой моделью эмбеддинга, сбрасываем кэш и удаляем коллекцию.
        # Важно: создаём клиента с настройками по умолчанию (как делает
        # langchain.Chroma), иначе конструктор ругается на «different settings».
        try:
            from chromadb.api.shared_system_client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_dir))
            client.delete_collection(self.cfg.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось удалить коллекцию Chroma: %s", exc)

    def reindex(self, recreate: bool = True) -> dict:
        # При смене модели эмбеддинга/параметров старые векторы несовместимы.
        # Нужно полностью сбросить Chroma: иначе singleton-клиент на том же пути
        # не даст пересоздать коллекцию с другими настройками эмбеддинга.
        if recreate:
            self._wipe_vectorstore()
            self.rag.vectorstore = None
            self.rag.retriever = None
            self.rag.chain = None

        result = ingest_documents(cfg=self.cfg, recreate=recreate)
        self.rag._try_load_vectorstore()
        if not self.rag.vectorstore and not recreate:
            import gc

            gc.collect()
            result = ingest_documents(cfg=self.cfg, recreate=False)
            self.rag._try_load_vectorstore()
        return result
