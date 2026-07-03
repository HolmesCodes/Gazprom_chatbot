from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from factory_assistant.service import FactoryAssistantService

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
service: FactoryAssistantService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = FactoryAssistantService()
    yield
    service = None


app = FastAPI(
    title="Factory Assistant API",
    description="RAG-помощник специалиста на заводе (локальный Ollama)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class OnboardingStartRequest(BaseModel):
    session_id: str | None = None


class OnboardingMessageRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)


class ReindexRequest(BaseModel):
    recreate: bool = True


class ConfigUpdateRequest(BaseModel):
    llm_model: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    fetch_k: int | None = Field(default=None, ge=1, le=100)
    relevance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def _get_service() -> FactoryAssistantService:
    if service is None:
        raise HTTPException(status_code=503, detail="Сервис ещё не инициализирован")
    return service


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    return _get_service().get_config()


@app.patch("/api/admin/config")
def update_config(payload: ConfigUpdateRequest) -> dict:
    return _get_service().update_config(payload.model_dump(exclude_none=True))


@app.get("/api/admin/models")
def list_models() -> dict:
    try:
        models = _get_service().list_models()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"models": models, "current": _get_service().get_config()}


@app.get("/api/admin/index/status")
def index_status() -> dict:
    return _get_service().get_index_status()


@app.get("/api/admin/index/preview")
def index_preview() -> dict:
    return _get_service().preview_index()


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict:
    return _get_service().ask(payload.question)


@app.post("/api/onboarding/start")
def onboarding_start(payload: OnboardingStartRequest) -> dict:
    return _get_service().start_onboarding(payload.session_id)


@app.post("/api/onboarding/message")
def onboarding_message(payload: OnboardingMessageRequest) -> dict:
    try:
        return _get_service().onboarding_message(payload.session_id, payload.message)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/admin/reindex")
def reindex(payload: ReindexRequest) -> dict:
    return _get_service().reindex(recreate=payload.recreate)


@app.post("/api/admin/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    auto_reindex: bool = Query(True),
) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")
    try:
        return _get_service().upload_document(file.filename or "document.txt", content, auto_reindex=auto_reindex)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/admin/documents/{filename}")
def remove_document(filename: str, auto_reindex: bool = True) -> dict:
    try:
        return _get_service().remove_document(filename, auto_reindex=auto_reindex)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Legacy endpoints
@app.post("/ask")
def ask_legacy(payload: AskRequest) -> dict:
    return ask(payload)


@app.post("/onboarding/start")
def onboarding_start_legacy(payload: OnboardingStartRequest) -> dict:
    return onboarding_start(payload)


@app.post("/onboarding/message")
def onboarding_message_legacy(payload: OnboardingMessageRequest) -> dict:
    return onboarding_message(payload)


@app.post("/admin/reindex")
def reindex_legacy(payload: ReindexRequest) -> dict:
    return reindex(payload)


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
