from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "deepseek/deepseek-v4-flash"
    embedding_model: str = "baai/bge-m3"

    llm_provider: str = "openai"
    llm_api_base_url: str = "https://polza.ai/api/v1"
    llm_api_key: str = ""

    embedding_mode: str = "api"

    whisper_model: str = "whisper-1"

    documents_dir: Path = Path("data/documents")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "factory_docs"

    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 8
    fetch_k: int = 20
    relevance_threshold: float = 0.35

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    onboarding_doc_name: str = "onboarding_checklist.txt"


settings = Settings()
