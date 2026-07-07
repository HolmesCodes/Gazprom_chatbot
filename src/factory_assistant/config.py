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
    embedding_model: str = "nomic-embed-text:latest"

    llm_provider: str = "ollama"
    llm_api_base_url: str = ""
    llm_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    whisper_model: str = "large-v3-turbo"

    documents_dir: Path = Path("data/documents")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "factory_docs"

    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 8
    fetch_k: int = 20
    relevance_threshold: float = 0.15

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    onboarding_doc_name: str = "onboarding_checklist.txt"


settings = Settings()
