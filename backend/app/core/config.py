from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Document Assistant"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    top_k: int = 5

    project_root: Path = Path(__file__).resolve().parents[3]
    vector_store_dir: Path | None = None

    frontend_origin: str = "http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolved_vector_store_dir(self) -> Path:
        if self.vector_store_dir:
            return Path(self.vector_store_dir)
        return self.project_root / "data" / "vector_store"


@lru_cache
def get_settings() -> Settings:
    return Settings()
