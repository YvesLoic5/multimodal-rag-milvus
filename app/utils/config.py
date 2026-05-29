"""Application configuration loaded from environment variables via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — all values can be overridden via .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llava:13b", alias="OLLAMA_MODEL")

    # Milvus
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="multimodal_rag", alias="MILVUS_COLLECTION")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    redis_ttl: int = Field(default=3600, alias="REDIS_TTL")
    cache_similarity_threshold: float = Field(default=0.92, alias="CACHE_SIMILARITY_THRESHOLD")

    # Embedding models
    embedding_model_text: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL_TEXT")
    embedding_model_image: str = Field(
        default="openai/clip-vit-base-patch32", alias="EMBEDDING_MODEL_IMAGE"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # Retrieval
    top_k_retrieval: int = Field(default=20, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")

    # Ingestion
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "text"] = Field(default="text", alias="LOG_FORMAT")

    @field_validator("cache_similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("cache_similarity_threshold must be between 0 and 1")
        return v

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key != "sk-...")

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
