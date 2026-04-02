"""Application configuration using pydantic-settings."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "mall_shops"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Moderation
    moderation_enabled: bool = True
    moderation_model: str = "omni-moderation-latest"

    # Normalization review
    normalization_review_model: str = "gpt-4o-mini"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.5
    hybrid_candidate_multiplier: int = 4
    hybrid_min_score: float = 0.2

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    environment: Literal["development", "production"] = "development"

    # Data paths
    data_csv_path: str = "data/shops.csv"
    name_mappings_path: str = "data/name_mappings.json"

    # Chunking
    chunk_strategy: Literal["single", "hierarchical"] = "single"
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 50


def get_settings() -> Settings:
    """Factory function for cached settings instance."""
    return Settings()
