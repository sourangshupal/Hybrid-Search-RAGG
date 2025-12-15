"""
Configuration settings for HybridRAG.

Uses pydantic-settings for type-safe configuration from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """HybridRAG configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB Atlas
    mongodb_uri: SecretStr = Field(
        ...,
        description="MongoDB Atlas connection URI",
    )
    mongodb_database: str = Field(
        default="hybridrag",
        description="MongoDB database name",
    )
    mongodb_workspace: str = Field(
        default="default",
        description="Workspace prefix for collections",
    )

    # Voyage AI (for embeddings and reranking)
    voyage_api_key: SecretStr | None = Field(
        default=None,
        description="Voyage AI API key (required if embedding_provider=voyage)",
    )
    voyage_embedding_model: str = Field(
        default="voyage-3-large",
        description="Voyage embedding model (voyage-3-large, voyage-3, voyage-code-3)",
    )
    voyage_context_model: str = Field(
        default="voyage-context-3",
        description="Voyage contextualized embedding model",
    )
    voyage_rerank_model: str = Field(
        default="rerank-2.5",
        description="Voyage reranking model",
    )

    # LLM Provider Selection
    llm_provider: Literal["anthropic", "openai", "gemini"] = Field(
        default="gemini",
        description="LLM provider to use (anthropic, openai, gemini)",
    )

    # Anthropic (Claude)
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (required if llm_provider=anthropic)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model for generation",
    )

    # OpenAI
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (required if llm_provider=openai)",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model",
    )

    # Google Gemini
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google AI API key (required if llm_provider=gemini)",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model (gemini-2.5-flash, gemini-2.0-flash)",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model",
    )

    # Embedding Provider - VOYAGE ONLY (best quality)
    # Note: We only support Voyage AI for embeddings - no fallback to OpenAI/Gemini
    embedding_provider: Literal["voyage"] = Field(
        default="voyage",
        description="Embedding provider (Voyage AI only - best quality)",
    )

    # Query settings
    default_query_mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Default query mode",
    )
    default_top_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Default number of results to retrieve",
    )
    default_rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of results after reranking",
    )
    enable_rerank: bool = Field(
        default=True,
        description="Enable reranking by default",
    )

    # Enhancement settings
    enable_implicit_expansion: bool = Field(
        default=True,
        description="Enable implicit entity expansion",
    )
    implicit_expansion_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for implicit expansion",
    )
    implicit_expansion_max: int = Field(
        default=10,
        ge=1,
        description="Maximum entities from implicit expansion",
    )
    enable_entity_boosting: bool = Field(
        default=True,
        description="Enable entity boosting in reranking",
    )
    entity_boost_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for entity overlap boost",
    )

    # Embedding settings
    embedding_dim: int = Field(
        default=1024,
        description="Embedding dimension (1024 for voyage-3-large)",
    )
    max_token_size: int = Field(
        default=4096,
        description="Maximum tokens for embedding",
    )
    embedding_batch_size: int = Field(
        default=128,
        ge=1,
        le=128,
        description="Batch size for embedding API calls",
    )

    # Context limits
    max_token_for_text_unit: int = Field(
        default=4000,
        description="Maximum tokens per text unit",
    )
    max_token_for_local_context: int = Field(
        default=4000,
        description="Maximum tokens for local context",
    )
    max_token_for_global_context: int = Field(
        default=4000,
        description="Maximum tokens for global context",
    )

    # Observability (optional)
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public key",
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        description="Langfuse secret key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
