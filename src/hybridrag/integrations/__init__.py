"""
HybridRAG Integrations.

Provider integrations for LLM and embeddings:
- Voyage AI: Embeddings (voyage-3-large) and Reranking (rerank-2.5)
- Anthropic: Claude LLM
- OpenAI: GPT LLM and embeddings
- Gemini: Google AI LLM and embeddings
- Langfuse: Observability and tracing
"""

from .voyage import (
    VoyageEmbedder,
    VoyageReranker,
    create_embedding_func,
    create_rerank_func,
)

from .langfuse import (
    get_langfuse,
    flush_langfuse,
    is_enabled as langfuse_enabled,
    get_status as langfuse_status,
    trace_rag_query,
    trace_span,
    log_rag_query,
    log_ingestion,
    create_traced_llm_func,
    create_traced_embedding_func,
)

__all__ = [
    # Voyage AI
    "VoyageEmbedder",
    "VoyageReranker",
    "create_embedding_func",
    "create_rerank_func",
    # Langfuse
    "get_langfuse",
    "flush_langfuse",
    "langfuse_enabled",
    "langfuse_status",
    "trace_rag_query",
    "trace_span",
    "log_rag_query",
    "log_ingestion",
    "create_traced_llm_func",
    "create_traced_embedding_func",
]
