"""
HybridRAG Prompts Module.

Central repository for all RAG chatbot prompts, optimized for:
- Large context window LLMs (Gemini, Claude, GPT-4)
- Multi-language support (English + other languages)
- RAG-grounded responses with citations
- Knowledge graph extraction and entity recognition

Prompt Categories:
- System: Main chat system prompts with persona and instructions
- Reranking: Query-type specific reranking instructions
- Entity: Knowledge graph entity/relationship extraction
- Memory: Conversation summarization and session context
- Topic: Semantic tagging and topic extraction
"""

from .system_prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    create_system_prompt,
)
from .reranking_prompt import (
    RERANK_INSTRUCTION_GENERAL,
    RERANK_INSTRUCTION_SUMMARY,
    RERANK_INSTRUCTION_TOOLS,
    RERANK_INSTRUCTION_TROUBLESHOOTING,
    QUERY_TYPE_PATTERNS,
    QueryType,
    detect_query_type,
    select_rerank_instruction,
)
from .entity_extraction_prompt import (
    ENTITY_EXTRACTION_PROMPT,
    QUERY_ENTITY_EXTRACTION_PROMPT,
    ENTITY_NORMALIZATION_PROMPT,
)
from .memory_prompt import (
    MEMORY_SUMMARIZATION_PROMPT,
    MEMORY_SUMMARIZATION_PROMPT_LITE,
    SESSION_CONTEXT_PROMPT,
)
from .topic_extraction_prompt import (
    TOPIC_EXTRACTION_PROMPT,
    BATCH_TOPIC_EXTRACTION_PROMPT,
    TOPIC_CLUSTERING_PROMPT,
)

__all__ = [
    # System prompts
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_COMPACT",
    "create_system_prompt",
    # Reranking
    "RERANK_INSTRUCTION_GENERAL",
    "RERANK_INSTRUCTION_SUMMARY",
    "RERANK_INSTRUCTION_TOOLS",
    "RERANK_INSTRUCTION_TROUBLESHOOTING",
    "QUERY_TYPE_PATTERNS",
    "QueryType",
    "detect_query_type",
    "select_rerank_instruction",
    # Entity extraction
    "ENTITY_EXTRACTION_PROMPT",
    "QUERY_ENTITY_EXTRACTION_PROMPT",
    "ENTITY_NORMALIZATION_PROMPT",
    # Memory
    "MEMORY_SUMMARIZATION_PROMPT",
    "MEMORY_SUMMARIZATION_PROMPT_LITE",
    "SESSION_CONTEXT_PROMPT",
    # Topics
    "TOPIC_EXTRACTION_PROMPT",
    "BATCH_TOPIC_EXTRACTION_PROMPT",
    "TOPIC_CLUSTERING_PROMPT",
]

# Prompt Selection Guide
#
# MAIN CHAT:
# - SYSTEM_PROMPT: Full version (~4000 tokens) - use for best quality
# - SYSTEM_PROMPT_COMPACT: Condensed (~500 tokens) - use if context limited
# - create_system_prompt(): Factory for custom domains
#
# RERANKING:
# - detect_query_type(): Detect query intent from patterns
# - select_rerank_instruction(): Get appropriate instruction for query type
#
# ENTITIES:
# - ENTITY_EXTRACTION_PROMPT: Full extraction from documents
# - QUERY_ENTITY_EXTRACTION_PROMPT: Light extraction from user queries
# - ENTITY_NORMALIZATION_PROMPT: Normalize/deduplicate entities
#
# MEMORY:
# - MEMORY_SUMMARIZATION_PROMPT: Full progressive summarization
# - MEMORY_SUMMARIZATION_PROMPT_LITE: Quick summarization
# - SESSION_CONTEXT_PROMPT: Generate welcome-back message
#
# TOPICS:
# - TOPIC_EXTRACTION_PROMPT: Single document topic extraction
# - BATCH_TOPIC_EXTRACTION_PROMPT: Multiple documents at once
# - TOPIC_CLUSTERING_PROMPT: Group related topics
