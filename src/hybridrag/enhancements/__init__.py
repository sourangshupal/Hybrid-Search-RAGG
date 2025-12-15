"""
HybridRAG Enhancements.

Advanced RAG capabilities:
- Entity Boosting: Structural relevance signal from KG entities
- Implicit Expansion: Find semantically related entities without explicit graph edges
"""

from .entity_boosting import EntityBoostingReranker, create_boosted_rerank_func
from .implicit_expansion import ImplicitExpander

__all__ = [
    "EntityBoostingReranker",
    "create_boosted_rerank_func",
    "ImplicitExpander",
]
