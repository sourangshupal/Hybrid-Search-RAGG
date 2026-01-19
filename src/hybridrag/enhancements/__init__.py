"""
HybridRAG Enhancements.

Advanced RAG capabilities:
- Entity Boosting: Structural relevance signal from KG entities
- Implicit Expansion: Find semantically related entities without explicit graph edges
- MongoDB Hybrid Search: Native $rankFusion and manual RRF with fuzzy text matching
- Filter Builders: MongoDB 8.2 vector search and Atlas Search prefiltering
"""

from .entity_boosting import EntityBoostingReranker, create_boosted_rerank_func
from .filters import (
    AtlasSearchFilterConfig,
    VectorSearchFilterConfig,
    build_atlas_search_filters,
    build_vector_search_filters,
)
from .implicit_expansion import ImplicitExpander
from .mongodb_hybrid_search import (
    MongoDBHybridSearchConfig,
    MongoDBHybridSearcher,
    SearchResult,
    create_hybrid_searcher,
    hybrid_search_with_rank_fusion,
    hybrid_search_with_score_fusion,
    manual_hybrid_search_with_rrf,
    multi_field_text_search,
    reciprocal_rank_fusion,
    text_only_search,
    vector_only_search,
)

__all__ = [
    # Entity Boosting
    "EntityBoostingReranker",
    "create_boosted_rerank_func",
    # Implicit Expansion
    "ImplicitExpander",
    # Config
    "MongoDBHybridSearchConfig",
    "VectorSearchFilterConfig",
    "AtlasSearchFilterConfig",
    # Searcher
    "MongoDBHybridSearcher",
    "create_hybrid_searcher",
    # Search functions
    "hybrid_search_with_rank_fusion",
    "hybrid_search_with_score_fusion",
    "manual_hybrid_search_with_rrf",
    "multi_field_text_search",
    "text_only_search",
    "vector_only_search",
    # Utilities
    "reciprocal_rank_fusion",
    "SearchResult",
    # Filter builders
    "build_vector_search_filters",
    "build_atlas_search_filters",
]
