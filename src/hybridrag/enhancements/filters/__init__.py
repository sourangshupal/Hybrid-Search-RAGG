"""
MongoDB Filter Builders for HybridRAG.

This module provides two distinct filter builder systems:
1. Vector Search Filters - Use standard MongoDB operators ($gte, $lte, $eq)
2. Atlas Search Filters - Use Atlas Search operators (range, equals)

CRITICAL: These use DIFFERENT syntaxes for the same logical filters!
"""

from .vector_search_filters import (
    VectorSearchFilterConfig,
    build_vector_search_filters,
)
from .atlas_search_filters import (
    AtlasSearchFilterConfig,
    build_atlas_search_filters,
)

__all__ = [
    "build_vector_search_filters",
    "VectorSearchFilterConfig",
    "build_atlas_search_filters",
    "AtlasSearchFilterConfig",
]
