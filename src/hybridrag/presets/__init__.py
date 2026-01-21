"""
HybridRAG Search Presets

Predefined search configurations for common use cases.
"""

from .search_presets import (
    PRESETS,
    SearchPreset,
    get_preset,
    list_presets,
    register_preset,
)

__all__ = [
    "SearchPreset",
    "PRESETS",
    "get_preset",
    "list_presets",
    "register_preset",
]
