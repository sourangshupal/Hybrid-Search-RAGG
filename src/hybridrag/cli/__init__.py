"""
HybridRAG CLI Module.

Provides an interactive command-line interface for HybridRAG with:
- Conversational RAG queries
- Real-time streaming responses
- Rich terminal UI
"""

from .main import main, run_cli

__all__ = ["main", "run_cli"]
