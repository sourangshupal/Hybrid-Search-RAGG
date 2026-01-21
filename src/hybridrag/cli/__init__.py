"""
HybridRAG CLI Module.

Provides both command-line and interactive interfaces for HybridRAG with:
- Typer-based CLI commands (ingest, query, status, etc.)
- Interactive conversational chat
- Real-time streaming responses
- Rich terminal UI
"""

from .app import run as run_cli
from .main import conversation_loop, main

__all__ = ["run_cli", "main", "conversation_loop"]
