"""
HybridRAG Ingestion Module.

This module provides document processing and chunking capabilities:
- Multi-format document support (PDF, Word, PPT, Excel, HTML, Audio)
- Structure-aware chunking via Docling HybridChunker
- Audio transcription via Whisper ASR
"""

from .types import (
    DocumentChunk,
    ChunkingConfig,
    IngestionConfig,
    IngestionResult,
    ProcessedDocument,
)
from .chunker import DoclingHybridChunker, create_chunker
from .document_processor import DocumentProcessor
from .pipeline import DocumentIngestionPipeline

# Tavily processor (optional dependency)
try:
    from .tavily_processor import TavilyProcessor, create_tavily_processor

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyProcessor = None  # type: ignore
    create_tavily_processor = None  # type: ignore

__all__ = [
    # Types
    "DocumentChunk",
    "ChunkingConfig",
    "IngestionConfig",
    "IngestionResult",
    "ProcessedDocument",
    # Chunker
    "DoclingHybridChunker",
    "create_chunker",
    # Processor
    "DocumentProcessor",
    # Pipeline
    "DocumentIngestionPipeline",
    # Tavily (optional)
    "TavilyProcessor",
    "create_tavily_processor",
]
