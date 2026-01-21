"""
Type definitions for the ingestion module.

This module defines the core data structures used throughout the ingestion pipeline:
- DocumentChunk: A single chunk of a document with metadata
- ChunkingConfig: Configuration for the chunker
- IngestionConfig: Configuration for the full pipeline
- IngestionResult: Result of document ingestion
- ProcessedDocument: A processed document with markdown and optional DoclingDocument
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        max_tokens: Maximum tokens per chunk (for embedding models).
        chunk_size: Target characters per chunk (used in fallback).
        chunk_overlap: Character overlap between chunks (used in fallback).
        max_chunk_size: Maximum chunk size in characters (used in fallback).
        min_chunk_size: Minimum chunk size in characters (used in fallback).
        tokenizer_model: HuggingFace model ID for tokenizer.
        merge_peers: Whether to merge small adjacent chunks.
    """

    max_tokens: int = 512
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    merge_peers: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")


@dataclass
class IngestionConfig:
    """Configuration for the full ingestion pipeline.

    Attributes:
        chunking: Chunking configuration.
        clean_before_ingest: Whether to clean existing data before ingestion.
        batch_size: Batch size for embedding generation.
        enable_audio_transcription: Whether to enable audio transcription.
    """

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    clean_before_ingest: bool = True
    batch_size: int = 100
    enable_audio_transcription: bool = True


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata.

    Attributes:
        content: The chunk text content.
        index: Chunk index within the document.
        start_char: Start character position in original document.
        end_char: End character position in original document.
        metadata: Additional metadata (title, source, etc.).
        token_count: Number of tokens in the chunk.
        embedding: Optional embedding vector.
    """

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int | None = None
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        """Calculate token count estimate if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "content": self.content,
            "chunk_index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "embedding": self.embedding,
        }


@dataclass
class ProcessedDocument:
    """A processed document with markdown content and optional DoclingDocument.

    Attributes:
        content: Document content in markdown format.
        title: Document title.
        source: Document source path or identifier.
        metadata: Additional metadata.
        docling_document: Optional DoclingDocument for HybridChunker.
        format_type: Original format type (pdf, docx, audio, etc.).
    """

    content: str
    title: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    docling_document: Any | None = None  # DoclingDocument from docling_core
    format_type: str = "text"

    @property
    def has_docling_document(self) -> bool:
        """Check if DoclingDocument is available for HybridChunker."""
        return self.docling_document is not None


@dataclass
class IngestionResult:
    """Result of document ingestion.

    Attributes:
        document_id: ID of the ingested document.
        title: Document title.
        chunks_created: Number of chunks created.
        processing_time_ms: Processing time in milliseconds.
        errors: List of errors encountered.
        source: Document source path.
        format_type: Original format type.
    """

    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: list[str] = field(default_factory=list)
    source: str = ""
    format_type: str = "text"

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and self.chunks_created > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chunks_created": self.chunks_created,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
            "source": self.source,
            "format_type": self.format_type,
            "success": self.success,
        }
