"""Tests for Voyage AI integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


class TestVoyageEmbedder:
    """Tests for VoyageEmbedder class."""

    @pytest.fixture
    def mock_voyage_client(self):
        """Create mock Voyage client."""
        mock = MagicMock()
        mock.embed = MagicMock(
            return_value=MagicMock(embeddings=[[0.1] * 1024, [0.2] * 1024])
        )
        return mock

    @pytest.fixture
    def mock_async_voyage_client(self):
        """Create mock async Voyage client."""
        mock = AsyncMock()
        mock.embed = AsyncMock(
            return_value=MagicMock(embeddings=[[0.1] * 1024, [0.2] * 1024])
        )
        return mock

    @pytest.mark.asyncio
    async def test_embed_async_returns_numpy_array(self, mock_async_voyage_client):
        """Test that embed_async returns numpy array."""
        with patch("voyageai.AsyncClient", return_value=mock_async_voyage_client):
            with patch("voyageai.Client"):
                from hybridrag.integrations.voyage import VoyageEmbedder

                embedder = VoyageEmbedder(api_key="test-key")
                embedder._async_client = mock_async_voyage_client

                result = await embedder.embed_async(["text1", "text2"])

                assert isinstance(result, np.ndarray)
                assert result.shape == (2, 1024)

    @pytest.mark.asyncio
    async def test_embed_async_empty_input(self, mock_async_voyage_client):
        """Test embed_async with empty input."""
        with patch("voyageai.AsyncClient", return_value=mock_async_voyage_client):
            with patch("voyageai.Client"):
                from hybridrag.integrations.voyage import VoyageEmbedder

                embedder = VoyageEmbedder(api_key="test-key")

                result = await embedder.embed_async([])

                assert isinstance(result, np.ndarray)
                assert result.size == 0


class TestVoyageReranker:
    """Tests for VoyageReranker class."""

    @pytest.fixture
    def mock_rerank_result(self):
        """Create mock rerank result."""
        result1 = MagicMock()
        result1.index = 1
        result1.document = "doc2"
        result1.relevance_score = 0.95

        result2 = MagicMock()
        result2.index = 0
        result2.document = "doc1"
        result2.relevance_score = 0.85

        return MagicMock(results=[result1, result2])

    @pytest.mark.asyncio
    async def test_rerank_async_returns_list(self, mock_rerank_result):
        """Test that rerank_async returns list of dicts."""
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(return_value=mock_rerank_result)

        with patch("voyageai.AsyncClient", return_value=mock_client):
            with patch("voyageai.Client"):
                from hybridrag.integrations.voyage import VoyageReranker

                reranker = VoyageReranker(api_key="test-key")
                reranker._async_client = mock_client

                result = await reranker.rerank_async("query", ["doc1", "doc2"])

                assert isinstance(result, list)
                assert len(result) == 2
                assert result[0]["relevance_score"] == 0.95
                assert result[0]["index"] == 1

    @pytest.mark.asyncio
    async def test_rerank_async_empty_documents(self):
        """Test rerank_async with empty documents."""
        with patch("voyageai.AsyncClient"):
            with patch("voyageai.Client"):
                from hybridrag.integrations.voyage import VoyageReranker

                reranker = VoyageReranker(api_key="test-key")

                result = await reranker.rerank_async("query", [])

                assert result == []


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_embedding_func(self):
        """Test create_embedding_func returns callable."""
        with patch("voyageai.AsyncClient"):
            with patch("voyageai.Client"):
                from hybridrag.integrations.voyage import create_embedding_func

                func = create_embedding_func(api_key="test-key")

                assert callable(func)

    def test_create_rerank_func(self):
        """Test create_rerank_func returns callable."""
        with patch("voyageai.AsyncClient"):
            with patch("voyageai.Client"):
                from hybridrag.integrations.voyage import create_rerank_func

                func = create_rerank_func(api_key="test-key")

                assert callable(func)
