"""Tests for HybridRAG enhancements."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from unittest.mock import AsyncMock


class TestImplicitExpander:
    """Tests for ImplicitExpander."""

    @pytest.fixture
    def mock_embedding_func(self):
        """Create mock embedding function."""

        async def embed(texts):
            # Return random embeddings for testing
            return np.random.rand(len(texts), 1024).astype(np.float32)

        return embed

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            {"id": "entity1", "description": "MongoDB database system"},
            {"id": "entity2", "description": "Vector search capabilities"},
        ]

    @pytest.fixture
    def sample_entity_embeddings(self):
        """Sample entity embeddings."""
        np.random.seed(42)
        return {
            "entity1": ("MongoDB database system", np.random.rand(1024).tolist()),
            "entity2": ("Vector search capabilities", np.random.rand(1024).tolist()),
            "entity3": ("Document storage", np.random.rand(1024).tolist()),
            "entity4": ("Graph database", np.random.rand(1024).tolist()),
        }

    @pytest.mark.asyncio
    async def test_expand_from_query(
        self, mock_embedding_func, sample_entity_embeddings
    ):
        """Test expand_from_query returns dict with expected keys."""
        from hybridrag.enhancements.implicit_expansion import ImplicitExpander

        expander = ImplicitExpander(
            embedding_func=mock_embedding_func,
            similarity_threshold=0.0,  # Low threshold to get results
            max_expansions=5,
        )

        result = await expander.expand_from_query(
            query="What is MongoDB?",
            all_entity_embeddings=sample_entity_embeddings,
        )

        assert isinstance(result, dict)
        assert "query_implicit" in result
        assert "entity_implicit" in result
        assert isinstance(result["query_implicit"], list)

    @pytest.mark.asyncio
    async def test_expand_from_entities(
        self, mock_embedding_func, sample_entities, sample_entity_embeddings
    ):
        """Test expand_from_entities returns list."""
        from hybridrag.enhancements.implicit_expansion import ImplicitExpander

        expander = ImplicitExpander(
            embedding_func=mock_embedding_func,
            similarity_threshold=0.0,
            max_expansions=5,
        )

        result = await expander.expand_from_entities(
            explicit_entities=sample_entities,
            all_entity_embeddings=sample_entity_embeddings,
            exclude_ids={"entity1", "entity2"},
        )

        assert isinstance(result, list)
        # Should return entities not in exclude_ids
        for entity in result:
            assert entity["id"] not in {"entity1", "entity2"}


class TestEntityBoostingReranker:
    """Tests for EntityBoostingReranker."""

    @pytest.fixture
    def mock_base_rerank(self):
        """Create mock base rerank function."""

        async def rerank(query, documents, top_n=10):
            return [
                {"index": i, "document": doc, "relevance_score": 0.9 - (i * 0.1)}
                for i, doc in enumerate(documents[:top_n])
            ]

        return rerank

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks with entity IDs."""
        return [
            {"content": "Chunk about MongoDB", "entity_ids": ["mongodb", "database"]},
            {"content": "Chunk about vectors", "entity_ids": ["vector", "search"]},
            {"content": "Chunk about graphs", "entity_ids": ["graph", "nodes"]},
        ]

    @pytest.mark.asyncio
    async def test_rerank_with_boost(self, mock_base_rerank, sample_chunks):
        """Test rerank_with_boost adds entity boost."""
        from hybridrag.enhancements.entity_boosting import EntityBoostingReranker

        reranker = EntityBoostingReranker(
            base_rerank_func=mock_base_rerank,
            boost_weight=0.2,
        )

        result = await reranker.rerank_with_boost(
            query="MongoDB query",
            chunks=sample_chunks,
            relevant_entity_ids={"mongodb", "database"},
            top_n=10,
        )

        assert isinstance(result, list)
        # First chunk should have entity boost
        assert result[0]["entity_overlap"] >= 0
        assert "final_score" in result[0]

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self, mock_base_rerank):
        """Test rerank with empty chunks."""
        from hybridrag.enhancements.entity_boosting import EntityBoostingReranker

        reranker = EntityBoostingReranker(
            base_rerank_func=mock_base_rerank,
            boost_weight=0.2,
        )

        result = await reranker.rerank_with_boost(
            query="test",
            chunks=[],
            relevant_entity_ids=set(),
            top_n=10,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_callable_interface(self, mock_base_rerank):
        """Test callable interface with string documents."""
        from hybridrag.enhancements.entity_boosting import EntityBoostingReranker

        reranker = EntityBoostingReranker(
            base_rerank_func=mock_base_rerank,
            boost_weight=0.2,
        )

        # Test with string documents (no entity boosting)
        result = await reranker(
            query="test",
            documents=["doc1", "doc2", "doc3"],
            top_n=10,
        )

        assert isinstance(result, list)
        assert len(result) == 3


class TestCreateBoostedRerankFunc:
    """Tests for create_boosted_rerank_func."""

    def test_returns_callable(self):
        """Test that factory returns a callable function."""
        from hybridrag.enhancements.entity_boosting import create_boosted_rerank_func

        async def mock_rerank(query, documents, top_n=10):
            return []

        result = create_boosted_rerank_func(mock_rerank, boost_weight=0.3)

        # Now returns a callable function, not EntityBoostingReranker instance
        assert callable(result)

    @pytest.mark.asyncio
    async def test_factory_function_works(self):
        """Test that the factory-created function works correctly."""
        from hybridrag.enhancements.entity_boosting import create_boosted_rerank_func

        async def mock_rerank(query, documents, top_n=10):
            return [
                {"index": i, "document": doc, "relevance_score": 0.9 - (i * 0.1)}
                for i, doc in enumerate(documents[:top_n])
            ]

        boosted_func = create_boosted_rerank_func(mock_rerank, boost_weight=0.3)

        result = await boosted_func(
            query="test query",
            documents=["doc1", "doc2", "doc3"],
            top_n=10,
        )

        assert isinstance(result, list)
        assert len(result) == 3
