"""
Integration test for full RAG pipeline.

Tests end-to-end functionality with real MongoDB connection.
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_and_query(rag, test_documents):
    """Test document ingestion and querying."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"], metadata=doc["metadata"])

    # Query
    results = await rag.query(
        query="What is MongoDB Atlas?",
        mode="hybrid",
        top_k=5,
    )

    # Verify
    assert len(results) > 0, "Should return results"
    assert results[0].content is not None, "Should have content"
    assert results[0].score > 0, "Should have positive score"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_with_answer(rag, test_documents):
    """Test query with LLM answer generation."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"], metadata=doc["metadata"])

    # Query with answer
    answer = await rag.query_with_answer(
        query="What is vector search?",
        mode="hybrid",
        top_k=3,
    )

    # Verify
    assert isinstance(answer, str), "Should return string answer"
    assert len(answer) > 0, "Answer should not be empty"
    assert "vector" in answer.lower() or "search" in answer.lower(), (
        "Answer should be relevant"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_modes(rag, test_documents):
    """Test all search modes."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"], metadata=doc["metadata"])

    query = "MongoDB search features"

    # Test each mode
    for mode in ["vector", "keyword", "hybrid"]:
        results = await rag.query(query=query, mode=mode, top_k=3)
        assert len(results) > 0, f"{mode} mode should return results"
        assert all(r.score > 0 for r in results), (
            f"{mode} mode should have positive scores"
        )
