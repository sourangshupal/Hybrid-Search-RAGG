"""
End-to-end test for HybridRAG.

Requirements:
- MONGODB_URI set to MongoDB Atlas cluster
- VOYAGE_API_KEY set
- One of: GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY
- Atlas Vector Search index created on {database}_vdb collection

Usage:
    python tests/test_e2e.py                    # Uses Gemini (default)
    python tests/test_e2e.py --llm anthropic    # Uses Claude
    python tests/test_e2e.py --llm openai       # Uses OpenAI
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybridrag import create_hybridrag, QueryParam, Settings


# Test document
TEST_DOCUMENT = """
MongoDB Atlas is a fully-managed cloud database service that provides vector search capabilities
through the $vectorSearch aggregation stage. It enables developers to build AI-powered applications
by combining traditional database operations with vector similarity search.

Key features of MongoDB Atlas Vector Search include:
1. Native vector indexing using HNSW algorithm
2. Support for multiple similarity metrics (cosine, euclidean, dotProduct)
3. Pre-filtering with MQL expressions
4. Integration with the aggregation pipeline

The $vectorSearch stage performs approximate nearest neighbor (ANN) search on vector embeddings.
It requires a vector search index to be created on the collection containing the embeddings.

Example vector search query:
{
  "$vectorSearch": {
    "index": "vector_index",
    "path": "embedding",
    "queryVector": [0.1, 0.2, ...],
    "numCandidates": 100,
    "limit": 10
  }
}

MongoDB Atlas also provides $graphLookup for graph traversal operations, which can be combined
with vector search for knowledge graph retrieval patterns.
"""

TEST_QUERIES = [
    "What is MongoDB Atlas Vector Search?",
    "How does $vectorSearch work?",
    "What similarity metrics does MongoDB support?",
]


async def test_initialization(llm_provider: str = "gemini"):
    """Test RAG initialization."""
    print(f"\n[TEST] Initializing HybridRAG with {llm_provider}...")

    try:
        # Create settings with specified LLM provider
        os.environ["LLM_PROVIDER"] = llm_provider

        rag = await create_hybridrag(
            working_dir="./hybridrag_workspace",
            auto_initialize=True,
        )
        print("[PASS] HybridRAG initialized successfully")

        # Print status
        status = await rag.get_status()
        print(f"[INFO] Status: {status}")

        return rag
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        raise


async def test_insert(rag):
    """Test document insertion."""
    print("\n[TEST] Inserting test document...")

    try:
        await rag.insert(TEST_DOCUMENT)
        print("[PASS] Document inserted successfully")
    except Exception as e:
        print(f"[FAIL] Insert failed: {e}")
        raise


async def test_query_modes(rag):
    """Test different query modes."""
    modes = ["naive", "local", "global", "hybrid", "mix"]

    for mode in modes:
        print(f"\n[TEST] Query mode: {mode}")
        try:
            response = await rag.query(
                TEST_QUERIES[0],
                mode=mode,
                enable_rerank=True,
            )
            print(f"[PASS] {mode} mode returned response ({len(response)} chars)")
            print(f"  Preview: {response[:200]}...")
        except Exception as e:
            print(f"[FAIL] {mode} mode failed: {e}")


async def test_reranking(rag):
    """Test with and without reranking."""
    print("\n[TEST] Testing reranking...")

    try:
        # Without reranking
        response_no_rerank = await rag.query(
            TEST_QUERIES[1],
            mode="mix",
            enable_rerank=False,
        )
        print(f"[INFO] Without rerank: {len(response_no_rerank)} chars")

        # With reranking
        response_rerank = await rag.query(
            TEST_QUERIES[1],
            mode="mix",
            enable_rerank=True,
        )
        print(f"[INFO] With rerank: {len(response_rerank)} chars")

        print("[PASS] Reranking test completed")
    except Exception as e:
        print(f"[FAIL] Reranking test failed: {e}")


async def test_quick_query(rag):
    """Quick query test without document insertion."""
    print("\n[TEST] Quick query test...")

    try:
        response = await rag.query("What is MongoDB?", mode="mix")
        print(f"[PASS] Query returned response ({len(response)} chars)")
        print(f"  Preview: {response[:150]}...")
    except Exception as e:
        print(f"[FAIL] Query failed: {e}")
        raise


async def run_all_tests(llm_provider: str = "gemini", quick: bool = False):
    """Run all tests."""
    print("=" * 60)
    print(f"HybridRAG End-to-End Test (LLM: {llm_provider})")
    print("=" * 60)

    # Check environment
    print("\n[CHECK] Environment variables:")
    base_vars = ["MONGODB_URI", "VOYAGE_API_KEY"]
    llm_vars = {
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_vars = base_vars + [llm_vars.get(llm_provider, "GEMINI_API_KEY")]

    for var in env_vars:
        status = "SET" if os.getenv(var) else "MISSING"
        print(f"  {var}: {status}")

    if not all(os.getenv(var) for var in env_vars):
        print("\n[ERROR] Missing required environment variables")
        return

    # Run tests
    rag = await test_initialization(llm_provider)

    if quick:
        await test_quick_query(rag)
    else:
        await test_insert(rag)
        await test_query_modes(rag)
        await test_reranking(rag)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybridRAG E2E Test")
    parser.add_argument(
        "--llm",
        choices=["gemini", "anthropic", "openai"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (query only, no insert)",
    )
    args = parser.parse_args()

    asyncio.run(run_all_tests(args.llm, args.quick))
