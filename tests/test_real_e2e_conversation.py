"""
Real E2E Test: 20-Turn Human-like Conversation

This test uses REAL API calls to validate the complete HybridRAG pipeline:
1. Document ingestion with Voyage AI embeddings
2. Knowledge graph extraction with Claude
3. 20-turn conversation with memory (like a real chatbot)

Requires environment variables:
- MONGODB_URI
- VOYAGE_API_KEY
- ANTHROPIC_API_KEY
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

# Load environment variables from .env file (project root)
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded .env from {env_file}")
else:
    print(f"⚠ No .env file found at {env_file}")
    load_dotenv()

# Credentials loaded from .env file - DO NOT hardcode
# Use Gemini as LLM provider
if not os.environ.get("LLM_PROVIDER"):
    os.environ["LLM_PROVIDER"] = "gemini"
if not os.environ.get("GEMINI_MODEL"):
    os.environ["GEMINI_MODEL"] = "gemini-2.5-flash-lite"

# Add src to path
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "lightrag-fork"))

from hybridrag import HybridRAG, create_hybridrag


# Test document - MongoDB Atlas Vector Search (real technical content)
TEST_DOCUMENT = """
# MongoDB Atlas Vector Search: Complete Guide

## Introduction
MongoDB Atlas Vector Search is a fully managed service that enables semantic search
capabilities on your data stored in MongoDB Atlas. It uses vector embeddings to find
documents that are semantically similar to a query, rather than relying on exact keyword matches.

## Key Concepts

### Vector Embeddings
Vector embeddings are numerical representations of data (text, images, audio) in a
high-dimensional space. Similar items are positioned close together in this space.
For text, embeddings capture semantic meaning - words like "happy" and "joyful"
would have similar vector representations.

Common embedding dimensions:
- OpenAI text-embedding-3-small: 1536 dimensions
- Voyage AI voyage-3-large: 1024 dimensions
- Cohere embed-english-v3: 1024 dimensions

### Vector Search Index
Before performing vector search, you must create a vector search index on your collection.
The index specifies:
- The field containing the vector embeddings
- The number of dimensions
- The similarity metric (cosine, euclidean, or dotProduct)

Example index definition:
```json
{
  "type": "vectorSearch",
  "fields": [{
    "path": "embedding",
    "numDimensions": 1024,
    "similarity": "cosine"
  }]
}
```

### $vectorSearch Aggregation Stage
The $vectorSearch stage is used in aggregation pipelines to perform vector search.
Key parameters:
- index: Name of the vector search index
- path: Field containing the vector
- queryVector: The query embedding (array of floats)
- numCandidates: Number of candidates to consider (should be 10-20x limit)
- limit: Maximum number of results to return

## Architecture

### How It Works
1. Documents are stored with their vector embeddings in a field
2. Atlas builds an approximate nearest neighbor (ANN) index
3. At query time, the query is converted to a vector
4. The ANN index efficiently finds similar vectors
5. Results are returned sorted by similarity score

### Hierarchical Navigable Small World (HNSW)
Atlas Vector Search uses the HNSW algorithm for efficient similarity search.
HNSW creates a multi-layer graph structure that allows logarithmic search complexity.
This enables searching billions of vectors in milliseconds.

## Best Practices

### Embedding Quality
- Use domain-appropriate embedding models
- For technical content, consider code-specific models
- Voyage AI voyage-3-large offers excellent quality for general text
- Always use the same model for indexing and querying

### Index Configuration
- Set numCandidates to at least 10x your limit
- Use cosine similarity for normalized embeddings
- Consider creating separate indexes for different embedding types

### Hybrid Search
Combine vector search with traditional filters for better results:
```javascript
{
  $vectorSearch: {
    index: "vector_index",
    path: "embedding",
    queryVector: [...],
    numCandidates: 100,
    limit: 10,
    filter: { category: "technical" }
  }
}
```

### Performance Optimization
- Pre-filter when possible to reduce search space
- Use appropriate numCandidates (higher = more accurate, slower)
- Consider sharding for very large collections
- Monitor query latency and adjust parameters

## Integration with RAG Systems

### Retrieval-Augmented Generation
Vector search is a key component of RAG (Retrieval-Augmented Generation) systems:
1. User asks a question
2. Question is converted to embedding
3. Vector search finds relevant documents
4. Documents are passed to LLM as context
5. LLM generates answer grounded in retrieved content

### Reranking
For better accuracy, use a reranker after initial retrieval:
1. Vector search retrieves top-100 candidates
2. Reranker (e.g., Voyage rerank-2.5) scores each candidate
3. Top-10 after reranking are used for generation

This two-stage approach balances speed and accuracy.

## Pricing and Limits
- Vector search is included in Atlas pricing
- No additional cost for vector search queries
- Index storage counts toward cluster storage
- Recommended: M10+ clusters for production workloads

## Common Issues

### Dimension Mismatch
Error: "vector dimension mismatch"
Solution: Ensure query vector dimensions match index dimensions exactly.

### Index Not Found
Error: "index not found"
Solution: Verify index name and that index creation completed successfully.

### Slow Queries
Cause: numCandidates too high or collection too large
Solution: Reduce numCandidates, add pre-filters, or scale cluster.
"""


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_number: int
    user_query: str
    assistant_response: str
    response_time: float
    context_length: int
    history_used: int


@dataclass
class TestResults:
    """Track test results."""
    turns: list[ConversationTurn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        print(f"\n{'='*60}")
        print(f"TURN {turn.turn_number}")
        print(f"{'='*60}")
        print(f"USER: {turn.user_query}")
        print(f"\nASSISTANT: {turn.assistant_response[:500]}{'...' if len(turn.assistant_response) > 500 else ''}")
        print(f"\n[Time: {turn.response_time:.2f}s | Context: {turn.context_length} chars | History: {turn.history_used} msgs]")

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        print(f"\n❌ ERROR: {error}")

    def summary(self) -> None:
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total turns: {len(self.turns)}")
        print(f"Errors: {len(self.errors)}")
        if self.turns:
            avg_time = sum(t.response_time for t in self.turns) / len(self.turns)
            print(f"Avg response time: {avg_time:.2f}s")
        if self.errors:
            print("\nErrors encountered:")
            for e in self.errors:
                print(f"  - {e}")


# 20 realistic conversation turns - like a human exploring a topic
CONVERSATION_TURNS = [
    # Initial exploration
    "What is MongoDB Atlas Vector Search?",

    # Follow-up for more detail
    "How do vector embeddings work?",

    # Clarification request
    "What do you mean by 'high-dimensional space'?",

    # Specific technical question
    "What embedding dimensions does Voyage AI use?",

    # Comparison question
    "How does Voyage compare to OpenAI embeddings?",

    # Practical question
    "How do I create a vector search index?",

    # Reference to earlier topic
    "You mentioned cosine similarity earlier - what is that?",

    # Deep dive
    "Explain the HNSW algorithm in simple terms",

    # Best practices
    "What are the best practices for vector search?",

    # Specific parameter question
    "What should numCandidates be set to?",

    # Use case question
    "How is vector search used in RAG systems?",

    # Follow-up on RAG
    "What is reranking and why is it important?",

    # Reference to context
    "Which reranker did you mention works well?",

    # Performance question
    "How can I optimize vector search performance?",

    # Troubleshooting
    "What causes dimension mismatch errors?",

    # Pricing question
    "Is vector search free with Atlas?",

    # Architecture question
    "How does the ANN index work internally?",

    # Practical implementation
    "Show me an example of hybrid search",

    # Summary request
    "Summarize the key points about Atlas Vector Search",

    # Final clarification
    "What's the most important thing to remember when implementing vector search?",
]


async def run_conversation_test() -> TestResults:
    """Run the full 20-turn conversation test."""
    results = TestResults()

    print("\n" + "="*60)
    print("REAL E2E TEST: 20-Turn Conversation")
    print("="*60)

    # Check environment (using Gemini LLM)
    required_vars = ["MONGODB_URI", "VOYAGE_API_KEY", "GEMINI_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        results.add_error(f"Missing environment variables: {missing}")
        return results

    print("\n✓ Environment variables present")

    # Initialize HybridRAG
    print("\n[INIT] Initializing HybridRAG...")
    try:
        rag = await create_hybridrag(
            working_dir="./test_conversation_workspace",
            auto_initialize=True,
        )
        print("✓ HybridRAG initialized")
    except Exception as e:
        results.add_error(f"Failed to initialize: {e}")
        return results

    # Ingest test document
    print("\n[INGEST] Ingesting test document...")
    try:
        start = time.time()
        await rag.insert(TEST_DOCUMENT, file_paths=["mongodb_vector_search_guide.md"])
        ingest_time = time.time() - start
        print(f"✓ Document ingested in {ingest_time:.2f}s")
    except Exception as e:
        results.add_error(f"Failed to ingest: {e}")
        return results

    # Wait for indexing
    print("\n[WAIT] Waiting for indexing...")
    await asyncio.sleep(3)

    # Create conversation session
    session_id = f"test_session_{int(time.time())}"
    print(f"\n[SESSION] Created session: {session_id}")

    # Run 20-turn conversation
    print("\n[CONVERSATION] Starting 20-turn conversation...")

    for i, query in enumerate(CONVERSATION_TURNS, 1):
        try:
            start = time.time()

            # Use query_with_memory for conversation continuity
            response = await rag.query_with_memory(
                query=query,
                session_id=session_id,
                mode="mix",  # Use mix mode (KG + vector)
                top_k=60,
                max_history_messages=10,
            )

            elapsed = time.time() - start

            turn = ConversationTurn(
                turn_number=i,
                user_query=query,
                assistant_response=response.get("answer", "No response"),
                response_time=elapsed,
                context_length=len(response.get("context", "")),
                history_used=response.get("history_used", 0),
            )
            results.add_turn(turn)

            # Small delay between turns to be realistic
            await asyncio.sleep(0.5)

        except Exception as e:
            results.add_error(f"Turn {i} failed: {e}")
            # Continue with remaining turns
            continue

    # Print summary
    results.summary()

    return results


async def main():
    """Main entry point."""
    print("\n" + "#"*60)
    print("# HybridRAG Real E2E Test")
    print("# 20-Turn Human-like Conversation")
    print("#"*60)

    results = await run_conversation_test()

    # Exit with appropriate code
    if results.errors:
        print(f"\n❌ Test completed with {len(results.errors)} errors")
        sys.exit(1)
    elif len(results.turns) < 20:
        print(f"\n⚠ Test incomplete: only {len(results.turns)}/20 turns completed")
        sys.exit(1)
    else:
        print(f"\n✓ Test completed successfully: {len(results.turns)} turns")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
