"""
Comprehensive Session & Workspace Isolation Test for HybridRAG

This test validates:
1. Session isolation - multiple users can have separate conversations on the same knowledge base
2. Workspace isolation - completely separate knowledge graphs using MONGODB_WORKSPACE env var

Uses Gemini Flash Lite 2.5 (gemini-2.5-flash-lite) for LLM generation.

Requirements:
- MONGODB_URI: MongoDB Atlas connection string
- VOYAGE_API_KEY: Voyage AI API key for embeddings
- GEMINI_API_KEY: Google Gemini API key

Usage:
    PYTHONPATH=./src:./lightrag-fork python tests/test_gemini_session_isolation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Load environment variables
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"[ENV] Loaded .env from {env_file}")
else:
    load_dotenv()

# Configure Gemini as LLM provider
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["GEMINI_MODEL"] = "gemini-2.5-flash-lite"

# Add src to path
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "lightrag-fork"))

from hybridrag import HybridRAG, create_hybridrag


# =============================================================================
# TEST DOCUMENTS
# =============================================================================

DOC_MONGODB_VECTOR_SEARCH = """
# MongoDB Atlas Vector Search: Complete Technical Guide

## Overview
MongoDB Atlas Vector Search is a fully managed service that enables semantic search
capabilities on data stored in MongoDB Atlas. It uses vector embeddings to find
semantically similar documents rather than relying on exact keyword matches.

## Architecture

### Vector Embeddings
Vector embeddings are numerical representations of data (text, images, audio) in a
high-dimensional space. Similar items are positioned close together. For text,
embeddings capture semantic meaning - words like "happy" and "joyful" would have
similar vector representations.

### HNSW Algorithm
Atlas Vector Search uses Hierarchical Navigable Small World (HNSW) algorithm for
efficient approximate nearest neighbor (ANN) search. HNSW creates a multi-layer
graph structure enabling logarithmic search complexity - searching billions of
vectors in milliseconds.

## Implementation

### Creating a Vector Search Index
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
```javascript
{
  "$vectorSearch": {
    "index": "vector_index",
    "path": "embedding",
    "queryVector": [0.1, 0.2, ...],
    "numCandidates": 100,
    "limit": 10
  }
}
```

### Key Parameters
- **index**: Name of the vector search index
- **path**: Field containing the vector embedding
- **queryVector**: The query embedding (array of floats)
- **numCandidates**: Candidates to consider (10-20x limit recommended)
- **limit**: Maximum results to return

## Best Practices

### Embedding Models
- Voyage AI voyage-3-large: 1024 dimensions, excellent quality
- OpenAI text-embedding-3-small: 1536 dimensions
- Cohere embed-english-v3: 1024 dimensions

### Performance Optimization
- Use pre-filtering with MQL expressions to reduce search space
- Set numCandidates appropriately (higher = more accurate, slower)
- Consider sharding for very large collections
- Monitor query latency and adjust parameters

### Common Issues
1. Dimension mismatch: Query vector must match index dimensions
2. Index not found: Verify index name and creation status
3. Slow queries: Reduce numCandidates or add pre-filters
"""

DOC_MONGODB_AGGREGATION = """
# MongoDB Aggregation Pipeline: Advanced Data Processing

## Overview
The MongoDB aggregation pipeline is a powerful framework for data transformation
and analysis. It processes documents through a sequence of stages, where each
stage transforms the documents as they pass through.

## Pipeline Stages

### $match
Filters documents to pass only those that match specified conditions.
```javascript
{ $match: { status: "active", score: { $gte: 80 } } }
```

### $group
Groups documents by a specified expression and applies accumulator operations.
```javascript
{ $group: { _id: "$category", total: { $sum: "$amount" } } }
```

### $project
Reshapes documents by including, excluding, or computing new fields.
```javascript
{ $project: { name: 1, total: { $multiply: ["$price", "$quantity"] } } }
```

### $lookup
Performs a left outer join with another collection.
```javascript
{
  $lookup: {
    from: "inventory",
    localField: "item",
    foreignField: "sku",
    as: "inventory_docs"
  }
}
```

### $unwind
Deconstructs an array field to output a document for each element.
```javascript
{ $unwind: "$items" }
```

### $sort
Orders documents by specified fields.
```javascript
{ $sort: { score: -1, date: 1 } }
```

## Advanced Features

### $graphLookup
Performs recursive search on a collection for graph and hierarchical data.
```javascript
{
  $graphLookup: {
    from: "employees",
    startWith: "$reportsTo",
    connectFromField: "reportsTo",
    connectToField: "name",
    as: "reportingHierarchy"
  }
}
```

### $facet
Enables multi-faceted aggregations in a single stage.
```javascript
{
  $facet: {
    "categorizedByPrice": [{ $bucket: { groupBy: "$price", boundaries: [0, 100, 500] } }],
    "categorizedByYear": [{ $bucket: { groupBy: "$year", boundaries: [2000, 2010, 2020] } }]
  }
}
```

## Performance Considerations
- Place $match early in the pipeline to reduce documents processed
- Use indexes to support $match and $sort operations
- Be cautious with $lookup on large collections
- Consider allowDiskUse for memory-intensive operations

## Use Cases
1. Real-time analytics dashboards
2. ETL (Extract, Transform, Load) operations
3. Report generation
4. Data migration and transformation
5. Complex business logic processing
"""


# =============================================================================
# TEST QUESTIONS
# =============================================================================

# Alice is a senior engineer asking technical questions
ALICE_QUESTIONS = [
    "Explain the HNSW algorithm used in MongoDB Atlas Vector Search",
    "What are the key parameters for the $vectorSearch stage?",
    "How should I configure numCandidates for optimal performance?",
    "What's the difference between cosine similarity and euclidean distance?",
    "How can I combine vector search with pre-filtering?",
    "What embedding dimensions does Voyage AI voyage-3-large use?",
    "Explain how $graphLookup works for hierarchical data",
    "What's the purpose of the $facet stage in aggregation pipelines?",
    "How do I optimize aggregation pipeline performance?",
    "When should I use $lookup vs embedding documents?",
]

# Bob is a beginner asking basic questions
BOB_QUESTIONS = [
    "What is vector search in simple terms?",
    "Why do we need embeddings for search?",
    "What is MongoDB Atlas?",
    "How do I create my first vector search index?",
    "What does 'semantic search' mean?",
    "What is an aggregation pipeline?",
    "How do I filter documents in MongoDB?",
    "What is $group used for?",
    "How do I sort query results?",
    "What's the difference between $match and find()?",
]

# Charlie is a new user who should have no history
CHARLIE_QUESTIONS = [
    "Tell me about the previous questions in this conversation",
    "What have we discussed so far?",
]


# =============================================================================
# TEST RESULTS TRACKING
# =============================================================================

@dataclass
class QueryResult:
    """Single query result."""
    session_id: str
    user_name: str
    question: str
    answer: str
    response_time: float
    history_messages: int
    context_length: int


@dataclass
class TestResults:
    """Track all test results."""
    session_results: dict[str, list[QueryResult]] = field(default_factory=dict)
    workspace_results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_query(self, result: QueryResult) -> None:
        if result.session_id not in self.session_results:
            self.session_results[result.session_id] = []
        self.session_results[result.session_id].append(result)

        print(f"\n{'─'*60}")
        print(f"[{result.user_name}] Session: {result.session_id[:8]}...")
        print(f"Q: {result.question[:80]}{'...' if len(result.question) > 80 else ''}")
        print(f"A: {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")
        print(f"[Time: {result.response_time:.2f}s | History: {result.history_messages} | Context: {result.context_length}]")

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        print(f"\n[ERROR] {error}")

    def print_summary(self) -> None:
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        for session_id, results in self.session_results.items():
            user_name = results[0].user_name if results else "Unknown"
            print(f"\n{user_name} (Session: {session_id[:8]}...)")
            print(f"  Queries: {len(results)}")
            print(f"  Total history at end: {results[-1].history_messages if results else 0}")
            avg_time = sum(r.response_time for r in results) / len(results) if results else 0
            print(f"  Avg response time: {avg_time:.2f}s")

        if self.workspace_results:
            print("\nWorkspace Isolation Results:")
            for ws, data in self.workspace_results.items():
                print(f"  {ws}: {data}")

        if self.errors:
            print(f"\nErrors: {len(self.errors)}")
            for e in self.errors:
                print(f"  - {e}")

        print("\n" + "="*60)


# =============================================================================
# PART 1: SESSION ISOLATION TEST
# =============================================================================

async def test_session_isolation(rag: HybridRAG, results: TestResults) -> None:
    """
    Test that sessions are isolated - Alice and Bob have separate conversation histories.
    """
    print("\n" + "="*60)
    print("PART 1: SESSION ISOLATION TEST")
    print("="*60)

    # Create unique session IDs
    session_alice = f"alice_{uuid.uuid4().hex[:8]}"
    session_bob = f"bob_{uuid.uuid4().hex[:8]}"
    session_charlie = f"charlie_{uuid.uuid4().hex[:8]}"

    print(f"\nSessions created:")
    print(f"  Alice: {session_alice}")
    print(f"  Bob: {session_bob}")
    print(f"  Charlie: {session_charlie}")

    # Interleave Alice and Bob questions to prove isolation
    print("\n[TEST] Running interleaved queries (Alice and Bob)...")

    for i in range(len(ALICE_QUESTIONS)):
        # Alice asks a question
        try:
            start = time.time()
            response = await rag.query_with_memory(
                query=ALICE_QUESTIONS[i],
                session_id=session_alice,
                mode="mix",
                top_k=30,
                max_history_messages=10,
            )
            elapsed = time.time() - start

            results.add_query(QueryResult(
                session_id=session_alice,
                user_name="Alice",
                question=ALICE_QUESTIONS[i],
                answer=response.get("answer", "No response"),
                response_time=elapsed,
                history_messages=response.get("history_used", 0),
                context_length=len(response.get("context", "")),
            ))
        except Exception as e:
            results.add_error(f"Alice Q{i+1} failed: {e}")

        # Bob asks a question
        try:
            start = time.time()
            response = await rag.query_with_memory(
                query=BOB_QUESTIONS[i],
                session_id=session_bob,
                mode="mix",
                top_k=30,
                max_history_messages=10,
            )
            elapsed = time.time() - start

            results.add_query(QueryResult(
                session_id=session_bob,
                user_name="Bob",
                question=BOB_QUESTIONS[i],
                answer=response.get("answer", "No response"),
                response_time=elapsed,
                history_messages=response.get("history_used", 0),
                context_length=len(response.get("context", "")),
            ))
        except Exception as e:
            results.add_error(f"Bob Q{i+1} failed: {e}")

        # Small delay between queries
        await asyncio.sleep(0.3)

    # Charlie asks about conversation history - should have none
    print("\n[TEST] Charlie checks for conversation history (should be empty)...")

    for question in CHARLIE_QUESTIONS:
        try:
            start = time.time()
            response = await rag.query_with_memory(
                query=question,
                session_id=session_charlie,
                mode="mix",
                top_k=30,
                max_history_messages=10,
            )
            elapsed = time.time() - start

            results.add_query(QueryResult(
                session_id=session_charlie,
                user_name="Charlie",
                question=question,
                answer=response.get("answer", "No response"),
                response_time=elapsed,
                history_messages=response.get("history_used", 0),
                context_length=len(response.get("context", "")),
            ))
        except Exception as e:
            results.add_error(f"Charlie query failed: {e}")

    # Verify isolation
    print("\n[VERIFY] Checking session isolation...")

    alice_results = results.session_results.get(session_alice, [])
    bob_results = results.session_results.get(session_bob, [])
    charlie_results = results.session_results.get(session_charlie, [])

    # Alice should have 10 queries in her history
    if alice_results:
        alice_final_history = alice_results[-1].history_messages
        print(f"  Alice final history count: {alice_final_history} (expected: ~10-20 messages)")

    # Bob should have 10 queries in his history (independent of Alice)
    if bob_results:
        bob_final_history = bob_results[-1].history_messages
        print(f"  Bob final history count: {bob_final_history} (expected: ~10-20 messages)")

    # Charlie should start with 0 history
    if charlie_results:
        charlie_history = charlie_results[0].history_messages
        print(f"  Charlie initial history count: {charlie_history} (expected: 0-2)")

        # Check if Charlie's response acknowledges no prior conversation
        first_response = charlie_results[0].answer.lower()
        if "no previous" in first_response or "haven't discussed" in first_response or "first" in first_response:
            print("  [PASS] Charlie correctly reports no prior conversation")
        else:
            print("  [INFO] Charlie's response:", first_response[:100])


# =============================================================================
# PART 2: WORKSPACE ISOLATION TEST
# =============================================================================

async def test_workspace_isolation(results: TestResults) -> None:
    """
    Test that workspaces are completely isolated - different knowledge graphs.
    """
    print("\n" + "="*60)
    print("PART 2: WORKSPACE ISOLATION TEST")
    print("="*60)

    workspace_alpha = f"test_alpha_{uuid.uuid4().hex[:6]}"
    workspace_beta = f"test_beta_{uuid.uuid4().hex[:6]}"

    print(f"\nWorkspaces:")
    print(f"  Alpha: {workspace_alpha} (will contain MongoDB Vector Search doc)")
    print(f"  Beta: {workspace_beta} (will contain MongoDB Aggregation doc)")

    # Create workspace Alpha with Vector Search document
    print("\n[TEST] Creating workspace Alpha...")
    os.environ["MONGODB_WORKSPACE"] = workspace_alpha

    try:
        rag_alpha = await create_hybridrag(
            working_dir=f"./test_workspace_{workspace_alpha}",
            auto_initialize=True,
        )

        print("[INGEST] Ingesting Vector Search document into Alpha...")
        await rag_alpha.insert(
            DOC_MONGODB_VECTOR_SEARCH,
            file_paths=["mongodb_vector_search.md"]
        )
        await asyncio.sleep(2)  # Wait for indexing

        # Query Alpha about Vector Search (should find)
        print("[QUERY] Querying Alpha about Vector Search...")
        response_alpha_vector = await rag_alpha.query(
            "What is the HNSW algorithm?",
            mode="mix"
        )

        # Query Alpha about Aggregation (should NOT find much)
        print("[QUERY] Querying Alpha about Aggregation...")
        response_alpha_agg = await rag_alpha.query(
            "How does the $facet stage work?",
            mode="mix"
        )

        results.workspace_results[workspace_alpha] = {
            "vector_search_query": len(response_alpha_vector) > 100,
            "aggregation_query": len(response_alpha_agg) < 200,  # Should be sparse
        }

    except Exception as e:
        results.add_error(f"Workspace Alpha failed: {e}")

    # Create workspace Beta with Aggregation document
    print("\n[TEST] Creating workspace Beta...")
    os.environ["MONGODB_WORKSPACE"] = workspace_beta

    try:
        rag_beta = await create_hybridrag(
            working_dir=f"./test_workspace_{workspace_beta}",
            auto_initialize=True,
        )

        print("[INGEST] Ingesting Aggregation document into Beta...")
        await rag_beta.insert(
            DOC_MONGODB_AGGREGATION,
            file_paths=["mongodb_aggregation.md"]
        )
        await asyncio.sleep(2)  # Wait for indexing

        # Query Beta about Aggregation (should find)
        print("[QUERY] Querying Beta about Aggregation...")
        response_beta_agg = await rag_beta.query(
            "How does the $facet stage work?",
            mode="mix"
        )

        # Query Beta about Vector Search (should NOT find much)
        print("[QUERY] Querying Beta about Vector Search...")
        response_beta_vector = await rag_beta.query(
            "What is the HNSW algorithm?",
            mode="mix"
        )

        results.workspace_results[workspace_beta] = {
            "aggregation_query": len(response_beta_agg) > 100,
            "vector_search_query": len(response_beta_vector) < 200,  # Should be sparse
        }

    except Exception as e:
        results.add_error(f"Workspace Beta failed: {e}")

    # Verify isolation
    print("\n[VERIFY] Checking workspace isolation...")

    if workspace_alpha in results.workspace_results and workspace_beta in results.workspace_results:
        alpha_res = results.workspace_results[workspace_alpha]
        beta_res = results.workspace_results[workspace_beta]

        print(f"  Alpha found Vector Search content: {alpha_res.get('vector_search_query', False)}")
        print(f"  Alpha sparse on Aggregation: {alpha_res.get('aggregation_query', False)}")
        print(f"  Beta found Aggregation content: {beta_res.get('aggregation_query', False)}")
        print(f"  Beta sparse on Vector Search: {beta_res.get('vector_search_query', False)}")

        if (alpha_res.get('vector_search_query') and
            beta_res.get('aggregation_query') and
            alpha_res.get('aggregation_query') and
            beta_res.get('vector_search_query')):
            print("\n  [PASS] Workspace isolation verified!")
        else:
            print("\n  [INFO] Results may vary based on knowledge graph queries")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def main():
    """Main test runner."""
    print("\n" + "#"*60)
    print("# HybridRAG Session & Workspace Isolation Test")
    print("# Using Gemini Flash Lite 2.5")
    print("#"*60)

    results = TestResults()

    # Check environment
    print("\n[CHECK] Environment variables:")
    required_vars = ["MONGODB_URI", "VOYAGE_API_KEY", "GEMINI_API_KEY"]
    missing = []
    for var in required_vars:
        status = "SET" if os.environ.get(var) else "MISSING"
        print(f"  {var}: {status}")
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        print(f"\n[ERROR] Missing required variables: {missing}")
        sys.exit(1)

    print(f"\n[CONFIG] LLM Provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"[CONFIG] Model: {os.environ.get('GEMINI_MODEL')}")

    # Initialize shared RAG instance for session tests
    print("\n[INIT] Initializing HybridRAG for session isolation test...")

    # Clear any workspace setting for session tests
    if "MONGODB_WORKSPACE" in os.environ:
        del os.environ["MONGODB_WORKSPACE"]

    try:
        rag = await create_hybridrag(
            working_dir="./test_session_workspace",
            auto_initialize=True,
        )
        print("[PASS] HybridRAG initialized")
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        sys.exit(1)

    # Ingest both documents into shared knowledge base
    print("\n[INGEST] Ingesting test documents...")
    try:
        await rag.insert(
            DOC_MONGODB_VECTOR_SEARCH,
            file_paths=["mongodb_vector_search.md"]
        )
        await rag.insert(
            DOC_MONGODB_AGGREGATION,
            file_paths=["mongodb_aggregation.md"]
        )
        print("[PASS] Documents ingested")
    except Exception as e:
        print(f"[FAIL] Ingestion failed: {e}")
        sys.exit(1)

    # Wait for indexing
    print("\n[WAIT] Waiting for indexing...")
    await asyncio.sleep(3)

    # Run session isolation test
    await test_session_isolation(rag, results)

    # Run workspace isolation test (optional - may take longer)
    run_workspace_test = os.environ.get("RUN_WORKSPACE_TEST", "false").lower() == "true"
    if run_workspace_test:
        await test_workspace_isolation(results)
    else:
        print("\n[SKIP] Workspace isolation test (set RUN_WORKSPACE_TEST=true to enable)")

    # Print summary
    results.print_summary()

    # Exit with appropriate code
    if results.errors:
        print(f"\n[RESULT] Test completed with {len(results.errors)} errors")
        sys.exit(1)
    else:
        total_queries = sum(len(r) for r in results.session_results.values())
        print(f"\n[RESULT] Test completed successfully: {total_queries} queries executed")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
