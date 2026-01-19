#!/usr/bin/env python3
"""
HybridRAG Mix Mode Search Example
=================================

This example demonstrates mix mode search capabilities:
1. Graph traversal configuration
2. Mix mode search execution
3. Result analysis with source breakdown
4. Entity-only graph search

Prerequisites:
    pip install mongodb-hybridrag[all]

Run:
    python examples/04_mix_mode_search.py
"""

import asyncio

# Add src to path for development
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybridrag.enhancements import (
    GraphTraversalConfig,
    MixModeConfig,
    MixModeSearchResult,
    MongoDBHybridSearchConfig,
)


async def demo_graph_config() -> None:
    """Demonstrate graph traversal configuration."""
    print("\n" + "=" * 60)
    print("1. Graph Traversal Configuration")
    print("=" * 60)

    # Default configuration
    default_config = GraphTraversalConfig()
    print("\n  Default GraphTraversalConfig:")
    print(f"    edges_collection: {default_config.edges_collection}")
    print(f"    chunks_collection: {default_config.chunks_collection}")
    print(f"    entities_collection: {default_config.entities_collection}")
    print(f"    max_depth: {default_config.max_depth}")
    print(f"    max_nodes: {default_config.max_nodes}")

    # Custom configuration
    custom_config = GraphTraversalConfig(
        edges_collection="knowledge_edges",
        chunks_collection="document_chunks",
        entities_collection="knowledge_entities",
        max_depth=3,
        max_nodes=100,
        workspace="my_workspace",
    )
    print("\n  Custom GraphTraversalConfig:")
    print(f"    edges_collection: {custom_config.edges_collection}")
    print(f"    max_depth: {custom_config.max_depth}")
    print(f"    workspace: {custom_config.workspace}")


async def demo_mix_mode_config() -> None:
    """Demonstrate mix mode configuration."""
    print("\n" + "=" * 60)
    print("2. Mix Mode Configuration")
    print("=" * 60)

    # Configure hybrid search
    hybrid_config = MongoDBHybridSearchConfig(
        vector_weight=0.6,
        text_weight=0.4,
        fuzzy_max_edits=2,
        cosine_threshold=0.3,
    )

    # Configure graph traversal
    graph_config = GraphTraversalConfig(
        max_depth=2,
        max_nodes=50,
    )

    # Configure mix mode
    config = MixModeConfig(
        hybrid_config=hybrid_config,
        graph_config=graph_config,
        enable_graph_traversal=True,
        enable_entity_boosting=True,
        enable_reranking=True,
        entity_boost_weight=0.2,
        entity_only_weight=0.5,
    )

    print("\n  MixModeConfig:")
    print(f"    enable_graph_traversal: {config.enable_graph_traversal}")
    print(f"    enable_entity_boosting: {config.enable_entity_boosting}")
    print(f"    entity_boost_weight: {config.entity_boost_weight}")
    print(f"    entity_only_weight: {config.entity_only_weight}")
    print("\n  Hybrid weights:")
    print(f"    vector_weight: {config.hybrid_config.vector_weight}")
    print(f"    text_weight: {config.hybrid_config.text_weight}")


async def demo_result_structure() -> None:
    """Demonstrate mix mode result structure."""
    print("\n" + "=" * 60)
    print("3. Mix Mode Result Structure")
    print("=" * 60)

    # Create a sample result
    sample_result = MixModeSearchResult(
        chunk_id="507f1f77bcf86cd799439011",
        document_id="507f1f77bcf86cd799439012",
        content="MongoDB Atlas provides vector search capabilities...",
        score=0.85,
        metadata={"source": "mongodb-docs.pdf", "page": 42},
        search_type="mix_mode",
        source_scores={
            "vector": 0.92,
            "text": 0.78,
            "entity": 0.65,
        },
        graph_entities=["MongoDB", "Atlas", "vector search"],
        entity_boost=0.13,
        document_title="MongoDB Documentation",
        document_source="mongodb-docs.pdf",
    )

    print("\n  MixModeSearchResult fields:")
    print(f"    chunk_id: {sample_result.chunk_id}")
    print(f"    score: {sample_result.score}")
    print(f"    search_type: {sample_result.search_type}")
    print(f"    content: {sample_result.content[:50]}...")

    print("\n  Source score breakdown:")
    for source, score in sample_result.source_scores.items():
        print(f"    {source}: {score}")

    print("\n  Graph metadata:")
    print(f"    graph_entities: {sample_result.graph_entities}")
    print(f"    entity_boost: {sample_result.entity_boost}")


async def demo_search_patterns() -> None:
    """Demonstrate common search patterns."""
    print("\n" + "=" * 60)
    print("4. Common Search Patterns")
    print("=" * 60)

    print("\n  Pattern 1: Basic mix mode search")
    print(
        """
    from hybridrag.enhancements import mix_mode_search

    results = await mix_mode_search(
        db=db,
        query="How does MongoDB handle vector search?",
        query_vector=embedding,
        top_k=10,
    )
    """
    )

    print("\n  Pattern 2: With entity extraction")
    print(
        """
    results = await mix_mode_search(
        db=db,
        query="How does MongoDB handle vector search?",
        query_vector=embedding,
        top_k=10,
        query_entities=["MongoDB", "vector search"],  # Pre-extracted
    )
    """
    )

    print("\n  Pattern 3: Using MixModeSearcher class")
    print(
        """
    from hybridrag.enhancements import create_mix_mode_searcher

    searcher = await create_mix_mode_searcher(
        db=db,
        vector_weight=0.6,
        text_weight=0.4,
        entity_boost_weight=0.2,
    )

    results = await searcher.search(
        query="...",
        query_vector=embedding,
        top_k=10,
    )
    """
    )

    print("\n  Pattern 4: Graph-only search")
    print(
        """
    results = await searcher.search_with_graph_only(
        query_entities=["MongoDB", "Atlas"],
        top_k=10,
    )
    """
    )


async def demo_analysis() -> None:
    """Demonstrate result analysis patterns."""
    print("\n" + "=" * 60)
    print("5. Result Analysis")
    print("=" * 60)

    # Simulate some results
    results = [
        MixModeSearchResult(
            chunk_id="1",
            content="MongoDB Atlas provides...",
            score=0.85,
            source_scores={"vector": 0.9, "text": 0.8, "entity": 0.5},
            search_type="mix_mode",
            graph_entities=["MongoDB", "Atlas"],
            entity_boost=0.1,
        ),
        MixModeSearchResult(
            chunk_id="2",
            content="Vector search enables...",
            score=0.72,
            source_scores={"vector": 0.7, "text": 0.6, "entity": 0.8},
            search_type="entity_only",
            graph_entities=["vector search"],
            entity_boost=0.16,
        ),
    ]

    print("\n  Analyzing search results:")
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Combined score: {r.score:.2f}")
        print(f"    Search type: {r.search_type}")

        # Analyze source contribution
        total = sum(r.source_scores.values())
        if total > 0:
            print("    Source contributions:")
            for source, score in r.source_scores.items():
                pct = (score / total) * 100
                print(f"      {source}: {score:.2f} ({pct:.1f}%)")

        print(f"    Entity boost applied: {r.entity_boost:.2f}")

    # Aggregate statistics
    print("\n  Aggregate statistics:")
    avg_score = sum(r.score for r in results) / len(results)
    entity_only_count = sum(1 for r in results if r.search_type == "entity_only")
    all_entities = set()
    for r in results:
        all_entities.update(r.graph_entities)

    print(f"    Average score: {avg_score:.2f}")
    print(f"    Entity-only results: {entity_only_count}/{len(results)}")
    print(f"    Unique entities discovered: {len(all_entities)}")
    print(f"    Entities: {', '.join(all_entities)}")


async def main() -> None:
    """Run all mix mode search demos."""
    print("\n" + "=" * 60)
    print("HybridRAG Mix Mode Search Examples")
    print("=" * 60)

    await demo_graph_config()
    await demo_mix_mode_config()
    await demo_result_structure()
    await demo_search_patterns()
    await demo_analysis()

    print("\n" + "=" * 60)
    print("✓ All mix mode search demos complete!")
    print("=" * 60)
    print(
        """
Usage in your code:

    from hybridrag.enhancements import (
        mix_mode_search,
        create_mix_mode_searcher,
        MixModeConfig,
        GraphTraversalConfig,
    )

    # Quick usage
    results = await mix_mode_search(
        db=db,
        query="Your question here",
        query_vector=embedding,
        query_entities=["extracted", "entities"],
    )

    # Or use the searcher class
    searcher = await create_mix_mode_searcher(db=db)
    results = await searcher.search(query, query_vector, top_k=10)
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
