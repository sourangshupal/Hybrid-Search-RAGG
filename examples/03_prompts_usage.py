#!/usr/bin/env python3
"""
HybridRAG Prompts Usage Example
===============================

This example demonstrates the prompts module:
1. System prompts (full and compact)
2. Custom domain-specific prompts
3. Query-type detection and reranking
4. Entity extraction prompts
5. Memory and topic prompts

Prerequisites:
    pip install mongodb-hybridrag

Run:
    python examples/03_prompts_usage.py
"""

# Add src to path for development
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybridrag.prompts import (
    # Entity extraction
    ENTITY_EXTRACTION_PROMPT,
    # Memory
    MEMORY_SUMMARIZATION_PROMPT,
    QUERY_ENTITY_EXTRACTION_PROMPT,
    QUERY_TYPE_PATTERNS,
    SESSION_CONTEXT_PROMPT,
    # System prompts
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    # Topics
    TOPIC_EXTRACTION_PROMPT,
    QueryType,
    create_system_prompt,
    # Reranking
    detect_query_type,
    select_rerank_instruction,
)


def demo_system_prompts() -> None:
    """Demonstrate system prompt usage."""
    print("\n" + "=" * 60)
    print("1. System Prompts")
    print("=" * 60)

    print("\n  SYSTEM_PROMPT (Full version):")
    print(f"    Length: {len(SYSTEM_PROMPT)} characters")
    print(f"    Preview: {SYSTEM_PROMPT[:200]}...")

    print("\n  SYSTEM_PROMPT_COMPACT (Token-efficient):")
    print(f"    Length: {len(SYSTEM_PROMPT_COMPACT)} characters")
    print(f"    Preview: {SYSTEM_PROMPT_COMPACT[:200]}...")


def demo_custom_prompts() -> None:
    """Demonstrate custom domain-specific prompts."""
    print("\n" + "=" * 60)
    print("2. Custom Domain Prompts")
    print("=" * 60)

    # Medical research domain
    medical_prompt = create_system_prompt(
        domain="Medical Research",
        persona="Medical Research Assistant",
        language="English",
        response_style="comprehensive",
        entity_types=["disease", "drug", "gene", "protein", "symptom", "treatment"],
        custom_instructions="""
When discussing medications, always include:
- Generic and brand names
- Common dosages
- Contraindications
- FDA approval status
""",
    )

    print("\n  Medical Research Prompt:")
    print(f"    Length: {len(medical_prompt)} characters")
    print(f"    Preview: {medical_prompt[:300]}...")

    # Legal domain
    legal_prompt = create_system_prompt(
        domain="Legal Research",
        persona="Legal Research Analyst",
        language="English",
        response_style="concise",
        entity_types=["case", "statute", "regulation", "court", "judge", "party"],
    )

    print("\n  Legal Research Prompt:")
    print(f"    Length: {len(legal_prompt)} characters")


def demo_query_type_detection() -> None:
    """Demonstrate query type detection."""
    print("\n" + "=" * 60)
    print("3. Query Type Detection")
    print("=" * 60)

    test_queries = [
        "What is the capital of France?",
        "How do I implement a binary search tree?",
        "Error: ModuleNotFoundError: No module named 'pandas'",
        "Tell me about the history of machine learning",
        "Best practices for API design",
        "Why is my database query slow?",
    ]

    print("\n  Query Classification:")
    for query in test_queries:
        query_type = detect_query_type(query)
        instruction = select_rerank_instruction(query_type)

        print(f'\n  Q: "{query}"')
        print(f"     Type: {query_type.value}")
        print(f"     Rerank: {instruction[:80]}...")

    print("\n  Available Query Types:")
    for qt in QueryType:
        print(f"    - {qt.value}")


def demo_query_type_patterns() -> None:
    """Show the patterns used for query type detection."""
    print("\n" + "=" * 60)
    print("4. Query Type Patterns")
    print("=" * 60)

    for query_type, patterns in QUERY_TYPE_PATTERNS.items():
        print(f"\n  {query_type.value}:")
        for pattern in patterns:
            print(f"    - {pattern}")


def demo_entity_extraction() -> None:
    """Demonstrate entity extraction prompts."""
    print("\n" + "=" * 60)
    print("5. Entity Extraction Prompts")
    print("=" * 60)

    # Format the prompt with sample data
    sample_text = (
        "John Smith from Acme Corp announced a new AI product in San Francisco."
    )

    formatted_prompt = ENTITY_EXTRACTION_PROMPT.format(
        text=sample_text,
        entity_types="person, organization, location, product",
        language="English",
    )

    print("\n  ENTITY_EXTRACTION_PROMPT (formatted):")
    print(f'    Input text: "{sample_text}"')
    print(f"    Prompt length: {len(formatted_prompt)} characters")
    print(f"    Preview: {formatted_prompt[:300]}...")

    print("\n  QUERY_ENTITY_EXTRACTION_PROMPT (for search queries):")
    query_prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(
        query="Who is the CEO of Tesla?",
        entity_types="person, organization, role",
    )
    print(f"    Preview: {query_prompt[:300]}...")


def demo_memory_prompts() -> None:
    """Demonstrate memory/session prompts."""
    print("\n" + "=" * 60)
    print("6. Memory Prompts")
    print("=" * 60)

    print("\n  MEMORY_SUMMARIZATION_PROMPT:")
    print(f"    Length: {len(MEMORY_SUMMARIZATION_PROMPT)} characters")

    # Format with sample data
    sample_summary = "User asked about MongoDB Atlas setup."
    sample_messages = "User: How do I connect?\nAssistant: Use the connection string..."

    formatted = MEMORY_SUMMARIZATION_PROMPT.format(
        existing_summary=sample_summary,
        new_messages=sample_messages,
        max_tokens=500,
    )
    print(f"    Formatted length: {len(formatted)} characters")

    print("\n  SESSION_CONTEXT_PROMPT:")
    session_prompt = SESSION_CONTEXT_PROMPT.format(
        session_summary="Discussed RAG implementation strategies",
        time_elapsed="2 hours ago",
    )
    print(f"    Preview: {session_prompt[:200]}...")


def demo_topic_prompts() -> None:
    """Demonstrate topic extraction prompts."""
    print("\n" + "=" * 60)
    print("7. Topic Extraction Prompts")
    print("=" * 60)

    sample_text = """
    This article discusses the implementation of vector search
    using MongoDB Atlas. We cover embedding generation with
    Voyage AI and hybrid search strategies combining semantic
    and keyword-based retrieval.
    """

    formatted = TOPIC_EXTRACTION_PROMPT.format(
        text=sample_text,
        max_topics=5,
        existing_taxonomy="machine-learning, database, search, api",
    )

    print("\n  TOPIC_EXTRACTION_PROMPT:")
    print(f"    Input text length: {len(sample_text)} characters")
    print(f"    Formatted prompt length: {len(formatted)} characters")
    print(f"    Preview: {formatted[:300]}...")


def main() -> None:
    """Run all prompts demos."""
    print("\n" + "=" * 60)
    print("HybridRAG Prompts Usage Examples")
    print("=" * 60)

    demo_system_prompts()
    demo_custom_prompts()
    demo_query_type_detection()
    demo_query_type_patterns()
    demo_entity_extraction()
    demo_memory_prompts()
    demo_topic_prompts()

    print("\n" + "=" * 60)
    print("✓ All prompts demos complete!")
    print("=" * 60)
    print("\nUsage in your code:")
    print(
        """
    from hybridrag import (
        SYSTEM_PROMPT,
        create_system_prompt,
        detect_query_type,
        select_rerank_instruction,
    )

    # Use default system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Create domain-specific prompt
    medical_prompt = create_system_prompt(
        domain="Medical Research",
        persona="Medical Assistant"
    )

    # Query-aware reranking
    query_type = detect_query_type(user_query)
    rerank_instruction = select_rerank_instruction(query_type)
    """
    )


if __name__ == "__main__":
    main()
