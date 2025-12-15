#!/usr/bin/env python3
"""
Quick test to verify HybridRAG queries work with already ingested data.
Skips ingestion - assumes data was already loaded from test_alice_comprehensive.py
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hybridrag import create_hybridrag, Settings


TEST_QUESTIONS = [
    "What was written on the bottle that Alice drank from in the hall?",
    "What did the Queen of Hearts repeatedly say at the croquet ground?",
    "What animals did Alice encounter during her adventure?",
]


async def main():
    print("=" * 70)
    print("HybridRAG Query Test (No Ingestion)")
    print("=" * 70)
    print()

    # Check environment
    if not os.environ.get("MONGODB_URI"):
        print("[ERROR] MONGODB_URI not set")
        return

    if not os.environ.get("VOYAGE_API_KEY"):
        print("[ERROR] VOYAGE_API_KEY not set")
        return

    llm_var = None
    for var in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        if os.environ.get(var):
            llm_var = var
            break

    if not llm_var:
        print("[ERROR] No LLM API key found")
        return

    print(f"[INFO] Using LLM: {llm_var.replace('_API_KEY', '')}")

    # Initialize HybridRAG with the same database used during ingestion
    settings = Settings(
        mongodb_database="hybridrag_alice_test",
        llm_provider="gemini" if "GEMINI" in llm_var else "anthropic" if "ANTHROPIC" in llm_var else "openai",
    )

    print("[TEST] Initializing HybridRAG...")
    rag = await create_hybridrag(settings)
    print("[PASS] HybridRAG initialized")
    print()

    # Test queries
    print("=" * 70)
    print("TESTING QUERIES")
    print("=" * 70)
    print()

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[Q{i}] {question}")
        try:
            response = await rag.query(
                query=question,
                mode="mix",
                only_context=False
            )
            print(f"     [RESPONSE] {response[:500]}...")
            print()
        except Exception as e:
            print(f"     [ERROR] {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
