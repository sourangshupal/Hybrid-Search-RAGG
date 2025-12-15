#!/usr/bin/env python3
"""
Comprehensive test of HybridRAG using Alice's Adventures in Wonderland.

Tests:
1. Document ingestion
2. Manual RRF hybrid search quality
3. Entity extraction from literature
4. Multi-hop reasoning
5. Exact fact retrieval
6. Character relationship understanding
"""

import asyncio
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hybridrag import create_hybridrag, Settings


# Challenging questions that test different capabilities
CHALLENGING_QUESTIONS = [
    # 1. Exact fact retrieval - specific detail from the text
    {
        "question": "What was written on the bottle that Alice drank from in the hall?",
        "expected_keywords": ["DRINK ME", "drink me", "label"],
        "difficulty": "easy",
        "test_type": "exact_fact",
    },
    # 2. Character relationship - who interacted with whom
    {
        "question": "What did the Caterpillar ask Alice when they first met?",
        "expected_keywords": ["Who are you", "mushroom", "three inches", "size"],
        "difficulty": "medium",
        "test_type": "character_interaction",
    },
    # 3. Multi-hop reasoning - connecting multiple facts
    {
        "question": "How did Alice change size after drinking from the bottle and eating the cake?",
        "expected_keywords": ["shrink", "grow", "small", "tall", "ten inches", "telescope"],
        "difficulty": "medium",
        "test_type": "multi_hop",
    },
    # 4. Specific quote retrieval
    {
        "question": "What did the Queen of Hearts repeatedly say at the croquet ground?",
        "expected_keywords": ["Off with", "head", "Off with her head", "Off with his head"],
        "difficulty": "easy",
        "test_type": "quote",
    },
    # 5. Plot sequence understanding
    {
        "question": "What happened at the Mad Tea-Party with the March Hare, Hatter, and Dormouse?",
        "expected_keywords": ["tea", "time", "riddle", "raven", "writing desk", "sleepy", "treacle"],
        "difficulty": "medium",
        "test_type": "plot_sequence",
    },
    # 6. Character attribute - specific detail
    {
        "question": "What color were the White Rabbit's eyes?",
        "expected_keywords": ["pink"],
        "difficulty": "hard",
        "test_type": "exact_detail",
    },
    # 7. Reasoning about story logic
    {
        "question": "Why was the trial held at the end of the story?",
        "expected_keywords": ["tarts", "stolen", "Knave", "Hearts", "stole"],
        "difficulty": "medium",
        "test_type": "reasoning",
    },
    # 8. Cross-chapter understanding
    {
        "question": "What animals did Alice encounter during her adventure? Name at least five.",
        "expected_keywords": ["rabbit", "caterpillar", "cheshire", "cat", "mouse", "dormouse", "mock turtle", "griffin", "lobster", "flamingo", "hedgehog"],
        "difficulty": "medium",
        "test_type": "cross_chapter",
    },
    # 9. Specific detail from text
    {
        "question": "What was the Mock Turtle's story about his education?",
        "expected_keywords": ["school", "sea", "lessons", "reeling", "writhing", "drawing", "master", "tortoise"],
        "difficulty": "hard",
        "test_type": "specific_detail",
    },
    # 10. Challenging inference
    {
        "question": "How does Alice wake up from Wonderland?",
        "expected_keywords": ["cards", "flying", "sister", "dream", "wake", "lap", "falling"],
        "difficulty": "medium",
        "test_type": "inference",
    },
]


def evaluate_response(response: str, expected_keywords: list[str]) -> dict:
    """Evaluate if response contains expected keywords."""
    response_lower = response.lower()
    found_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    return {
        "score": score,
        "found": found_keywords,
        "missing": missing_keywords,
        "total_expected": len(expected_keywords),
    }


async def main():
    print("=" * 70)
    print("HybridRAG Comprehensive Test: Alice in Wonderland")
    print("=" * 70)
    print()

    # Check environment
    required_vars = ["MONGODB_URI", "VOYAGE_API_KEY"]
    llm_var = None
    for var in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        if os.environ.get(var):
            llm_var = var
            break

    if not llm_var:
        print("[ERROR] No LLM API key found (GEMINI, ANTHROPIC, or OPENAI)")
        return

    for var in required_vars:
        if not os.environ.get(var):
            print(f"[ERROR] Missing required environment variable: {var}")
            return

    print(f"[INFO] Using LLM: {llm_var.replace('_API_KEY', '')}")
    print()

    # Initialize HybridRAG with a fresh workspace for this test
    settings = Settings(
        mongodb_database="hybridrag_alice_test",
        llm_provider="gemini" if "GEMINI" in llm_var else "anthropic" if "ANTHROPIC" in llm_var else "openai",
    )

    print("[TEST] Initializing HybridRAG...")
    rag = await create_hybridrag(settings)
    print("[PASS] HybridRAG initialized")
    print()

    # Read the book
    book_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "alice_in_wonderland.txt")
    if not os.path.exists(book_path):
        print(f"[ERROR] Book not found at {book_path}")
        return

    with open(book_path, "r", encoding="utf-8") as f:
        book_content = f.read()

    # Remove Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = book_content.find(start_marker)
    if start_idx != -1:
        start_idx = book_content.find("\n", start_idx) + 1
    else:
        start_idx = 0

    end_idx = book_content.find(end_marker)
    if end_idx == -1:
        end_idx = len(book_content)

    book_content = book_content[start_idx:end_idx].strip()

    print(f"[INFO] Book loaded: {len(book_content)} characters")
    print()

    # Ingest the book
    print("[TEST] Ingesting Alice in Wonderland...")
    start_time = time.time()
    try:
        await rag.insert(book_content)
        ingest_time = time.time() - start_time
        print(f"[PASS] Book ingested in {ingest_time:.1f} seconds")
    except Exception as e:
        print(f"[FAIL] Ingestion failed: {e}")
        return
    print()

    # Test queries
    print("=" * 70)
    print("TESTING CHALLENGING QUESTIONS")
    print("=" * 70)
    print()

    results = []
    total_score = 0

    for i, q in enumerate(CHALLENGING_QUESTIONS, 1):
        print(f"[Q{i}] {q['question']}")
        print(f"     Difficulty: {q['difficulty']} | Type: {q['test_type']}")

        try:
            # Use mix mode for best results (Knowledge Graph + Vector Search)
            start_time = time.time()
            response = await rag.query(
                query=q["question"],
                mode="mix",
                only_context=False
            )
            query_time = time.time() - start_time

            # Evaluate response
            evaluation = evaluate_response(response, q["expected_keywords"])
            score = evaluation["score"]
            total_score += score

            # Print results
            status = "[PASS]" if score >= 0.5 else "[WEAK]" if score > 0 else "[FAIL]"
            print(f"     {status} Score: {score:.0%} ({len(evaluation['found'])}/{evaluation['total_expected']} keywords)")
            print(f"     Found: {evaluation['found']}")
            if evaluation["missing"]:
                print(f"     Missing: {evaluation['missing']}")
            print(f"     Response preview: {response[:200]}...")
            print(f"     Query time: {query_time:.1f}s")

            results.append({
                "question": q["question"],
                "score": score,
                "found": evaluation["found"],
                "missing": evaluation["missing"],
                "response_length": len(response),
                "query_time": query_time,
                "difficulty": q["difficulty"],
                "test_type": q["test_type"],
            })
        except Exception as e:
            print(f"     [ERROR] Query failed: {e}")
            results.append({
                "question": q["question"],
                "score": 0,
                "error": str(e),
            })

        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    avg_score = total_score / len(CHALLENGING_QUESTIONS)
    print(f"Overall Score: {avg_score:.0%}")
    print()

    # By difficulty
    for difficulty in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r.get("difficulty") == difficulty]
        if diff_results:
            diff_score = sum(r["score"] for r in diff_results) / len(diff_results)
            print(f"  {difficulty.capitalize()}: {diff_score:.0%} ({len(diff_results)} questions)")

    print()

    # By test type
    print("By Test Type:")
    test_types = set(r.get("test_type") for r in results if r.get("test_type"))
    for test_type in sorted(test_types):
        type_results = [r for r in results if r.get("test_type") == test_type]
        if type_results:
            type_score = sum(r["score"] for r in type_results) / len(type_results)
            print(f"  {test_type}: {type_score:.0%}")

    print()
    print("=" * 70)
    print(f"Test completed! Average score: {avg_score:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
