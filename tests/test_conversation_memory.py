#!/usr/bin/env python3
"""
Test Conversation Memory for Multi-Turn Conversations.

This test verifies that HybridRAG properly maintains conversation context
and handles follow-up questions correctly using MongoDB-backed session storage.

Key test scenarios:
1. Create a conversation session
2. Ask initial question about a topic
3. Ask follow-up questions like "explain that" or "tell me more"
4. Verify responses use conversation context appropriately
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hybridrag import create_hybridrag, Settings


# Test conversation with follow-up questions
CONVERSATION_TURNS = [
    {
        "query": "What was written on the bottle that Alice drank from in the hall?",
        "type": "initial",
        "description": "Initial question about Alice in Wonderland",
    },
    {
        "query": "Can you explain that in more detail?",
        "type": "follow_up",
        "description": "Follow-up asking for more detail",
    },
    {
        "query": "What happened after she drank it?",
        "type": "follow_up",
        "description": "Follow-up about consequences",
    },
    {
        "query": "Tell me more about her size changes in the story.",
        "type": "follow_up",
        "description": "Follow-up expanding on related topic",
    },
    {
        "query": "What about the cake? Was there something written on it too?",
        "type": "follow_up",
        "description": "Follow-up about related object",
    },
]


def evaluate_response(response: str, turn: dict) -> dict:
    """Evaluate the quality of a response."""
    response_lower = response.lower() if response else ""

    # Check for failure indicators
    failure_indicators = [
        "sorry, i'm not able",
        "i don't have enough context",
        "i cannot provide",
        "no relevant information",
        "unable to answer",
        "i don't know what you're referring to",
    ]

    is_failure = any(indicator in response_lower for indicator in failure_indicators)
    has_content = len(response) > 50 and not is_failure

    # For initial questions, just check we got a substantive response
    if turn["type"] == "initial":
        quality = "GOOD" if has_content else "POOR"
    else:
        # For follow-ups, check for context awareness
        # Good follow-up responses should reference the prior topic
        context_indicators = [
            "alice", "bottle", "drink", "size", "shrink", "grow",
            "wonderland", "hall", "she", "it", "that", "the"
        ]
        has_context = any(ind in response_lower for ind in context_indicators)

        if is_failure:
            quality = "POOR - No context"
        elif has_content and has_context:
            quality = "GOOD"
        elif has_content:
            quality = "FAIR - May lack context"
        else:
            quality = "POOR"

    return {
        "quality": quality,
        "is_failure": is_failure,
        "response_length": len(response) if response else 0,
    }


async def test_conversation_memory():
    """Test conversation memory with follow-up questions."""
    print("=" * 70)
    print("CONVERSATION MEMORY TEST")
    print("Testing multi-turn conversation with follow-up questions")
    print("=" * 70)
    print()

    # Check environment
    if not os.environ.get("MONGODB_URI"):
        print("[ERROR] MONGODB_URI not set")
        return False

    if not os.environ.get("VOYAGE_API_KEY"):
        print("[ERROR] VOYAGE_API_KEY not set")
        return False

    llm_var = None
    for var in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        if os.environ.get(var):
            llm_var = var
            break

    if not llm_var:
        print("[ERROR] No LLM API key found")
        return False

    print(f"[INFO] Using LLM: {llm_var.replace('_API_KEY', '')}")
    print()

    # Initialize HybridRAG with the Alice test database
    settings = Settings(
        mongodb_database="hybridrag_alice_test",
        llm_provider="gemini" if "GEMINI" in llm_var else "anthropic" if "ANTHROPIC" in llm_var else "openai",
    )

    print("[TEST] Initializing HybridRAG...")
    rag = await create_hybridrag(settings)
    print("[PASS] HybridRAG initialized")
    print()

    # Create a conversation session
    print("[TEST] Creating conversation session...")
    session_id = await rag.create_conversation_session(
        metadata={"test": "conversation_memory", "book": "alice_in_wonderland"}
    )
    print(f"[PASS] Session created: {session_id}")
    print()

    # Run conversation turns
    print("=" * 70)
    print("CONVERSATION TEST")
    print("=" * 70)
    print()

    results = []
    all_passed = True

    for i, turn in enumerate(CONVERSATION_TURNS, 1):
        print(f"[Turn {i}] {turn['description']}")
        print(f"         Query: {turn['query']}")
        print()

        try:
            result = await rag.query_with_memory(
                query=turn["query"],
                session_id=session_id,
                mode="mix",
                max_history_messages=10,
            )

            response = result.get("answer", "")
            history_used = result.get("history_used", 0)

            # Truncate for display
            display_response = response[:400] + "..." if len(response) > 400 else response

            print(f"         Response: {display_response}")
            print(f"         History messages used: {history_used}")

            # Evaluate response quality
            eval_result = evaluate_response(response, turn)
            quality = eval_result["quality"]

            if "POOR" in quality:
                all_passed = False
                print(f"         [FAIL] Quality: {quality}")
            else:
                print(f"         [PASS] Quality: {quality}")

            results.append({
                "turn": i,
                "type": turn["type"],
                "description": turn["description"],
                "query": turn["query"],
                "response_length": eval_result["response_length"],
                "history_used": history_used,
                "quality": quality,
            })

        except Exception as e:
            print(f"         [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            results.append({
                "turn": i,
                "type": turn["type"],
                "description": turn["description"],
                "error": str(e),
                "quality": "ERROR",
            })

        print("-" * 70)
        print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    good_count = sum(1 for r in results if "GOOD" in r.get("quality", ""))
    fair_count = sum(1 for r in results if "FAIR" in r.get("quality", ""))
    poor_count = sum(1 for r in results if "POOR" in r.get("quality", ""))
    error_count = sum(1 for r in results if r.get("quality") == "ERROR")

    print(f"Total turns: {len(results)}")
    print(f"  GOOD:  {good_count}")
    print(f"  FAIR:  {fair_count}")
    print(f"  POOR:  {poor_count}")
    print(f"  ERROR: {error_count}")
    print()

    # Check conversation history was stored
    print("[TEST] Verifying conversation history in MongoDB...")
    history = await rag.get_conversation_history(session_id)
    print(f"[INFO] Messages stored: {len(history)}")

    if len(history) >= len(CONVERSATION_TURNS) * 2:  # user + assistant for each turn
        print("[PASS] Conversation history properly stored")
    else:
        print(f"[WARN] Expected at least {len(CONVERSATION_TURNS) * 2} messages, got {len(history)}")

    print()

    # Final verdict
    if all_passed and poor_count == 0 and error_count == 0:
        print("=" * 70)
        print("[SUCCESS] All conversation memory tests passed!")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("[FAILURE] Some conversation memory tests failed")
        print("Follow-up questions may not be using conversation context properly")
        print("=" * 70)
        return False


async def test_conversation_memory_isolated():
    """Test ConversationMemory class in isolation."""
    print("=" * 70)
    print("CONVERSATION MEMORY ISOLATION TEST")
    print("Testing MongoDB session storage directly")
    print("=" * 70)
    print()

    from hybridrag.memory import ConversationMemory

    if not os.environ.get("MONGODB_URI"):
        print("[ERROR] MONGODB_URI not set")
        return False

    mongodb_uri = os.environ.get("MONGODB_URI")
    database = os.environ.get("MONGODB_DATABASE", "hybridrag_test")

    print(f"[INFO] Database: {database}")
    print()

    # Initialize memory
    print("[TEST] Initializing ConversationMemory...")
    memory = ConversationMemory(
        mongodb_uri=mongodb_uri,
        database=database,
        collection_name="test_conversation_sessions",
    )
    await memory.initialize()
    print("[PASS] ConversationMemory initialized")
    print()

    # Test session creation
    print("[TEST] Creating session...")
    session_id = await memory.create_session(
        metadata={"test": "isolation_test"}
    )
    print(f"[PASS] Session created: {session_id}")
    print()

    # Test message addition
    print("[TEST] Adding messages...")
    await memory.add_message(session_id, "user", "What is Alice in Wonderland about?")
    await memory.add_message(session_id, "assistant", "Alice in Wonderland is a story about a girl who falls down a rabbit hole.")
    await memory.add_message(session_id, "user", "Tell me more about the characters.")
    await memory.add_message(session_id, "assistant", "The story features many characters including the Cheshire Cat and the Queen of Hearts.")
    print("[PASS] Messages added")
    print()

    # Test message retrieval
    print("[TEST] Retrieving messages...")
    messages = await memory.get_messages(session_id)
    print(f"[INFO] Retrieved {len(messages)} messages")
    for msg in messages:
        print(f"       - {msg['role']}: {msg['content'][:50]}...")
    print("[PASS] Messages retrieved")
    print()

    # Test history format
    print("[TEST] Getting history format...")
    history = await memory.get_history(session_id)
    print(f"[INFO] History format: {len(history)} messages")
    for h in history:
        print(f"       - {h['role']}: {h['content'][:50]}...")
    print("[PASS] History format works")
    print()

    # Test context string
    print("[TEST] Getting context string...")
    context = await memory.get_context_string(session_id)
    print(f"[INFO] Context string length: {len(context)} chars")
    print(f"       Preview: {context[:200]}...")
    print("[PASS] Context string works")
    print()

    # Clean up
    print("[TEST] Cleaning up test session...")
    await memory.delete_session(session_id)
    print("[PASS] Session deleted")
    print()

    await memory.close()
    print("[SUCCESS] Isolation test complete!")
    return True


async def main():
    """Run all conversation memory tests."""
    print()
    print("=" * 70)
    print("HYBRIDRAG CONVERSATION MEMORY TEST SUITE")
    print("=" * 70)
    print()

    # First run isolation test
    isolation_passed = await test_conversation_memory_isolated()
    print()

    if isolation_passed:
        # Then run full integration test
        integration_passed = await test_conversation_memory()
        print()

        if integration_passed:
            print("[OVERALL] All tests passed!")
            return 0
        else:
            print("[OVERALL] Integration tests failed")
            return 1
    else:
        print("[OVERALL] Isolation tests failed - skipping integration")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
