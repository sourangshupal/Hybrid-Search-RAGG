"""
Conversation Memory Module for HybridRAG.

Provides MongoDB-backed conversation history management
to enable multi-turn conversations with context retention.

Inspired by langchain-ai/langchain-mongodb MongoDBChatMessageHistory.

Usage:
    ```python
    from hybridrag.memory import ConversationMemory

    memory = ConversationMemory(mongodb_uri="...", database="...")
    await memory.initialize()

    # Create or get session
    session_id = await memory.create_session()

    # Add messages
    await memory.add_message(session_id, "user", "What is this book about?")
    await memory.add_message(session_id, "assistant", "This book is about...")

    # Get history for context
    history = await memory.get_messages(session_id)
    ```
"""

from .conversation import ConversationMemory, ConversationSession

__all__ = ["ConversationMemory", "ConversationSession"]
