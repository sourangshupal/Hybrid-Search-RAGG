"""
MongoDB-backed Conversation Memory for HybridRAG.

Inspired by langchain-ai/langchain-mongodb MongoDBChatMessageHistory.

Key Features:
- Session-based conversation storage
- Configurable history size limit
- Async MongoDB operations
- Compatible with multi-turn conversation workflows

Schema:
    Collection: conversation_sessions
    Document:
    {
        "_id": ObjectId,
        "session_id": "unique-session-id",
        "messages": [
            {"role": "user", "content": "...", "timestamp": ISODate},
            {"role": "assistant", "content": "...", "timestamp": ISODate}
        ],
        "created_at": ISODate,
        "updated_at": ISODate,
        "metadata": {}
    }
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger("hybridrag.memory")

# Default configuration
DEFAULT_COLLECTION_NAME = "conversation_sessions"
DEFAULT_HISTORY_SIZE = 20  # Keep last N message pairs (40 messages total)
DEFAULT_SESSION_ID_KEY = "session_id"
DEFAULT_MESSAGES_KEY = "messages"


@dataclass
class ConversationSession:
    """Represents a conversation session with its messages."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_history_format(self, max_messages: int | None = None) -> list[dict[str, str]]:
        """
        Convert messages to conversation history format.

        Args:
            max_messages: Optional limit on number of messages to include

        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        messages = self.messages
        if max_messages is not None:
            messages = messages[-max_messages:]

        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    def to_context_string(self, max_messages: int | None = None) -> str:
        """
        Convert messages to a context string for retrieval augmentation.

        Args:
            max_messages: Optional limit on number of messages to include

        Returns:
            Formatted string of conversation history
        """
        messages = self.messages
        if max_messages is not None:
            messages = messages[-max_messages:]

        if not messages:
            return ""

        lines = ["Previous conversation:"]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            # Truncate very long messages for context
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)


class ConversationMemory:
    """
    MongoDB-backed conversation memory for multi-turn RAG conversations.

    Stores conversation sessions in MongoDB and provides methods to:
    - Create/retrieve sessions
    - Add messages to sessions
    - Get conversation history for RAG queries
    - Clear or delete sessions

    Example:
        ```python
        memory = ConversationMemory(
            mongodb_uri="mongodb+srv://...",
            database="hybridrag"
        )
        await memory.initialize()

        # Start conversation
        session_id = await memory.create_session()

        # Add user message
        await memory.add_message(session_id, "user", "What is this book about?")

        # Get history for RAG query
        history = await memory.get_history(session_id)

        # Add assistant response
        await memory.add_message(session_id, "assistant", "This book is about...")
        ```
    """

    def __init__(
        self,
        mongodb_uri: str | None = None,
        database: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        history_size: int | None = DEFAULT_HISTORY_SIZE,
        session_id_key: str = DEFAULT_SESSION_ID_KEY,
        messages_key: str = DEFAULT_MESSAGES_KEY,
    ):
        """
        Initialize ConversationMemory.

        Args:
            mongodb_uri: MongoDB connection string (defaults to MONGODB_URI env var)
            database: Database name (defaults to MONGODB_DATABASE env var)
            collection_name: Collection name for sessions
            history_size: Max number of message pairs to keep (None for unlimited)
            session_id_key: Field name for session ID
            messages_key: Field name for messages array
        """
        self._mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        self._database_name = database or os.environ.get("MONGODB_DATABASE", "hybridrag")
        self._collection_name = collection_name
        self._history_size = history_size
        self._session_id_key = session_id_key
        self._messages_key = messages_key

        self._client: AsyncIOMotorClient | None = None
        self._db: AsyncIOMotorDatabase | None = None
        self._initialized = False

        if not self._mongodb_uri:
            raise ValueError(
                "MongoDB URI required. Set MONGODB_URI environment variable "
                "or pass mongodb_uri parameter."
            )

    async def initialize(self) -> None:
        """Initialize MongoDB connection and create indexes."""
        if self._initialized:
            return

        logger.info(f"[MEMORY] Initializing ConversationMemory: collection={self._collection_name}")

        self._client = AsyncIOMotorClient(self._mongodb_uri)
        self._db = self._client[self._database_name]
        self._collection = self._db[self._collection_name]

        # Create index on session_id for fast lookups
        await self._collection.create_index(self._session_id_key, unique=True)
        await self._collection.create_index("created_at")
        await self._collection.create_index("updated_at")

        self._initialized = True
        logger.info("[MEMORY] ConversationMemory initialized")

    def _ensure_initialized(self) -> None:
        """Ensure memory is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "ConversationMemory not initialized. Call 'await memory.initialize()' first."
            )

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional custom session ID (generated if not provided)
            metadata: Optional metadata to store with session

        Returns:
            Session ID
        """
        self._ensure_initialized()

        session_id = session_id or str(uuid.uuid4())
        now = datetime.utcnow()

        doc = {
            self._session_id_key: session_id,
            self._messages_key: [],
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }

        try:
            await self._collection.insert_one(doc)
            logger.info(f"[MEMORY] Created session: {session_id}")
        except Exception as e:
            # Session might already exist
            if "duplicate key" in str(e).lower():
                logger.debug(f"[MEMORY] Session already exists: {session_id}")
            else:
                raise

        return session_id

    async def get_session(self, session_id: str) -> ConversationSession | None:
        """
        Get a conversation session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ConversationSession or None if not found
        """
        self._ensure_initialized()

        doc = await self._collection.find_one({self._session_id_key: session_id})

        if not doc:
            return None

        return ConversationSession(
            session_id=doc[self._session_id_key],
            messages=doc.get(self._messages_key, []),
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
            metadata=doc.get("metadata", {}),
        )

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional message metadata
        """
        self._ensure_initialized()

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
        }
        if metadata:
            message["metadata"] = metadata

        # Use upsert to create session if it doesn't exist
        result = await self._collection.update_one(
            {self._session_id_key: session_id},
            {
                "$push": {self._messages_key: message},
                "$set": {"updated_at": datetime.utcnow()},
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                    "metadata": {},
                },
            },
            upsert=True,
        )

        logger.debug(f"[MEMORY] Added {role} message to session {session_id}")

        # Trim history if needed
        if self._history_size is not None:
            await self._trim_history(session_id)

    async def _trim_history(self, session_id: str) -> None:
        """
        Trim history to keep only the most recent messages.

        Keeps the last N message pairs (2*history_size messages).
        """
        max_messages = self._history_size * 2  # Pairs of user/assistant

        doc = await self._collection.find_one(
            {self._session_id_key: session_id},
            {self._messages_key: 1}
        )

        if doc and len(doc.get(self._messages_key, [])) > max_messages:
            messages = doc[self._messages_key][-max_messages:]
            await self._collection.update_one(
                {self._session_id_key: session_id},
                {"$set": {self._messages_key: messages}}
            )
            logger.debug(f"[MEMORY] Trimmed history for session {session_id} to {max_messages} messages")

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages from a session.

        Args:
            session_id: Session ID
            limit: Optional limit on number of messages to return

        Returns:
            List of message dicts
        """
        self._ensure_initialized()

        doc = await self._collection.find_one(
            {self._session_id_key: session_id},
            {self._messages_key: 1}
        )

        if not doc:
            return []

        messages = doc.get(self._messages_key, [])

        if limit is not None:
            messages = messages[-limit:]

        return messages

    async def get_history(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Get conversation history in standard format.

        Args:
            session_id: Session ID
            max_messages: Optional limit on number of messages

        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        messages = await self.get_messages(session_id, limit=max_messages)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    # Alias for backwards compatibility
    async def get_lightrag_history(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, str]]:
        """Alias for get_history (backwards compatibility)."""
        return await self.get_history(session_id, max_messages)

    async def get_context_string(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> str:
        """
        Get conversation history as a context string for retrieval.

        This is useful for augmenting queries with conversation context
        to improve follow-up question handling.

        Args:
            session_id: Session ID
            max_messages: Max messages to include

        Returns:
            Formatted string of conversation history
        """
        session = await self.get_session(session_id)
        if not session:
            return ""
        return session.to_context_string(max_messages)

    async def clear_session(self, session_id: str) -> None:
        """
        Clear all messages from a session (keeps the session).

        Args:
            session_id: Session ID
        """
        self._ensure_initialized()

        await self._collection.update_one(
            {self._session_id_key: session_id},
            {
                "$set": {
                    self._messages_key: [],
                    "updated_at": datetime.utcnow(),
                }
            }
        )
        logger.info(f"[MEMORY] Cleared session: {session_id}")

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        self._ensure_initialized()

        result = await self._collection.delete_one({self._session_id_key: session_id})
        deleted = result.deleted_count > 0

        if deleted:
            logger.info(f"[MEMORY] Deleted session: {session_id}")
        else:
            logger.debug(f"[MEMORY] Session not found for deletion: {session_id}")

        return deleted

    async def list_sessions(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List conversation sessions.

        Args:
            limit: Max sessions to return
            skip: Number of sessions to skip

        Returns:
            List of session metadata dicts
        """
        self._ensure_initialized()

        cursor = self._collection.find(
            {},
            {
                self._session_id_key: 1,
                "created_at": 1,
                "updated_at": 1,
                "metadata": 1,
                f"{self._messages_key}": {"$slice": -1},  # Last message only
            }
        ).sort("updated_at", -1).skip(skip).limit(limit)

        sessions = []
        async for doc in cursor:
            sessions.append({
                "session_id": doc[self._session_id_key],
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "metadata": doc.get("metadata", {}),
                "message_count": len(doc.get(self._messages_key, [])),
                "last_message": doc.get(self._messages_key, [{}])[-1] if doc.get(self._messages_key) else None,
            })

        return sessions

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._initialized = False
            logger.info("[MEMORY] ConversationMemory closed")
