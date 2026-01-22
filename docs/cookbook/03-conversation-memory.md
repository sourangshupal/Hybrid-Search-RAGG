# Recipe 03: Multi-Turn Conversation Memory with MongoDB

Build AI agents with persistent, context-aware memory using MongoDB.

## Overview

Modern AI applications need to maintain conversation context across multiple turns. MongoDB provides the ideal foundation for:

- Session-based conversation storage
- Efficient history retrieval
- Automatic context window management
- Long-term memory with summarization

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Query                        │
│           "What did we discuss earlier?"            │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              ConversationMemory                      │
│  ├── Retrieve session by session_id                 │
│  ├── Get conversation history                        │
│  ├── Include running summary (if compacted)         │
│  └── Format for LLM context                         │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                 MongoDB Atlas                        │
│  Collection: conversation_sessions                   │
│  {                                                   │
│    "_id": ObjectId,                                  │
│    "session_id": "abc-123",                         │
│    "messages": [                                     │
│      {"role": "user", "content": "...", ...},       │
│      {"role": "assistant", "content": "...", ...}   │
│    ],                                                │
│    "summary": "Previous context...",                │
│    "created_at": ISODate,                           │
│    "updated_at": ISODate,                           │
│    "metadata": {}                                    │
│  }                                                   │
└─────────────────────────────────────────────────────┘
```

## Schema Design

### Conversation Session Document

```javascript
{
  "_id": ObjectId("..."),
  "session_id": "user-123-chat-456",
  "messages": [
    {
      "role": "user",
      "content": "What is MongoDB Atlas?",
      "timestamp": ISODate("2024-01-15T10:30:00Z")
    },
    {
      "role": "assistant",
      "content": "MongoDB Atlas is a fully managed cloud database...",
      "timestamp": ISODate("2024-01-15T10:30:02Z")
    }
  ],
  "summary": "",  // Populated when messages are compacted
  "summary_token_count": 0,
  "created_at": ISODate("2024-01-15T10:30:00Z"),
  "updated_at": ISODate("2024-01-15T10:35:00Z"),
  "metadata": {
    "user_id": "user-123",
    "agent_type": "support",
    "language": "en"
  }
}
```

## Implementation

### Basic Usage

```python
from hybridrag.memory import ConversationMemory

# Initialize memory
memory = ConversationMemory(
    mongodb_uri="mongodb+srv://...",
    database="hybridrag",
    collection_name="conversation_sessions",
    history_size=50,        # Keep last 50 message pairs
    max_token_limit=32000,  # Trigger compaction at this limit
)
await memory.initialize()

# Create a new session
session_id = await memory.create_session(
    metadata={"user_id": "user-123", "agent_type": "support"}
)

# Add user message
await memory.add_message(session_id, "user", "What is vector search?")

# Get history for RAG query
session = await memory.get_session(session_id)
history = session.to_history_format()
# Returns: [{"role": "user", "content": "What is vector search?"}]

# Add assistant response
await memory.add_message(
    session_id,
    "assistant",
    "Vector search allows you to find similar items..."
)
```

### Query with Memory (HybridRAG)

```python
from hybridrag import HybridRAG

rag = HybridRAG()
await rag.initialize()

# Create conversation session
session_id = await rag.create_conversation_session()

# First query
response1 = await rag.query_with_memory(
    "What is MongoDB Atlas?",
    session_id=session_id
)

# Follow-up query (context-aware)
response2 = await rag.query_with_memory(
    "How does it compare to self-hosted?",  # "it" refers to MongoDB Atlas
    session_id=session_id
)
```

### Context String for RAG

```python
# Get formatted context including summary and recent messages
session = await memory.get_session(session_id)
context = session.to_context_string(max_messages=10)

# Result:
# Summary of earlier conversation:
# User asked about MongoDB Atlas and its features...
#
# Recent conversation:
# User: How does it compare to self-hosted?
# Assistant: Compared to self-hosted MongoDB...
```

## Self-Compaction (Summarization)

When conversations exceed the token limit, older messages are automatically summarized:

```python
from hybridrag.memory import ConversationMemory

memory = ConversationMemory(
    max_token_limit=32000,
    llm_func=llm_generate,  # LLM function for summarization
)

# After many messages, when token count exceeds limit:
# 1. Older messages are summarized by LLM
# 2. Summary is stored in session.summary
# 3. Original messages are removed from session.messages
# 4. Token count is recalculated
```

### Summarization Prompt

```python
SUMMARY_PROMPT = """Progressively summarize the conversation below, adding onto the previous summary.
Return a new summary that incorporates the key points from both the previous summary and new messages.
Focus on important facts, decisions, and context that would be useful for future questions.

CURRENT SUMMARY:
{summary}

NEW MESSAGES:
{new_lines}

NEW SUMMARY:"""
```

## Advanced Patterns

### 1. Session Metadata for Filtering

```python
# Create session with rich metadata
session_id = await memory.create_session(
    metadata={
        "user_id": "user-123",
        "agent_type": "sales",
        "product_interest": ["atlas", "vector-search"],
        "language": "en",
        "started_at": datetime.now().isoformat()
    }
)

# Query sessions by metadata
sessions = await db.conversation_sessions.find({
    "metadata.user_id": "user-123",
    "metadata.agent_type": "sales"
}).to_list(length=None)
```

### 2. Multi-Agent Memory Sharing

```python
# Agent 1: Sales agent creates session
session_id = await memory.create_session(
    metadata={"agent_type": "sales", "user_id": "user-123"}
)
await memory.add_message(session_id, "user", "I need help choosing a plan")
await memory.add_message(session_id, "assistant", "Here are our pricing options...")

# Agent 2: Support agent continues same session
session = await memory.get_session(session_id)
context = session.to_context_string()
# Support agent has full context from sales conversation
```

### 3. Conversation Branching

```python
# Fork a conversation for A/B testing or exploration
original_session = await memory.get_session(original_session_id)

# Create new session with same history
branch_session_id = await memory.create_session(
    metadata={"branched_from": original_session_id}
)

# Copy messages to branch
for msg in original_session.messages:
    await memory.add_message(
        branch_session_id,
        msg["role"],
        msg["content"]
    )
```

### 4. History-Augmented Retrieval

```python
async def query_with_history_augmentation(
    query: str,
    session_id: str,
    rag: HybridRAG
) -> str:
    """Augment query with conversation context for better retrieval."""

    # Get conversation context
    session = await rag.memory.get_session(session_id)
    history_context = session.to_context_string(max_messages=5)

    # Augment query with context
    augmented_query = f"""Given this conversation context:
{history_context}

Current question: {query}"""

    # Use augmented query for retrieval
    results = await rag.retrieve(augmented_query)

    # Generate response with full context
    response = await rag.generate(
        query=query,
        context=results,
        conversation_history=session.to_history_format()
    )

    # Store in memory
    await rag.memory.add_message(session_id, "user", query)
    await rag.memory.add_message(session_id, "assistant", response)

    return response
```

## MongoDB Operations

### Create Indexes

```python
await memory.ensure_indexes()

# Creates:
# - Unique index on session_id
# - Index on created_at for time-based queries
# - Index on metadata fields for filtering
```

### Clear Session

```python
# Clear messages but keep session
await memory.clear_session(session_id)

# Delete entire session
await memory.delete_session(session_id)
```

### List Sessions

```python
# Get all sessions for a user
sessions = await memory.list_sessions(
    filter={"metadata.user_id": "user-123"},
    limit=10,
    sort_by="updated_at",
    sort_order=-1  # Newest first
)
```

## Token Counting

Efficient token estimation for context window management:

```python
def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token for English)."""
    return len(text) // 4

async def check_context_limit(session: ConversationSession, limit: int = 32000) -> bool:
    """Check if session is approaching context limit."""
    total = session.summary_token_count
    for msg in session.messages:
        total += estimate_tokens(msg["content"])
    return total > limit * 0.8  # 80% threshold
```

## Best Practices

### 1. Session ID Strategy

```python
import uuid

def generate_session_id(user_id: str, context: str = "chat") -> str:
    """Generate predictable, collision-free session IDs."""
    return f"{user_id}-{context}-{uuid.uuid4().hex[:8]}"

# Example: "user-123-support-a1b2c3d4"
```

### 2. Graceful Degradation

```python
async def get_history_safely(session_id: str) -> list:
    """Get history with fallback for missing sessions."""
    try:
        session = await memory.get_session(session_id)
        if session:
            return session.to_history_format()
    except Exception as e:
        logger.warning(f"Failed to get session: {e}")

    return []  # Return empty history on failure
```

### 3. Context Window Management

```python
def prepare_context_for_llm(
    session: ConversationSession,
    max_tokens: int = 8000
) -> list[dict]:
    """Prepare history that fits within token limit."""
    history = session.to_history_format()
    total_tokens = 0
    result = []

    # Start from most recent
    for msg in reversed(history):
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        result.insert(0, msg)
        total_tokens += msg_tokens

    # Always include summary if available
    if session.summary:
        result.insert(0, {
            "role": "system",
            "content": f"Summary of earlier conversation: {session.summary}"
        })

    return result
```

### 4. TTL for Session Cleanup

```javascript
// MongoDB TTL index for automatic session expiration
db.conversation_sessions.createIndex(
  { "updated_at": 1 },
  { expireAfterSeconds: 86400 * 30 }  // 30 days
)
```

## Integration with AI Agents

```python
class MemoryEnabledAgent:
    """AI agent with persistent MongoDB memory."""

    def __init__(self, rag: HybridRAG, session_id: str):
        self.rag = rag
        self.session_id = session_id

    async def process(self, user_input: str) -> str:
        # Store user message
        await self.rag.memory.add_message(
            self.session_id, "user", user_input
        )

        # Get context-aware response
        response = await self.rag.query_with_memory(
            user_input,
            session_id=self.session_id
        )

        # Store assistant response
        await self.rag.memory.add_message(
            self.session_id, "assistant", response
        )

        return response

    async def get_context(self) -> str:
        session = await self.rag.memory.get_session(self.session_id)
        return session.to_context_string()
```

## References

- [LangChain MongoDB Chat History](https://python.langchain.com/docs/integrations/memory/mongodb_chat_message_history)
- [MongoDB Schema Design](https://www.mongodb.com/docs/manual/core/data-modeling-introduction/)
- [TTL Indexes](https://www.mongodb.com/docs/manual/core/index-ttl/)

---

**Previous**: [Recipe 02: Lexical Prefilters](./02-lexical-prefilters.md)
**Next**: [Recipe 04: Vector Search Optimization](./04-vector-search-optimization.md)
