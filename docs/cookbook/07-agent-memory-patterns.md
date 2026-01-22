# Recipe 07: AI Agent Memory Patterns with MongoDB

Build intelligent AI agents with persistent, structured memory using MongoDB.

## Overview

Modern AI agents need more than conversation history. They need:

- **Short-term memory**: Current conversation context
- **Long-term memory**: Persistent facts and preferences
- **Working memory**: Active task state
- **Episodic memory**: Past interactions and outcomes

MongoDB provides the ideal foundation for all these memory types.

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Agent                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Short-term │  │  Working    │  │  Tool/Action            │  │
│  │  Memory     │  │  Memory     │  │  Results                │  │
│  │  (Context)  │  │  (Tasks)    │  │  (Observations)         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                │
│         └────────────────┼─────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MongoDB Atlas                         │   │
│  │  ├── conversation_sessions (short-term)                 │   │
│  │  ├── agent_memories (long-term facts)                   │   │
│  │  ├── task_states (working memory)                       │   │
│  │  └── action_logs (episodic memory)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Schema Designs

### 1. Short-Term Memory (Conversations)

```javascript
// Collection: conversation_sessions
{
  "_id": ObjectId("..."),
  "session_id": "agent-task-123",
  "agent_id": "support-agent-v1",
  "messages": [
    {"role": "user", "content": "...", "timestamp": ISODate("...")},
    {"role": "assistant", "content": "...", "timestamp": ISODate("...")},
    {"role": "tool", "name": "search", "result": {...}, "timestamp": ISODate("...")}
  ],
  "summary": "User asked about...",
  "created_at": ISODate("..."),
  "updated_at": ISODate("..."),
  "metadata": {
    "user_id": "user-456",
    "context": "support",
    "priority": "high"
  }
}
```

### 2. Long-Term Memory (Facts & Preferences)

```javascript
// Collection: agent_memories
{
  "_id": ObjectId("..."),
  "memory_id": "mem-789",
  "agent_id": "support-agent-v1",
  "user_id": "user-456",
  "memory_type": "fact",  // "fact", "preference", "instruction", "learned"
  "content": "User prefers detailed technical explanations",
  "importance": 0.9,  // 0-1 scale
  "vector": [...],  // Embedding for semantic retrieval
  "tags": ["user_preference", "communication_style"],
  "created_at": ISODate("..."),
  "last_accessed": ISODate("..."),
  "access_count": 15,
  "source": {
    "session_id": "agent-task-100",
    "message_index": 5
  },
  "expires_at": null  // Or ISODate for temporary memories
}
```

### 3. Working Memory (Task State)

```javascript
// Collection: task_states
{
  "_id": ObjectId("..."),
  "task_id": "task-abc",
  "agent_id": "research-agent-v1",
  "session_id": "agent-task-123",
  "status": "in_progress",  // "pending", "in_progress", "completed", "failed"
  "goal": "Research MongoDB vector search best practices",
  "plan": [
    {"step": 1, "action": "search", "params": {...}, "status": "completed"},
    {"step": 2, "action": "read", "params": {...}, "status": "in_progress"},
    {"step": 3, "action": "summarize", "params": {...}, "status": "pending"}
  ],
  "context": {
    "search_results": [...],
    "current_focus": "performance optimization",
    "discovered_entities": ["$rankFusion", "numCandidates"]
  },
  "created_at": ISODate("..."),
  "updated_at": ISODate("..."),
  "checkpoints": [
    {"timestamp": ISODate("..."), "state_snapshot": {...}}
  ]
}
```

### 4. Episodic Memory (Action Logs)

```javascript
// Collection: action_logs
{
  "_id": ObjectId("..."),
  "action_id": "act-xyz",
  "agent_id": "support-agent-v1",
  "session_id": "agent-task-123",
  "task_id": "task-abc",
  "action_type": "tool_call",
  "action": {
    "tool": "hybrid_search",
    "input": {"query": "vector search optimization"},
    "output": {...},
    "duration_ms": 150
  },
  "outcome": "success",  // "success", "failure", "partial"
  "feedback": {
    "user_rating": 5,
    "was_helpful": true
  },
  "timestamp": ISODate("..."),
  "metadata": {
    "model_version": "claude-3-opus",
    "tokens_used": 1500
  }
}
```

## Implementation Patterns

### 1. Memory-Enabled Agent Base Class

```python
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorDatabase
from hybridrag.memory import ConversationMemory

@dataclass
class AgentConfig:
    agent_id: str
    model: str = "claude-3-opus"
    max_context_tokens: int = 8000
    memory_importance_threshold: float = 0.7


class MemoryEnabledAgent:
    """Base class for AI agents with MongoDB-backed memory."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        config: AgentConfig,
        rag: "HybridRAG"
    ):
        self.db = db
        self.config = config
        self.rag = rag
        self.conversation_memory = ConversationMemory(
            mongodb_uri=None,  # Uses existing connection
            database=db.name
        )

    async def initialize(self):
        """Initialize memory systems."""
        await self.conversation_memory.initialize()
        await self._ensure_indexes()

    async def _ensure_indexes(self):
        """Create necessary indexes for memory collections."""
        # Long-term memory
        await self.db.agent_memories.create_index([
            ("agent_id", 1),
            ("user_id", 1),
            ("memory_type", 1)
        ])
        await self.db.agent_memories.create_index([
            ("importance", -1)
        ])

        # Task states
        await self.db.task_states.create_index([
            ("task_id", 1)
        ], unique=True)

        # Action logs
        await self.db.action_logs.create_index([
            ("session_id", 1),
            ("timestamp", -1)
        ])
```

### 2. Long-Term Memory Operations

```python
class LongTermMemory:
    """Persistent fact and preference storage."""

    def __init__(self, db: AsyncIOMotorDatabase, embed_func):
        self.db = db
        self.collection = db.agent_memories
        self.embed_func = embed_func

    async def store_memory(
        self,
        agent_id: str,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: list[str] = None
    ) -> str:
        """Store a new long-term memory."""
        # Generate embedding for semantic retrieval
        embedding = await self.embed_func([content])

        memory = {
            "memory_id": f"mem-{uuid.uuid4().hex[:8]}",
            "agent_id": agent_id,
            "user_id": user_id,
            "memory_type": memory_type,
            "content": content,
            "importance": importance,
            "vector": embedding[0],
            "tags": tags or [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0
        }

        await self.collection.insert_one(memory)
        return memory["memory_id"]

    async def retrieve_relevant_memories(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.5
    ) -> list[dict]:
        """Retrieve semantically relevant memories."""
        query_embedding = await self.embed_func([query])

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memory_vector_index",
                    "path": "vector",
                    "queryVector": query_embedding[0],
                    "numCandidates": top_k * 10,
                    "limit": top_k * 2,
                    "filter": {
                        "$and": [
                            {"agent_id": {"$eq": agent_id}},
                            {"user_id": {"$eq": user_id}},
                            {"importance": {"$gte": min_importance}}
                        ]
                    }
                }
            },
            {
                "$project": {
                    "memory_id": 1,
                    "content": 1,
                    "memory_type": 1,
                    "importance": 1,
                    "tags": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {"$limit": top_k}
        ]

        results = await self.collection.aggregate(pipeline).to_list(length=None)

        # Update access counts
        memory_ids = [r["memory_id"] for r in results]
        await self.collection.update_many(
            {"memory_id": {"$in": memory_ids}},
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed": datetime.utcnow()}
            }
        )

        return results

    async def decay_memories(
        self,
        agent_id: str,
        decay_factor: float = 0.95,
        min_importance: float = 0.1
    ):
        """Apply importance decay to less-accessed memories."""
        await self.collection.update_many(
            {
                "agent_id": agent_id,
                "importance": {"$gt": min_importance},
                "last_accessed": {
                    "$lt": datetime.utcnow() - timedelta(days=30)
                }
            },
            [
                {
                    "$set": {
                        "importance": {
                            "$max": [
                                min_importance,
                                {"$multiply": ["$importance", decay_factor]}
                            ]
                        }
                    }
                }
            ]
        )
```

### 3. Working Memory (Task State)

```python
class WorkingMemory:
    """Active task state management."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.task_states

    async def create_task(
        self,
        agent_id: str,
        session_id: str,
        goal: str,
        plan: list[dict] = None
    ) -> str:
        """Create a new task with initial plan."""
        task = {
            "task_id": f"task-{uuid.uuid4().hex[:8]}",
            "agent_id": agent_id,
            "session_id": session_id,
            "status": "pending",
            "goal": goal,
            "plan": plan or [],
            "context": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "checkpoints": []
        }

        await self.collection.insert_one(task)
        return task["task_id"]

    async def update_task_context(
        self,
        task_id: str,
        context_updates: dict
    ):
        """Update task context with new information."""
        await self.collection.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    **{f"context.{k}": v for k, v in context_updates.items()},
                    "updated_at": datetime.utcnow()
                }
            }
        )

    async def advance_plan_step(
        self,
        task_id: str,
        step_index: int,
        status: str = "completed",
        result: dict = None
    ):
        """Mark a plan step as completed and store result."""
        await self.collection.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    f"plan.{step_index}.status": status,
                    f"plan.{step_index}.result": result,
                    "updated_at": datetime.utcnow()
                }
            }
        )

    async def create_checkpoint(self, task_id: str):
        """Save current state as checkpoint for recovery."""
        task = await self.collection.find_one({"task_id": task_id})
        if task:
            checkpoint = {
                "timestamp": datetime.utcnow(),
                "state_snapshot": {
                    "plan": task["plan"],
                    "context": task["context"],
                    "status": task["status"]
                }
            }
            await self.collection.update_one(
                {"task_id": task_id},
                {"$push": {"checkpoints": checkpoint}}
            )

    async def restore_from_checkpoint(
        self,
        task_id: str,
        checkpoint_index: int = -1  # -1 for latest
    ):
        """Restore task state from checkpoint."""
        task = await self.collection.find_one({"task_id": task_id})
        if task and task.get("checkpoints"):
            checkpoint = task["checkpoints"][checkpoint_index]
            await self.collection.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "plan": checkpoint["state_snapshot"]["plan"],
                        "context": checkpoint["state_snapshot"]["context"],
                        "status": checkpoint["state_snapshot"]["status"],
                        "updated_at": datetime.utcnow()
                    }
                }
            )
```

### 4. Episodic Memory (Action Logging)

```python
class EpisodicMemory:
    """Action and outcome logging for learning."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.action_logs

    async def log_action(
        self,
        agent_id: str,
        session_id: str,
        task_id: str,
        action_type: str,
        action: dict,
        outcome: str,
        metadata: dict = None
    ) -> str:
        """Log an agent action with outcome."""
        log = {
            "action_id": f"act-{uuid.uuid4().hex[:8]}",
            "agent_id": agent_id,
            "session_id": session_id,
            "task_id": task_id,
            "action_type": action_type,
            "action": action,
            "outcome": outcome,
            "feedback": {},
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }

        await self.collection.insert_one(log)
        return log["action_id"]

    async def add_feedback(
        self,
        action_id: str,
        user_rating: int = None,
        was_helpful: bool = None,
        correction: str = None
    ):
        """Add user feedback to an action."""
        feedback = {}
        if user_rating is not None:
            feedback["user_rating"] = user_rating
        if was_helpful is not None:
            feedback["was_helpful"] = was_helpful
        if correction is not None:
            feedback["correction"] = correction

        await self.collection.update_one(
            {"action_id": action_id},
            {"$set": {f"feedback.{k}": v for k, v in feedback.items()}}
        )

    async def get_similar_past_actions(
        self,
        agent_id: str,
        action_type: str,
        context_embedding: list[float],
        top_k: int = 5,
        only_successful: bool = True
    ) -> list[dict]:
        """Find similar past actions for learning."""
        match_filter = {"agent_id": agent_id, "action_type": action_type}
        if only_successful:
            match_filter["outcome"] = "success"

        pipeline = [
            {"$match": match_filter},
            {"$sort": {"timestamp": -1}},
            {"$limit": top_k * 3},
            # Add more sophisticated matching here
            {"$limit": top_k}
        ]

        return await self.collection.aggregate(pipeline).to_list(length=None)

    async def get_success_rate(
        self,
        agent_id: str,
        action_type: str,
        time_window_days: int = 30
    ) -> dict:
        """Calculate action success rate."""
        cutoff = datetime.utcnow() - timedelta(days=time_window_days)

        pipeline = [
            {
                "$match": {
                    "agent_id": agent_id,
                    "action_type": action_type,
                    "timestamp": {"$gte": cutoff}
                }
            },
            {
                "$group": {
                    "_id": "$outcome",
                    "count": {"$sum": 1}
                }
            }
        ]

        results = await self.collection.aggregate(pipeline).to_list(length=None)

        counts = {r["_id"]: r["count"] for r in results}
        total = sum(counts.values())

        return {
            "total_actions": total,
            "success_rate": counts.get("success", 0) / total if total > 0 else 0,
            "failure_rate": counts.get("failure", 0) / total if total > 0 else 0,
            "counts": counts
        }
```

### 5. Complete Agent with All Memory Types

```python
class RAGAgent(MemoryEnabledAgent):
    """Full-featured RAG agent with all memory systems."""

    def __init__(self, db, config, rag):
        super().__init__(db, config, rag)
        self.long_term = LongTermMemory(db, rag.embed)
        self.working = WorkingMemory(db)
        self.episodic = EpisodicMemory(db)

    async def process_query(
        self,
        user_id: str,
        session_id: str,
        query: str
    ) -> str:
        """Process user query with full memory context."""

        # 1. Get conversation history (short-term)
        session = await self.conversation_memory.get_session(session_id)
        short_term_context = session.to_context_string(max_messages=10)

        # 2. Retrieve relevant long-term memories
        memories = await self.long_term.retrieve_relevant_memories(
            self.config.agent_id,
            user_id,
            query,
            top_k=5,
            min_importance=self.config.memory_importance_threshold
        )
        long_term_context = "\n".join([m["content"] for m in memories])

        # 3. Get active task context (working memory)
        task = await self.working.collection.find_one({
            "session_id": session_id,
            "status": "in_progress"
        })
        working_context = task["context"] if task else {}

        # 4. Build augmented context
        augmented_context = f"""
Long-term memories about this user:
{long_term_context}

Current conversation:
{short_term_context}

Active task context:
{json.dumps(working_context, indent=2)}
"""

        # 5. RAG retrieval with augmented query
        rag_results = await self.rag.retrieve(
            query,
            context_hint=augmented_context
        )

        # 6. Generate response
        response = await self.rag.generate(
            query=query,
            context=rag_results,
            system_context=augmented_context
        )

        # 7. Log action (episodic memory)
        await self.episodic.log_action(
            self.config.agent_id,
            session_id,
            task["task_id"] if task else None,
            "query_response",
            {"query": query, "rag_results_count": len(rag_results)},
            "success"
        )

        # 8. Store messages (short-term)
        await self.conversation_memory.add_message(session_id, "user", query)
        await self.conversation_memory.add_message(session_id, "assistant", response)

        # 9. Extract and store new facts (long-term)
        new_facts = await self._extract_facts(query, response)
        for fact in new_facts:
            await self.long_term.store_memory(
                self.config.agent_id,
                user_id,
                fact["content"],
                memory_type="learned",
                importance=fact.get("importance", 0.5)
            )

        return response

    async def _extract_facts(
        self,
        query: str,
        response: str
    ) -> list[dict]:
        """Extract storable facts from conversation."""
        # Use LLM to extract facts worth remembering
        extraction_prompt = f"""
Extract any facts about the user that should be remembered for future conversations.
Return as JSON array: [{{"content": "fact", "importance": 0.0-1.0}}]

User query: {query}
Assistant response: {response}

Facts (or empty array if none):"""

        result = await self.rag.llm(extraction_prompt)
        try:
            return json.loads(result)
        except:
            return []
```

## Index Configurations

```javascript
// Long-term memory vector index
{
  "name": "memory_vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {"type": "vector", "path": "vector", "numDimensions": 1024, "similarity": "cosine"},
      {"type": "filter", "path": "agent_id"},
      {"type": "filter", "path": "user_id"},
      {"type": "filter", "path": "importance"}
    ]
  }
}

// Text search on memory content
{
  "name": "memory_text_index",
  "type": "search",
  "definition": {
    "mappings": {
      "fields": {
        "content": {"type": "string", "analyzer": "lucene.standard"},
        "tags": {"type": "string"}
      }
    }
  }
}
```

## Best Practices

| Aspect | Recommendation |
|--------|---------------|
| Memory decay | Apply 0.95 decay to unused memories monthly |
| Importance threshold | Start at 0.5, tune based on usage |
| Checkpoint frequency | Every 5-10 agent steps |
| Log retention | Keep 30 days, summarize older |
| Context window | Reserve 50% for RAG, 25% for memory |

---

**Previous**: [Recipe 06: Filtering Strategies](./06-filtering-strategies.md)
**Next**: [Recipe 08: Production Deployment](./08-production-deployment.md)
