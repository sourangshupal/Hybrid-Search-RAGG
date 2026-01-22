# Building AI Agents with HybridRAG

This guide shows how to use HybridRAG as a foundation for building AI agents.

## Overview

HybridRAG provides a complete RAG pipeline:
1. Document ingestion with chunking
2. Embedding generation (Voyage AI)
3. Hybrid search (vector + keyword)
4. Reranking (Voyage rerank-2.5)
5. Response generation (Claude/GPT/Gemini)

When building agents, you wrap this entire pipeline as a **tool** that the agent
can invoke when it needs information from your knowledge base.

## Prerequisites

```bash
pip install mongodb-hybridrag[agent]
```

This installs:
- `langchain-core>=0.3.0`
- `langchain-anthropic>=0.3.0`
- `langgraph>=0.2.0`

## Basic Pattern: HybridRAG as a Tool

```python
from langchain_core.tools import tool
from hybridrag import create_hybridrag

# Initialize once at startup
rag = await create_hybridrag()

@tool
async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information.

    Use this tool when you need to find information from documents.
    """
    result = await rag.query_with_sources(
        query=query,
        mode="mix",  # Use mix mode for best quality
    )
    return result.get("answer", "No relevant information found.")
```

### Why `query_with_sources` Instead of Just `query`?

- `query()` returns just the generated response string
- `query_with_sources()` returns a dict with both `answer` and `context`

For tools, `query_with_sources` is better because:
1. You can format the response with or without context
2. You can handle empty results gracefully
3. The context can be shown to users for transparency

## LangGraph Agent Pattern

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 1. Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Create agent node
async def agent_node(state: AgentState):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    llm_with_tools = llm.bind_tools([search_knowledge_base])

    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

# 3. Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([search_knowledge_base]))
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda s: "tools" if s["messages"][-1].tool_calls else "end",
)
graph.add_edge("tools", "agent")

agent = graph.compile()
```

## When Should the Agent Use the Tool?

The agent should use `search_knowledge_base` when:
- User asks about specific documents or content
- Question requires domain-specific information
- User explicitly asks to search

The agent should answer directly when:
- General conversation (greetings, etc.)
- Questions answerable from general knowledge
- Follow-up questions about its previous responses
- Clarification requests

## Adding More Tools

HybridRAG supports multiple operations you can expose as tools:

```python
@tool
async def get_kb_stats() -> str:
    """Get knowledge base statistics."""
    stats = await rag.get_knowledge_base_stats()
    return f"Documents: {stats['documents']['total']}, Entities: {stats['entities']}"

@tool
async def ingest_url(url: str) -> str:
    """Ingest content from a URL into the knowledge base."""
    result = await rag.ingest_url(url)
    if result.success:
        return f"Ingested {result.chunks_created} chunks from {url}"
    return f"Failed: {result.errors}"
```

## Multi-Turn Conversations

For conversations that span multiple turns, use HybridRAG's memory:

```python
@tool
async def search_with_context(query: str, session_id: str = "default") -> str:
    """Search with conversation context."""
    result = await rag.query_with_memory(
        query=query,
        session_id=session_id,
        max_history_messages=10,
    )
    return result.get("answer", "No information found.")
```

## Complete Example

See `/examples/09_langgraph_agent.py` for a complete working example.

## Best Practices

1. **Initialize Once**: Create `HybridRAG` instance once and reuse it
2. **Use Async**: All HybridRAG operations are async, use `@tool` with async functions
3. **Handle Errors**: Wrap tool functions in try/except for graceful degradation
4. **Clear Tool Descriptions**: Agent decides tool use based on descriptions
5. **Mode Selection**: Use `mode="mix"` for best quality, `mode="naive"` for speed

## Troubleshooting

### Agent Never Uses Tools
- Check tool descriptions are clear about when to use them
- Add system prompt explaining when to search vs. answer directly

### Tool Returns Empty Results
- Verify documents are ingested: `await rag.get_knowledge_base_stats()`
- Try different query modes (`mix`, `hybrid`, `naive`)
- Check if query is too vague

### Slow Responses
- Reduce `top_k` in query
- Use `mode="naive"` for faster (vector-only) search
- Consider caching frequent queries
