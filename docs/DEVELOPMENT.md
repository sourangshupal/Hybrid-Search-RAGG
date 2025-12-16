# HybridRAG - AI Development Instructions

## CRITICAL: What We Are Building

**We are enhancing LightRAG (existing open-source project, 25K+ stars) with:**
1. **MongoDB Atlas** storage (instead of default file-based)
2. **Voyage AI** embeddings (instead of OpenAI)
3. **Voyage AI** reranking (rerank-2.5)
4. **Our enhancements**: Implicit expansion, entity boosting, contextualized embeddings

**We are NOT building a new RAG system from scratch.**
**We are NOT implementing "Ecphory" methodology.**
**We ARE forking and enhancing LightRAG.**

---

## GOLDEN RULES

1. **LightRAG IS THE BASE** - Don't reinvent what LightRAG already does
2. **WE ONLY ADD** - MongoDB storage, Voyage AI, and our enhancements
3. **FOLLOW RESEARCH** - The 6 docs in `research/` are the source of truth
4. **UPDATE BUILD-PROCESS** - Track all progress and decisions

---

## Build & Test Commands

```bash
# Environment Setup
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt

# Development
python -m pytest tests/ -v           # Run tests
python -m pytest tests/ -v --cov     # Tests with coverage
python -m black src/ tests/          # Format code
python -m isort src/ tests/          # Sort imports
python -m mypy src/                  # Type checking

# LightRAG Fork
cd lightrag-fork
pip install -e .                     # Install in dev mode

# Run Examples
python examples/basic_usage.py       # Basic RAG test
python examples/mongodb_test.py      # MongoDB connection test

# FastAPI Server (Production)
uvicorn src.api.main:app --reload    # Dev server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000  # Production
```

---

## Code Style

- **Python**: 3.11+, type hints required
- **Formatter**: Black (88 char line length)
- **Imports**: isort with Black profile
- **Naming**: `snake_case` functions, `PascalCase` classes
- **Docs**: Google-style docstrings

---

## Project Structure

```
HybridRAG/
├── CLAUDE.md                    # THIS FILE - AI instructions
├── research/                    # SOURCE OF TRUTH (6 docs)
│   ├── 00-index.md              # Navigation
│   ├── 00-internal-knowledge-base.md  # MongoDB APIs, LightRAG internals
│   ├── 01-core-architecture.md  # LightRAG architecture, integration points
│   ├── 02-mongodb-voyage-integration.md  # How to plug in Voyage + MongoDB
│   ├── 03-enhancements.md       # OUR additions on top of LightRAG
│   └── 04-production.md         # FastAPI, Langfuse, RAGAS
├── planning/                    # Implementation roadmap
├── build-process/               # AI memory (progress, decisions, learnings)
├── lightrag-fork/               # FORKED LightRAG codebase
├── src/                         # Our enhancement code (to be built)
└── tests/                       # Test suite (to be built)
```

---

## What LightRAG Already Provides (DON'T REBUILD)

LightRAG already has:
- Entity extraction (prompts in `lightrag/prompt.py`)
- Relationship extraction
- Knowledge graph storage
- Vector storage
- 6 query modes (local, global, hybrid, naive, mix, bypass)
- Chunking (in `lightrag/operate.py`)
- Query processing

**Our job is to PLUG IN MongoDB + Voyage AI, not rebuild these.**

---

## What WE Add (Our Enhancements)

| Enhancement | Description | Doc |
|-------------|-------------|-----|
| MongoDB Storage | Replace file-based with MongoDB Atlas | 02-mongodb-voyage-integration.md |
| Voyage Embeddings | voyage-3-large, voyage-context-3 | 02-mongodb-voyage-integration.md |
| Voyage Reranking | rerank-2.5 | 02-mongodb-voyage-integration.md |
| Implicit Expansion | Find related entities via vector similarity | 03-enhancements.md |
| Entity Boosting | Boost chunks with relevant entities | 03-enhancements.md |
| Contextualized Embeddings | +13% recall with voyage-context-3 | 03-enhancements.md |

---

## Research Documentation (6 Documents)

| Doc | Purpose |
|-----|---------|
| **00-internal-knowledge-base.md** | MongoDB APIs, LightRAG internals - **READ FIRST** |
| **00-index.md** | Navigation and quick reference |
| **01-core-architecture.md** | LightRAG architecture, where our code plugs in |
| **02-mongodb-voyage-integration.md** | Voyage AI wrappers, MongoDB storage config |
| **03-enhancements.md** | Our unique additions (implicit expansion, entity boosting) |
| **04-production.md** | FastAPI, Langfuse, RAGAS, Docker |

---

## Core Integration Points

From LightRAG's `lightrag.py`:

```python
# WHERE OUR CODE PLUGS IN
rag = LightRAG(  # Note: LightRAG class, not "EcphoryRAG" or "HybridRAG"
    # Voyage AI embeddings (our addition)
    embedding_func=create_voyage_embedding_func(),

    # Voyage AI reranker (our addition)
    rerank_model_func=create_voyage_rerank_func(),

    # MongoDB storage (our addition - instead of file-based)
    kv_storage="MongoKVStorage",
    vector_storage="MongoVectorDBStorage",
    graph_storage="MongoGraphStorage",
    doc_status_storage="MongoDocStatusStorage",

    # LLM (Claude)
    llm_model_func=create_claude_llm_func(),
)
```

---

## Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Base Framework** | LightRAG | Existing project we're enhancing |
| **Database** | MongoDB Atlas | Our replacement for file storage |
| **Embeddings** | Voyage AI | Our replacement for OpenAI |
| **Reranking** | Voyage AI rerank-2.5 | Our addition |
| **LLM** | Anthropic Claude | For generation |
| **API** | FastAPI | Production endpoints |
| **Observability** | Langfuse | Tracing |
| **Evaluation** | RAGAS | Testing |

---

## Environment Variables

```bash
# MongoDB Atlas
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=hybridrag

# Voyage AI
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxx

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxx
```

---

## Key LightRAG Files

| File | Purpose |
|------|---------|
| `lightrag/lightrag.py` | Main LightRAG class (4034 lines) |
| `lightrag/kg/mongo_impl.py` | MongoDB storage classes (2480 lines) |
| `lightrag/base.py` | EmbeddingFunc, QueryParam definitions |
| `lightrag/prompt.py` | Entity/relationship extraction prompts |
| `lightrag/operate.py` | Chunking, query operations |

---

## Query Modes (Built into LightRAG)

| Mode | Description |
|------|-------------|
| `local` | Entity-focused retrieval |
| `global` | Community summaries |
| `hybrid` | Local + global |
| `mix` | KG + vector (recommended) |
| `naive` | Vector only |
| `bypass` | Direct LLM |

---

## Session Workflow

```
START SESSION:
1. Read build-process/progress/ for current state
2. Read planning/README.md - What phase is active?
3. Read relevant research docs

DURING SESSION:
4. Execute tasks from planning phase
5. Update build-process/ as you go

END SESSION:
6. Update build-process/progress/
7. List pending items for next session
```

---

## Current Status

**What We're Building**: LightRAG + MongoDB + Voyage AI enhancements
**Research**: COMPLETE (6 focused documents)
**Next**: Phase 1 - Fork Analysis
**Last Updated**: 2025-12-08

---

*This file is the AI's primary instruction set. Follow it exactly.*
