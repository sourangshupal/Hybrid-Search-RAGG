<div align="center">

```
██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗ ██████╗  █████╗  ██████╗
██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██╔══██╗██╔════╝
███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║██████╔╝███████║██║  ███╗
██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║██╔══██╗██╔══██║██║   ██║
██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
```

# The Atomic RAG Boilerplate

### **MongoDB 8.2 Native • $rankFusion • Lexical Prefilters • Knowledge Graph**

**Stop syncing 4 databases. Store vectors, graphs, and docs in one ACID-compliant MongoDB document.**

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MongoDB](https://img.shields.io/badge/MongoDB-8.2+-47A248.svg)](https://www.mongodb.com/atlas)
[![Voyage AI](https://img.shields.io/badge/Voyage_AI-Embeddings-purple.svg)](https://www.voyageai.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-features) • [MongoDB 8.2](#-mongodb-82-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 The Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│  THE FRAGMENTED WAY                                                      │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Pinecone │  │  Neo4j   │  │  Redis   │  │ Postgres │                 │
│  │ Vectors  │  │  Graph   │  │  Cache   │  │ Metadata │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │             │             │             │                        │
│       └─────────────┴─────────────┴─────────────┘                        │
│                         │                                                │
│                    SYNC HELL 😱                                          │
│         If one write fails, your RAG returns                             │
│         vectors for deleted text                                         │
└─────────────────────────────────────────────────────────────────────────┘

                              VS

┌─────────────────────────────────────────────────────────────────────────┐
│  THE HYBRIDRAG WAY                                                       │
│                                                                          │
│                    ┌─────────────────────┐                               │
│                    │   MongoDB Atlas     │                               │
│                    │  ┌───┐ ┌───┐ ┌───┐  │                               │
│                    │  │ V │ │ G │ │ K │  │                               │
│                    │  └───┘ └───┘ └───┘  │                               │
│                    │  Vector Graph  KV   │                               │
│                    └─────────────────────┘                               │
│                              │                                           │
│                    ONE DOCUMENT = ATOMIC ✅                              │
│              All or nothing. Never inconsistent.                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔄 Core Capabilities
| Feature | Description |
|---------|-------------|
| **Atomic Updates** | Vector + metadata + graph in one transaction |
| **$rankFusion** | Native MongoDB 8.2 weighted hybrid search |
| **$scoreFusion** | Score-based fusion with normalization |
| **Knowledge Graph** | Automatic entity & relationship extraction |
| **Self-Compacting Memory** | Conversations auto-summarize |

</td>
<td width="50%">

### 🚀 MongoDB 8.2 Native
| Feature | Description |
|---------|-------------|
| **Lexical Prefilters** | Fuzzy, phrase, wildcard BEFORE vectors |
| **Dynamic numCandidates** | Auto-tuned (top_k × 20) |
| **scoreDetails** | Per-pipeline score debugging |
| **Explicit Weights** | Configurable vector/text weights |
| **Graceful Fallback** | Auto-degrades for older MongoDB |

</td>
</tr>
</table>

### 🔌 Integrations

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EMBEDDINGS       │  LLM PROVIDERS    │  OBSERVABILITY   │  UI           │
│  ────────────     │  ─────────────    │  ────────────    │  ──           │
│  ✓ Voyage AI      │  ✓ Claude         │  ✓ Langfuse      │  ✓ Chainlit   │
│  ✓ voyage-3-large │  ✓ GPT-4          │  ✓ RAGAS Eval    │  ✓ Rich CLI   │
│  ✓ Reranking      │  ✓ Gemini         │                  │  ✓ REST API   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🆕 MongoDB 8.2 Features

HybridRAG is built for **MongoDB 8.2** with native support for the latest search operators.

### Three Filter Systems

```python
from hybridrag import (
    # 1. Vector Search Filters (MQL - for $vectorSearch)
    VectorSearchFilterConfig,
    build_vector_search_filters,

    # 2. Atlas Search Filters (for $search compound queries)
    AtlasSearchFilterConfig,
    build_atlas_search_filters,

    # 3. Lexical Prefilters (NEW - for $search.vectorSearch)
    LexicalPrefilterConfig,
    build_lexical_prefilters,
    TextFilter, FuzzyFilter, PhraseFilter, WildcardFilter, GeoFilter,
)
```

### Lexical Prefilters (MongoDB 8.2+)

**The game-changer**: Apply Atlas Search operators (fuzzy, phrase, wildcard, geo) **BEFORE** vector search.

```python
from hybridrag import LexicalPrefilterConfig, HybridRAG

# Create a lexical prefilter config
filter_config = LexicalPrefilterConfig(
    # Fuzzy text matching (typo-tolerant)
    fuzzy_filters=[{"path": "content", "query": "machin lerning", "maxEdits": 2}],

    # Exact phrase matching
    phrase_filters=[{"path": "title", "query": "vector database"}],

    # Wildcard patterns
    wildcard_filters=[{"path": "tags", "query": "tech*"}],

    # Date range filtering
    range_filters={"timestamp": {"gte": "2024-01-01"}},

    # Geospatial (find docs near a location)
    geo_filters=[{"path": "location", "geometry": {"type": "Point", "coordinates": [-73.9, 40.7]}}],
)

# Use with hybrid search
results = await rag.query(
    query="machine learning best practices",
    mode="hybrid",
    lexical_filter_config=filter_config,
)
```

### Why Lexical Prefilters Matter

| Scenario | Legacy $vectorSearch | New $search.vectorSearch |
|----------|---------------------|--------------------------|
| "Find docs about *machin lerning*" | ❌ No fuzzy support | ✅ `fuzzy: {maxEdits: 2}` |
| "Exact phrase 'machine learning'" | ❌ Vector similarity only | ✅ `phrase: {slop: 0}` |
| "Tags matching tech*" | ❌ No wildcards | ✅ `wildcard: {query: "tech*"}` |
| "Docs within 10km of NYC" | ❌ No geo filtering | ✅ `geoWithin` |
| "Combined filters" | ❌ MQL only ($eq, $gte) | ✅ Full Atlas Search syntax |

### $meta Score Fields Reference

```python
# CRITICAL: Each operator uses a DIFFERENT $meta field!
OPERATOR_SCORE_FIELDS = {
    "$vectorSearch":        "vectorSearchScore",   # Legacy
    "$search.vectorSearch": "searchScore",         # MongoDB 8.2+
    "$rankFusion":          "rankFusionScore",
    "$scoreFusion":         "scoreFusionScore",
}
```

---

## 🔀 How Hybrid Search Works

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    $rankFusion (MongoDB 8.2 Native)                  │
  │                                                                      │
  │   ┌───────────────────────┐          ┌───────────────────────┐      │
  │   │    VECTOR PIPELINE    │          │    TEXT PIPELINE      │      │
  │   │  $search.vectorSearch │          │  $search.compound     │      │
  │   │                       │          │                       │      │
  │   │  ┌─────────────────┐  │          │  ┌─────────────────┐  │      │
  │   │  │ Lexical Prefilter│  │          │  │  Fuzzy Matching │  │      │
  │   │  │ (fuzzy/phrase/  │  │          │  │                 │  │      │
  │   │  │  wildcard/geo)  │  │          │  │                 │  │      │
  │   │  └────────┬────────┘  │          │  └────────┬────────┘  │      │
  │   │           ↓           │          │           ↓           │      │
  │   │  ┌─────────────────┐  │          │  ┌─────────────────┐  │      │
  │   │  │ Vector Similarity│  │          │  │  BM25 Scoring   │  │      │
  │   │  └────────┬────────┘  │          │  └────────┬────────┘  │      │
  │   └───────────┼───────────┘          └───────────┼───────────┘      │
  │               │                                  │                  │
  │               └──────────────┬───────────────────┘                  │
  │                              ↓                                      │
  │                  ┌───────────────────────┐                          │
  │                  │  Weighted Fusion      │                          │
  │                  │  vector: 0.6          │                          │
  │                  │  text:   0.4          │                          │
  │                  │  scoreDetails: true   │                          │
  │                  └───────────────────────┘                          │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │              KNOWLEDGE GRAPH ($graphLookup)                          │
  │                                                                      │
  │   Entity Boosting: KG relationships enhance reranking scores         │
  │   Mix Mode: Combines vector + text + graph for comprehensive RAG     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/romiluz13/HybridRAG.git
cd HybridRAG

# First-time setup (recommended)
make first-time-setup

# Or manual installation
pip install -e ".[all]"
```

### Configuration

```bash
# Create .env file
cat > .env << EOF
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DATABASE=hybridrag
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
EOF
```

### Basic Usage

```python
import asyncio
from hybridrag import create_hybridrag, LexicalPrefilterConfig

async def main():
    # Initialize
    rag = await create_hybridrag()

    # Ingest documents
    await rag.ingest_files("./documents/")

    # Simple query
    result = await rag.query_with_memory(
        query="What are the key findings?",
        mode="mix",  # Vector + Graph + Keyword
    )
    print(result["answer"])

    # Advanced: Query with lexical prefilters
    filter_config = LexicalPrefilterConfig(
        fuzzy_filters=[{"path": "content", "query": "machin lerning", "maxEdits": 2}],
        range_filters={"timestamp": {"gte": "2024-01-01"}},
    )

    result = await rag.query(
        query="machine learning trends",
        mode="hybrid",
        lexical_filter_config=filter_config,
    )

asyncio.run(main())
```

### CLI

```bash
# Launch interactive CLI
hybridrag chat

# Or use Typer commands
hybridrag ingest ./documents/
hybridrag query "What is MongoDB Atlas?"
hybridrag status
hybridrag benchmark
```

---

## 📊 Query Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `mix` | KG + Vector + Keyword | **Recommended** - General queries |
| `hybrid` | Vector + Keyword ($rankFusion) | Fast hybrid search |
| `local` | Entity-focused retrieval | Specific entities |
| `global` | Community summaries | High-level overview |
| `naive` | Vector search only | Simple similarity |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HybridRAG                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │   Voyage AI    │  │  Claude/GPT/   │  │      MongoDB Atlas 8.2     │ │
│  │   Embeddings   │  │    Gemini      │  │                            │ │
│  │   + Reranking  │  │                │  │  ┌──────────────────────┐  │ │
│  └────────────────┘  └────────────────┘  │  │ $rankFusion          │  │ │
│                                          │  │ $scoreFusion         │  │ │
│                                          │  │ $search.vectorSearch │  │ │
│                                          │  │ $graphLookup         │  │ │
│                                          │  └──────────────────────┘  │ │
│                                          └────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         FILTER SYSTEMS                              ││
│  │  VectorSearchFilterConfig │ AtlasSearchFilterConfig │ LexicalPrefilter││
│  │  (MQL: $eq, $gte, $in)    │ (Atlas: range, equals)  │ (fuzzy,phrase) ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         ENHANCEMENTS                                 ││
│  │  Entity Boosting │ Query Optimizer │ Self-Compacting Memory         ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        INTERFACES                                    ││
│  │        Chainlit UI  │  Typer CLI  │  REST API  │  Python SDK        ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/installation.md) | Setup and configuration |
| [Architecture Decisions](docs/adr/) | ADRs for key decisions |
| [Enhanced Search](docs/enhanced-search.md) | Graph traversal, mix mode |
| [Notebooks](notebooks/) | Interactive tutorials |
| [Examples](examples/) | Code examples |

### Architecture Decision Records

- [ADR-001: MongoDB Single Database](docs/adr/0001-mongodb-single-database.md)
- [ADR-002: Voyage AI Embeddings](docs/adr/0002-voyage-ai-embeddings.md)
- [ADR-003: Hybrid Search RRF](docs/adr/0003-hybrid-search-rrf.md)
- [ADR-004: Prompts Module](docs/adr/0004-prompts-module-architecture.md)
- [ADR-005: Filter Builder Systems](docs/adr/0005-filter-builder-systems.md)
- [ADR-006: Lexical Prefilters](docs/adr/0006-lexical-prefilters.md)

---

## 🧪 Development

```bash
# Setup development environment
make first-time-setup

# Run tests
make test              # All tests
make test-quick        # Fast unit tests
make test-cov          # With coverage

# Code quality
make lint              # Ruff linting
make format            # Auto-format
make typecheck         # MyPy

# Full CI suite
make ci
```

---

## 📊 Why MongoDB Over Postgres?

| Task | Postgres + pgvector | HybridRAG + MongoDB |
|------|---------------------|---------------------|
| Add metadata field | `ALTER TABLE` + backfill + reindex | Just add it |
| Change embedding model | Rewrite entire table (MVCC bloat) | Bulk update, no rewrite |
| Hybrid search | Manual result merging in app code | Single `$rankFusion` pipeline |
| Lexical prefilters | Not supported | `$search.vectorSearch` native |
| Filter vectors by metadata | Separate index, query planner struggles | Compound index, native |
| Time to first query | Hours (extensions, schema, indexes) | 30 minutes (Atlas free tier) |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/romiluz13/HybridRAG.git
cd HybridRAG
make first-time-setup

# Run tests before submitting
make ci
```

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   MongoDB 8.2 Native: $rankFusion • $scoreFusion • $search.vectorSearch   ║
║                                                                           ║
║   Three Filter Systems: Vector (MQL) • Atlas • Lexical Prefilters         ║
║                                                                           ║
║   One MongoDB document. Atomic updates. Never inconsistent.               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Built with MongoDB 8.2 • Voyage AI • Claude**

[⬆ Back to Top](#)

</div>
