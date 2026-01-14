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

**Stop syncing 4 databases. Store vectors, graphs, and docs in one ACID-compliant MongoDB document.**

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)](https://www.mongodb.com/atlas)
[![Voyage AI](https://img.shields.io/badge/Voyage_AI-Embeddings-purple.svg)](https://www.voyageai.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-hybrid-search-works) • [Documentation](#-documentation) • [Contributing](#-contributing)

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

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  🔄 ATOMIC UPDATES         Vector + metadata + graph in one transaction │
│  🔍 HYBRID SEARCH          Vector + Keyword via RRF, Graph via modes    │
│  🧠 KNOWLEDGE GRAPH        Automatic entity & relationship extraction   │
│  💬 SELF-COMPACTING MEMORY Conversations auto-summarize, never lost     │
│  🚀 ENTITY BOOSTING        Knowledge graph enhances vector reranking    │
│  📊 RAGAS EVALUATION       Built-in RAG quality metrics                 │
│  🔌 MULTI-LLM              Gemini, Claude, OpenAI - switch anytime      │
│  📈 LANGFUSE TRACING       Production observability built-in            │
│  🎨 CHAINLIT UI            Beautiful web chat interface                 │
│  ⚡ VOYAGE AI              State-of-the-art embeddings + reranking      │
│  🌐 TAVILY INTEGRATION     Web content extraction & crawling            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 How Hybrid Search Works

HybridRAG combines multiple retrieval methods:

```
  ┌─────────────────────────────────────────────────────┐
  │                  RRF FUSION ($rankFusion)           │
  │                                                     │
  │   ┌─────────────┐              ┌─────────────┐      │
  │   │   VECTOR    │              │   KEYWORD   │      │
  │   │   SEARCH    │              │   SEARCH    │      │
  │   │  Semantic   │              │   Text      │      │
  │   │  Similarity │              │  Matching   │      │
  │   └──────┬──────┘              └──────┬──────┘      │
  │          │                            │             │
  │          └────────────┬───────────────┘             │
  │                       │                             │
  │              ┌────────▼────────┐                    │
  │              │  RRF(d) = Σ 1   │                    │
  │              │         ─────   │                    │
  │              │         k + r   │                    │
  │              └─────────────────┘                    │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │              GRAPH SEARCH ($graphLookup)            │
  │                                                     │
  │   ┌─────────────┐    Used in mix/local modes        │
  │   │  KNOWLEDGE  │    for entity traversal.          │
  │   │    GRAPH    │    Enhances results via           │
  │   │  Entities & │    Entity Boosting.               │
  │   │  Relations  │                                   │
  │   └─────────────┘                                   │
  └─────────────────────────────────────────────────────┘
```

**RRF** combines Vector + Keyword results. **Graph** search via `$graphLookup` traverses entity relationships separately in `mix` and `local` query modes.

---

## 🚀 Quick Start

### System Requirements

**Hardware:**
- **CPU**: 2+ cores (x86_64 or ARM64)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 5 GB free space minimum
- **Network**: Stable internet connection

**Software:**
- **Python**: 3.11 or higher (3.12 recommended)
- **MongoDB**: MongoDB Community Edition (recommended for free tier) or MongoDB Atlas M10+ (for production)
  - **Note**: Atlas M0 free tier has a 3-index limit that prevents full hybrid search - use Community Edition for unlimited indexes
- **API Keys**: Voyage AI (required) + at least one LLM provider (Anthropic/OpenAI/Gemini)
- **Optional**: Tavily API key for web content ingestion

**Note**: No GPU required! All embeddings and LLM inference are handled via API calls.

### Installation

```bash
# Full installation with all features
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
pip install -e ".[all]"
```

For detailed installation instructions, see the [Installation Guide](docs/installation.md).

### Configuration

```bash
# Create .env file
cat > .env << EOF
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DATABASE=hybridrag
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx  # Optional: for web ingestion
EOF
```

### Launch Web UI

```bash
chainlit run src/hybridrag/ui/chat.py
```

Then open `http://localhost:8000` - drag & drop files to ingest, ask questions!

---

## 📖 Usage

### Python SDK

```python
import asyncio
from hybridrag import create_hybridrag

async def main():
    # Initialize (auto-initializes by default)
    rag = await create_hybridrag()

    # Ingest documents from folder (uses Docling processor)
    results = await rag.ingest_files("path/to/documents/")
    
    # Or ingest web content via Tavily
    result = await rag.ingest_url("https://docs.mongodb.com/atlas/")
    results = await rag.ingest_website("https://example.com", max_pages=10)
    
    # Or insert raw text directly
    await rag.insert(["Document 1 content...", "Document 2 content..."])

    # Query with conversation memory
    session_id = await rag.create_conversation_session()

    result = await rag.query_with_memory(
        query="What are the key findings?",
        session_id=session_id,
        mode="mix",  # Vector + Graph + Keyword
    )

    print(result["answer"])

asyncio.run(main())
```

### Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `mix` | KG + Vector + Keyword (recommended) | General queries |
| `local` | Entity-focused retrieval | Specific entities |
| `global` | Community summaries | High-level overview |
| `hybrid` | Local + Global combined | Comprehensive answers |
| `naive` | Vector search only | Simple similarity |
| `bypass` | Skip retrieval, direct LLM | Testing/debugging |

### CLI Interface

```bash
hybridrag  # Launch interactive CLI

# Commands:
# > ingest path/to/file.pdf
# > ingest-url https://docs.mongodb.com/atlas/
# > ingest-website https://example.com 10
# > What is this document about?
# > /mode mix
# > /status
# > exit
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HybridRAG                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │   Voyage AI    │  │  Claude/GPT/   │  │      MongoDB Atlas         │ │
│  │   Embeddings   │  │    Gemini      │  │                            │ │
│  │   + Reranking  │  │                │  │  ┌──────┐ ┌──────┐ ┌────┐  │ │
│  └────────────────┘  └────────────────┘  │  │Vector│ │Graph │ │ KV │  │ │
│                                          │  └──────┘ └──────┘ └────┘  │ │
│                                          └────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         ENHANCEMENTS                                 ││
│  │  Entity Boosting │ Implicit Expansion │ Self-Compacting Memory      ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        INTERFACES                                    ││
│  │        Chainlit UI  │  Rich CLI  │  REST API  │  Python SDK         ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Why Not Postgres?

| Task | Postgres + pgvector | HybridRAG |
|------|---------------------|-----------|
| Add metadata field | `ALTER TABLE` + backfill + reindex | Just add it |
| Change embedding model | Rewrite entire table (MVCC bloat) | Bulk update, no rewrite |
| Hybrid search | Manual result merging in app code | Single aggregation pipeline |
| Filter vectors by metadata | Separate index, query planner struggles | Compound index, native |
| Time to first query | Hours (extensions, schema, indexes) | 30 minutes (Atlas free tier) |

---

## 🔧 Configuration

```python
from hybridrag import Settings

settings = Settings(
    # MongoDB
    mongodb_database="hybridrag",

    # Embeddings
    embedding_model="voyage-3-large",
    embedding_dimensions=1024,

    # Reranking
    rerank_model="rerank-2.5",
    rerank_top_k=10,

    # LLM
    llm_provider="anthropic",  # or "openai", "gemini"
    llm_model="claude-sonnet-4-20250514",

    # Memory
    memory_max_tokens=32000,  # Self-compaction threshold
    
    # Web Ingestion (optional)
    tavily_api_key="tvly-xxxxxxxxxxxxx",  # For ingest_url() and ingest_website()
)
```

### Web Content Ingestion

HybridRAG supports web content ingestion via [Tavily](https://tavily.com) API:

```python
# Extract content from a single URL
result = await rag.ingest_url("https://docs.mongodb.com/atlas/vector-search/")

# Crawl and ingest multiple pages from a website
results = await rag.ingest_website(
    "https://docs.mongodb.com/atlas/",
    max_pages=10,
    max_depth=2
)

# Check results
for r in results:
    if r.success:
        print(f"✓ {r.title}: {r.chunks_created} chunks")
```

**Features:**
- RAG-optimized markdown content extraction
- Automatic chunking and knowledge graph extraction
- Same pipeline as file ingestion
- CLI commands: `ingest-url` and `ingest-website`
- UI actions: "🌐 Ingest URL" and "🕷️ Crawl Website" buttons

**Get your Tavily API key:** https://tavily.com

---

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Options](docs/configuration.md)
- [Query Modes Explained](docs/query-modes.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ && isort src/
```

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Vector + Keyword = RRF Fusion ($rankFusion)                 ║
║   Graph (KG) = Entity traversal in mix/local modes            ║
║                                                               ║
║   One MongoDB document. Atomic updates. Never inconsistent.   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

**Made with ❤️ for the RAG community**

[⬆ Back to Top](#)

</div>
