# Configuration Guide

Complete guide to configuring HybridRAG settings and options.

## Environment Variables

All configuration is done via environment variables in a `.env` file or system environment.

### Required Variables

```bash
# MongoDB Atlas Connection
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DATABASE=hybridrag

# Voyage AI (Required for embeddings)
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx

# LLM Provider (At least one required)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
# OR
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
# OR
GEMINI_API_KEY=xxxxxxxxxxxxx
```

### Optional Variables

```bash
# Observability (Langfuse)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## Python Configuration

Configure HybridRAG programmatically using the `Settings` class:

```python
from hybridrag import Settings
from pydantic import SecretStr

# Settings loads from environment variables by default
# Or create explicitly:
settings = Settings(
    # MongoDB Configuration (required)
    mongodb_uri=SecretStr("mongodb+srv://..."),
    mongodb_database="hybridrag",
    mongodb_workspace="default",
    
    # Voyage AI Configuration (required for embeddings)
    voyage_api_key=SecretStr("pa-..."),
    voyage_embedding_model="voyage-3-large",  # or "voyage-3", "voyage-context-3"
    voyage_context_model="voyage-context-3",
    voyage_rerank_model="rerank-2.5",
    
    # LLM Provider Configuration (choose one)
    llm_provider="anthropic",  # "anthropic", "openai", or "gemini"
    anthropic_api_key=SecretStr("sk-ant-..."),
    anthropic_model="claude-sonnet-4-20250514",
    # OR
    # openai_api_key=SecretStr("sk-..."),
    # openai_model="gpt-4o",
    # OR
    # gemini_api_key=SecretStr("..."),
    # gemini_model="gemini-2.5-flash",
    
    # Query Configuration
    default_query_mode="mix",  # "mix", "local", "global", "hybrid", "naive", "bypass"
    default_top_k=60,  # Number of results to retrieve
    default_rerank_top_k=10,  # Number of results after reranking
    enable_rerank=True,
    
    # Enhancement Configuration
    enable_implicit_expansion=True,
    implicit_expansion_threshold=0.75,
    implicit_expansion_max=10,
    enable_entity_boosting=True,
    entity_boost_weight=0.2,
    
    # Embedding Configuration
    embedding_dim=1024,  # 1024 for voyage-3-large
    max_token_size=4096,
    embedding_batch_size=128,
    
    # Context Limits
    max_token_for_text_unit=4000,
    max_token_for_local_context=4000,
    max_token_for_global_context=4000,
    
    # Observability (optional)
    langfuse_public_key="pk-lf-...",
    langfuse_secret_key=SecretStr("sk-lf-..."),
    langfuse_host="https://cloud.langfuse.com",
)
```

## Configuration Options

### Embedding Models

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| `voyage-3-large` | 1024 | Best quality, recommended (default) |
| `voyage-3` | 1536 | Balanced quality/speed |
| `voyage-context-3` | 1024 | Context-aware embeddings (+13% recall) |
| `voyage-code-3` | 1024 | Code-specific embeddings |

**Note**: Only Voyage AI embeddings are supported (no OpenAI/Gemini fallback).

### Reranking Models

| Model | Description |
|-------|-------------|
| `rerank-2.5` | Latest Voyage reranking model (recommended) |

### LLM Providers

**Anthropic Claude:**
- `claude-sonnet-4-20250514` (recommended)
- `claude-opus-3-20240229`
- `claude-haiku-3-20240307`

**OpenAI:**
- `gpt-4-turbo-preview`
- `gpt-4`
- `gpt-3.5-turbo`

**Google Gemini:**
- `gemini-2.5-flash` (default)
- `gemini-2.0-flash`

### Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `mix` | KG + Vector + Keyword (recommended) | General queries |
| `local` | Entity-focused retrieval | Specific entities |
| `global` | Community summaries | High-level overview |
| `hybrid` | Local + Global combined | Comprehensive answers |
| `naive` | Vector search only | Simple similarity |
| `bypass` | Skip retrieval, direct LLM | Testing/debugging |

## Advanced Configuration

### MongoDB Index Configuration

HybridRAG automatically creates indexes, but you can customize:

```python
settings = Settings(
    # Vector search index configuration
    vector_index_name="vector_index",
    vector_index_type="hnsw",  # "hnsw" or "ivf"
    
    # Text search index
    text_index_name="text_index",
    
    # Knowledge graph indexes
    entity_index_enabled=True,
    relationship_index_enabled=True,
)
```

### Memory Configuration

Control conversation memory behavior:

```python
settings = Settings(
    # Self-compacting memory
    memory_max_tokens=32000,  # Threshold for auto-compaction
    memory_compaction_strategy="summarize",  # "summarize" or "truncate"
    
    # Session management
    session_timeout_hours=24,  # Auto-cleanup old sessions
    max_sessions_per_user=100,
)
```

### Performance Tuning

```python
settings = Settings(
    # Concurrency
    max_concurrent_requests=10,
    embedding_batch_size=32,
    
    # Caching
    enable_embedding_cache=True,
    enable_query_cache=True,
    cache_ttl_seconds=3600,
    
    # Timeouts
    api_timeout_seconds=30,
    retry_attempts=3,
)
```

## Configuration Files

### Using pyproject.toml

You can also configure defaults in `pyproject.toml`:

```toml
[tool.hybridrag]
mongodb_database = "hybridrag"
embedding_model = "voyage-3-large"
llm_provider = "anthropic"
default_query_mode = "mix"
```

### Using YAML Config

Create `config.yaml`:

```yaml
mongodb:
  database: hybridrag
  
embedding:
  model: voyage-3-large
  dimensions: 1024
  
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  
query:
  default_mode: mix
  vector_top_k: 20
  keyword_top_k: 20
```

Load it:

```python
from hybridrag import Settings
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)
    
settings = Settings(**config)
```

## Environment-Specific Configuration

### Development

```bash
# .env.development
MONGODB_DATABASE=hybridrag_dev
LOG_LEVEL=DEBUG
```

### Production

```bash
# .env.production
MONGODB_DATABASE=hybridrag_prod
LOG_LEVEL=INFO
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Load based on environment:

```python
import os
from hybridrag import Settings

env = os.getenv("ENVIRONMENT", "development")
settings = Settings.from_env_file(f".env.{env}")
```

## Validation

HybridRAG validates configuration on startup:

```python
from hybridrag import Settings

try:
    settings = Settings()
    print("✓ Configuration valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

## Best Practices

1. **Never commit `.env` files** - Use `.env.example` as a template
2. **Use environment variables for secrets** - Never hardcode API keys
3. **Test configuration** - Validate settings before deployment
4. **Use different databases** - Separate dev/staging/production
5. **Monitor API usage** - Set up alerts for API key quotas

## Next Steps

- [Installation Guide](installation.md) - Set up HybridRAG
- [Query Modes](query-modes.md) - Understand query modes
- [API Reference](api.md) - Use the Python SDK

