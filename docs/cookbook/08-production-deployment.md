# Recipe 08: Production Deployment Guide

Deploy HybridRAG applications at scale with MongoDB Atlas.

## Overview

This guide covers everything needed to take your RAG application from development to production:

- Index configuration and optimization
- Scaling strategies
- Monitoring and observability
- Security hardening
- Performance tuning checklist

## Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Load Balancer                                │
│                    (HTTPS, Rate Limiting)                           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   App Server    │ │   App Server    │ │   App Server    │
│   (HybridRAG)   │ │   (HybridRAG)   │ │   (HybridRAG)   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MongoDB Atlas Cluster                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Primary    │  │  Secondary  │  │  Secondary  │                  │
│  │  (Writes)   │  │  (Reads)    │  │  (Reads)    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│                                                                      │
│  Atlas Search Nodes (Dedicated)                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Search 1   │  │  Search 2   │  │  Search 3   │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    External Services                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Voyage AI  │  │  Anthropic  │  │  Langfuse   │                  │
│  │ (Embeddings)│  │   (LLM)     │  │(Observability)│                │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Index Configuration

### 1. Vector Search Index (Production)

```javascript
{
  "name": "vector_knn_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "vector",
        "numDimensions": 1024,
        "similarity": "cosine",
        "quantization": "scalar"  // Enable for >100K documents
      },
      // Pre-filter fields
      {
        "type": "filter",
        "path": "metadata.source"
      },
      {
        "type": "filter",
        "path": "metadata.category"
      },
      {
        "type": "filter",
        "path": "metadata.timestamp"
      },
      {
        "type": "filter",
        "path": "metadata.tenant_id"  // Multi-tenant support
      }
    ]
  }
}
```

### 2. Atlas Search Index (Hybrid + Lexical Prefilters)

```javascript
{
  "name": "default",
  "type": "search",
  "definition": {
    "analyzer": "lucene.standard",
    "searchAnalyzer": "lucene.standard",
    "mappings": {
      "dynamic": false,
      "fields": {
        // Vector field for $search.vectorSearch
        "vector": {
          "type": "vector",
          "numDimensions": 1024,
          "similarity": "cosine"
        },
        // Text fields for hybrid search
        "content": {
          "type": "string",
          "analyzer": "lucene.standard",
          "indexOptions": {
            "termVector": {
              "storePositions": true
            }
          }
        },
        "title": {
          "type": "string",
          "analyzer": "lucene.standard"
        },
        // Filter fields
        "metadata.source": {
          "type": "token"
        },
        "metadata.category": {
          "type": "token"
        },
        "metadata.timestamp": {
          "type": "date"
        },
        "metadata.tenant_id": {
          "type": "token"
        },
        // Geo field for location-based filtering
        "metadata.location": {
          "type": "geo"
        }
      }
    },
    "storedSource": {
      "include": ["content", "title", "metadata"]
    }
  }
}
```

### 3. Supporting Indexes

```javascript
// Conversation sessions (TTL + queries)
db.conversation_sessions.createIndex(
  { "session_id": 1 },
  { unique: true }
)
db.conversation_sessions.createIndex(
  { "metadata.user_id": 1, "updated_at": -1 }
)
db.conversation_sessions.createIndex(
  { "updated_at": 1 },
  { expireAfterSeconds: 2592000 }  // 30 days TTL
)

// Agent memory
db.agent_memory.createIndex(
  { "agent_id": 1, "type": 1, "updated_at": -1 }
)
db.agent_memory.createIndex(
  { "agent_id": 1, "type": 1, "importance": -1 }
)

// Knowledge graph
db.entities.createIndex(
  { "normalized_name": 1 },
  { unique: true }
)
db.relationships.createIndex({ "source_entity": 1 })
db.relationships.createIndex({ "target_entity": 1 })

// Chunks with entity references
db.text_chunks.createIndex({ "entities": 1 })
db.text_chunks.createIndex({ "document_id": 1 })
```

## Scaling Strategies

### Horizontal Scaling

#### 1. Read Scaling with Read Preference

```python
from pymongo import MongoClient, ReadPreference

# Configure read preference for search workloads
client = MongoClient(
    mongodb_uri,
    readPreference="secondaryPreferred",  # Distribute reads
    maxPoolSize=100,
    minPoolSize=10,
    maxIdleTimeMS=30000,
    waitQueueTimeoutMS=5000
)

# For time-sensitive queries, use primary
async def query_with_freshness(collection, query, require_fresh=False):
    if require_fresh:
        return await collection.with_options(
            read_preference=ReadPreference.PRIMARY
        ).find(query).to_list(length=None)
    return await collection.find(query).to_list(length=None)
```

#### 2. Dedicated Search Nodes

For production Atlas clusters, enable dedicated search nodes:

```
Atlas UI → Cluster → Configuration → Search Nodes
- Enable Dedicated Search Nodes
- Node Type: M30 or higher for production
- Node Count: 3 (minimum for HA)
```

#### 3. Sharding for Large Collections

```javascript
// Shard key selection for RAG workloads
// Good: tenant_id (multi-tenant) or document_id (balanced)
sh.shardCollection(
  "hybridrag.text_chunks",
  { "metadata.tenant_id": 1, "_id": 1 }
)

// For single-tenant, use hashed sharding
sh.shardCollection(
  "hybridrag.text_chunks",
  { "_id": "hashed" }
)
```

### Vertical Scaling Guidelines

| Collection Size | Cluster Tier | Search Nodes | numCandidates |
|-----------------|--------------|--------------|---------------|
| < 100K docs | M10 | Shared | 20x limit |
| 100K - 1M docs | M30 | M30 x 3 | 15x limit |
| 1M - 10M docs | M50 | M50 x 3 | 10x limit |
| > 10M docs | M60+ | M60+ x 3 | 10x limit + quantization |

## Connection Management

### Production Connection Configuration

```python
from motor.motor_asyncio import AsyncIOMotorClient

def create_production_client(mongodb_uri: str) -> AsyncIOMotorClient:
    """Create production-ready MongoDB client."""
    return AsyncIOMotorClient(
        mongodb_uri,
        # Connection pool settings
        maxPoolSize=100,           # Max connections per host
        minPoolSize=10,            # Keep warm connections
        maxIdleTimeMS=30000,       # Close idle after 30s
        waitQueueTimeoutMS=5000,   # Fail fast if pool exhausted

        # Timeouts
        connectTimeoutMS=10000,    # 10s to establish connection
        socketTimeoutMS=30000,     # 30s for operations
        serverSelectionTimeoutMS=30000,

        # Retry settings
        retryWrites=True,
        retryReads=True,

        # Compression
        compressors=["zstd", "zlib", "snappy"],

        # Read preference for search
        readPreference="secondaryPreferred",

        # Write concern for durability
        w="majority",
        journal=True
    )
```

### Connection Pooling Best Practices

```python
import asyncio
from contextlib import asynccontextmanager

class MongoDBConnectionManager:
    """Production connection manager with health checks."""

    def __init__(self, uri: str):
        self.uri = uri
        self.client = None
        self._health_check_task = None

    async def initialize(self):
        """Initialize connection and start health checks."""
        self.client = create_production_client(self.uri)

        # Verify connection
        await self.client.admin.command("ping")

        # Start background health checks
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )

    async def _health_check_loop(self):
        """Periodic health checks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30s
                await self.client.admin.command("ping")
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    async def close(self):
        """Clean shutdown."""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self.client:
            self.client.close()

    @asynccontextmanager
    async def get_database(self, name: str):
        """Get database with automatic cleanup."""
        try:
            yield self.client[name]
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise
```

## Monitoring and Observability

### 1. Langfuse Integration

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from functools import wraps

langfuse = Langfuse()

def trace_rag_operation(operation_name: str):
    """Decorator to trace RAG operations."""
    def decorator(func):
        @wraps(func)
        @observe(name=operation_name)
        async def wrapper(*args, **kwargs):
            # Add metadata
            langfuse_context.update_current_observation(
                metadata={
                    "operation": operation_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)

                # Log metrics
                langfuse_context.update_current_observation(
                    metadata={
                        "latency_ms": (time.perf_counter() - start) * 1000,
                        "success": True
                    }
                )
                return result
            except Exception as e:
                langfuse_context.update_current_observation(
                    metadata={
                        "error": str(e),
                        "success": False
                    }
                )
                raise
        return wrapper
    return decorator

# Usage
@trace_rag_operation("hybrid_search")
async def search(query: str, top_k: int):
    # Search implementation
    pass
```

### 2. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
SEARCH_REQUESTS = Counter(
    "hybridrag_search_requests_total",
    "Total search requests",
    ["search_type", "status"]
)

SEARCH_LATENCY = Histogram(
    "hybridrag_search_latency_seconds",
    "Search latency",
    ["search_type"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

RESULTS_COUNT = Histogram(
    "hybridrag_results_count",
    "Number of results returned",
    ["search_type"],
    buckets=[0, 1, 5, 10, 20, 50, 100]
)

ACTIVE_SESSIONS = Gauge(
    "hybridrag_active_sessions",
    "Number of active conversation sessions"
)

# Instrument search
async def instrumented_search(query: str, search_type: str):
    with SEARCH_LATENCY.labels(search_type=search_type).time():
        try:
            results = await perform_search(query, search_type)
            SEARCH_REQUESTS.labels(search_type=search_type, status="success").inc()
            RESULTS_COUNT.labels(search_type=search_type).observe(len(results))
            return results
        except Exception as e:
            SEARCH_REQUESTS.labels(search_type=search_type, status="error").inc()
            raise
```

### 3. Structured Logging

```python
import structlog
from typing import Any

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
async def search_with_logging(query: str, session_id: str):
    log = logger.bind(
        session_id=session_id,
        query_length=len(query)
    )

    log.info("search_started")

    start = time.perf_counter()
    try:
        results = await perform_search(query)

        log.info(
            "search_completed",
            latency_ms=(time.perf_counter() - start) * 1000,
            results_count=len(results),
            top_score=results[0].similarity if results else 0
        )

        return results
    except Exception as e:
        log.error(
            "search_failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### 4. Atlas Monitoring Queries

```python
async def get_search_metrics(db) -> dict:
    """Get Atlas Search performance metrics."""

    # Check index status
    indexes = await db.text_chunks.list_search_indexes().to_list(length=None)

    # Get collection stats
    stats = await db.command("collStats", "text_chunks")

    # Profile slow queries (if profiling enabled)
    slow_queries = await db.system.profile.find({
        "op": "command",
        "command.aggregate": "text_chunks",
        "millis": {"$gt": 100}
    }).sort("millis", -1).limit(10).to_list(length=None)

    return {
        "indexes": [
            {
                "name": idx.get("name"),
                "status": idx.get("status"),
                "queryable": idx.get("queryable")
            }
            for idx in indexes
        ],
        "collection": {
            "document_count": stats.get("count"),
            "size_mb": stats.get("size", 0) / (1024 * 1024),
            "index_size_mb": stats.get("totalIndexSize", 0) / (1024 * 1024)
        },
        "slow_queries": len(slow_queries)
    }
```

## Security Hardening

### 1. Network Security

```python
# MongoDB Atlas Network Access
# - Enable IP Access List (whitelist your app servers)
# - Use VPC Peering or Private Link for production
# - Enable TLS 1.2+ (default in Atlas)

# Connection string with TLS
MONGODB_URI = (
    "mongodb+srv://user:password@cluster.mongodb.net/"
    "?retryWrites=true&w=majority&tls=true&tlsCAFile=/path/to/ca.pem"
)
```

### 2. Authentication & Authorization

```python
# Create database user with minimal permissions
# Atlas UI → Database Access → Add New Database User

# Role: Custom role for RAG application
{
    "roleName": "ragAppRole",
    "privileges": [
        {
            "resource": {"db": "hybridrag", "collection": "text_chunks"},
            "actions": ["find", "aggregate"]
        },
        {
            "resource": {"db": "hybridrag", "collection": "conversation_sessions"},
            "actions": ["find", "insert", "update", "delete"]
        },
        {
            "resource": {"db": "hybridrag", "collection": "agent_memory"},
            "actions": ["find", "insert", "update", "delete"]
        }
    ]
}
```

### 3. API Key Management

```python
import os
from cryptography.fernet import Fernet

class SecureKeyManager:
    """Secure API key management."""

    def __init__(self):
        # Use environment variables or secret manager
        self.encryption_key = os.environ.get("ENCRYPTION_KEY")
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode())

    def get_api_key(self, key_name: str) -> str:
        """Get API key from secure storage."""
        # Priority: Secret Manager > Environment > Encrypted File

        # 1. Try environment variable
        key = os.environ.get(key_name)
        if key:
            return key

        # 2. Try AWS Secrets Manager / GCP Secret Manager
        # (implementation depends on cloud provider)

        raise ValueError(f"API key {key_name} not found")

    def mask_key(self, key: str) -> str:
        """Mask API key for logging."""
        if len(key) < 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"
```

### 4. Input Validation

```python
from pydantic import BaseModel, Field, validator
import re

class QueryRequest(BaseModel):
    """Validated query request."""

    query: str = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(default=10, ge=1, le=100)
    session_id: str | None = Field(default=None, max_length=100)
    filters: dict | None = None

    @validator("query")
    def sanitize_query(cls, v):
        # Remove potential injection patterns
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)  # Control chars
        return v.strip()

    @validator("session_id")
    def validate_session_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError("Invalid session_id format")
        return v

    @validator("filters")
    def validate_filters(cls, v):
        if v:
            # Prevent NoSQL injection
            forbidden = ["$where", "$function", "$accumulator"]
            for key in v:
                if any(f in str(key) for f in forbidden):
                    raise ValueError("Invalid filter operator")
        return v
```

## Performance Optimization

### 1. Query Optimization

```python
from hybridrag.enhancements import MongoDBHybridSearchConfig

# Production-optimized configuration
PRODUCTION_CONFIG = MongoDBHybridSearchConfig(
    # Vector search
    vector_num_candidates=200,  # 20x for top_k=10
    cosine_threshold=0.3,       # Filter low-quality results

    # Hybrid search
    vector_weight=0.6,
    text_weight=0.4,

    # $rankFusion settings
    rank_constant=60,

    # Score details for debugging
    score_details=False,        # Disable in production for speed

    # Reranking
    enable_reranking=True,
    rerank_top_k=50,           # Rerank top 50

    # Caching
    enable_query_cache=True,
    cache_ttl_seconds=300      # 5 minute cache
)
```

### 2. Caching Strategy

```python
import hashlib
import json
from functools import lru_cache
from cachetools import TTLCache

class SearchCache:
    """Production search cache."""

    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def _make_key(self, query: str, filters: dict | None) -> str:
        """Create deterministic cache key."""
        data = {"query": query, "filters": filters}
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    async def get_or_search(
        self,
        query: str,
        filters: dict | None,
        search_func
    ):
        """Get from cache or execute search."""
        key = self._make_key(query, filters)

        if key in self.cache:
            logger.debug("cache_hit", key=key[:8])
            return self.cache[key]

        logger.debug("cache_miss", key=key[:8])
        results = await search_func()
        self.cache[key] = results

        return results

# Embedding cache (expensive operation)
@lru_cache(maxsize=10000)
def get_cached_embedding(text: str, model: str) -> tuple:
    """Cache embeddings for repeated queries."""
    # Note: Returns tuple for hashability
    embedding = voyage_client.embed([text], model=model).embeddings[0]
    return tuple(embedding)
```

### 3. Batch Operations

```python
async def batch_embed_and_insert(
    documents: list[dict],
    collection,
    batch_size: int = 100
) -> int:
    """Efficiently batch embed and insert documents."""

    total_inserted = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Batch embed
        texts = [doc["content"] for doc in batch]
        embeddings = await voyage_client.embed(
            texts=texts,
            model="voyage-4-large",
            input_type="document"
        )

        # Add vectors
        for doc, emb in zip(batch, embeddings.embeddings):
            doc["vector"] = emb

        # Batch insert
        result = await collection.insert_many(batch, ordered=False)
        total_inserted += len(result.inserted_ids)

        # Rate limiting
        await asyncio.sleep(0.1)

    return total_inserted
```

## Production Checklist

### Pre-Deployment

- [ ] **Indexes Created**
  - [ ] Vector search index with filter fields
  - [ ] Atlas Search index for hybrid search
  - [ ] Supporting indexes for sessions/memory
  - [ ] All indexes show status "READY"

- [ ] **Configuration**
  - [ ] Environment variables set (MongoDB URI, API keys)
  - [ ] Connection pool sized for expected load
  - [ ] Timeouts configured
  - [ ] Retry policies enabled

- [ ] **Security**
  - [ ] Database user has minimal required permissions
  - [ ] IP whitelist configured
  - [ ] TLS enabled
  - [ ] API keys in secret manager

### Monitoring Setup

- [ ] **Observability**
  - [ ] Langfuse tracing enabled
  - [ ] Prometheus metrics exposed
  - [ ] Structured logging configured
  - [ ] Alert thresholds defined

- [ ] **Health Checks**
  - [ ] Database connectivity check
  - [ ] Index status check
  - [ ] External API health (Voyage, Anthropic)

### Performance Validation

- [ ] **Load Testing**
  - [ ] p50 latency < 200ms
  - [ ] p99 latency < 1000ms
  - [ ] Error rate < 0.1%
  - [ ] Throughput meets requirements

- [ ] **Scaling**
  - [ ] Horizontal scaling tested
  - [ ] Connection pool exhaustion handled
  - [ ] Graceful degradation verified

### Post-Deployment

- [ ] **Monitoring Active**
  - [ ] Dashboards created
  - [ ] Alerts configured
  - [ ] On-call rotation defined

- [ ] **Runbooks**
  - [ ] Index rebuild procedure
  - [ ] Scaling procedure
  - [ ] Incident response

## Troubleshooting Guide

### Slow Queries

```python
# 1. Check explain output
async def diagnose_slow_search(collection, query_vector):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_knn_index",
                "path": "vector",
                "queryVector": query_vector,
                "numCandidates": 200,
                "limit": 10
            }
        }
    ]

    explain = await collection.aggregate(pipeline).explain()

    # Check execution stats
    stats = explain.get("executionStats", {})
    logger.info(
        "query_explain",
        execution_time_ms=stats.get("executionTimeMillis"),
        docs_examined=stats.get("totalDocsExamined"),
        keys_examined=stats.get("totalKeysExamined")
    )

    return explain

# 2. Common fixes
# - Increase numCandidates if recall is low
# - Enable quantization for large collections
# - Add pre-filters to reduce search space
# - Check if index is queryable
```

### Connection Issues

```python
async def diagnose_connection(client):
    """Diagnose connection issues."""
    try:
        # Test basic connectivity
        await client.admin.command("ping")
        logger.info("connection_ok")

        # Check topology
        topology = client.topology_description
        logger.info(
            "topology",
            type=topology.topology_type.name,
            servers=len(topology.server_descriptions())
        )

        # Check pool status
        for address, server in topology.server_descriptions().items():
            logger.info(
                "server_status",
                address=str(address),
                type=server.server_type.name
            )

    except Exception as e:
        logger.error("connection_failed", error=str(e))
        raise
```

### Index Problems

```python
async def diagnose_indexes(collection):
    """Diagnose index issues."""

    # List all search indexes
    indexes = await collection.list_search_indexes().to_list(length=None)

    issues = []
    for idx in indexes:
        status = idx.get("status")
        name = idx.get("name")

        if status != "READY":
            issues.append(f"Index {name} status: {status}")

        if not idx.get("queryable", False):
            issues.append(f"Index {name} is not queryable")

    if issues:
        logger.warning("index_issues", issues=issues)
    else:
        logger.info("indexes_healthy", count=len(indexes))

    return indexes
```

## References

- [MongoDB Atlas Best Practices](https://www.mongodb.com/docs/atlas/best-practices/)
- [Atlas Search Performance](https://www.mongodb.com/docs/atlas/atlas-search/performance/)
- [Connection Pooling Guide](https://www.mongodb.com/docs/manual/administration/connection-pool-overview/)
- [Security Checklist](https://www.mongodb.com/docs/manual/administration/security-checklist/)

---

**Previous**: [Recipe 07: Agent Memory Patterns](./07-agent-memory-patterns.md)
**Back to**: [Cookbook Overview](./README.md)
