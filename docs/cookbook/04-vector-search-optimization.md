# Recipe 04: Vector Search Optimization

Maximize retrieval quality and performance with MongoDB Atlas Vector Search.

## Overview

Vector search is the foundation of modern RAG systems. This recipe covers:

- Index configuration for optimal recall
- numCandidates tuning
- Quantization for scale
- Pre-filtering strategies
- Performance monitoring

## Index Configuration

### Basic Vector Search Index

```javascript
{
  "name": "vector_knn_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "vector",
        "numDimensions": 1024,    // Voyage AI dimension
        "similarity": "cosine"    // or "euclidean", "dotProduct"
      }
    ]
  }
}
```

### Index with Filter Fields

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
        "similarity": "cosine"
      },
      // Filter fields for pre-filtering
      {
        "type": "filter",
        "path": "metadata.category"
      },
      {
        "type": "filter",
        "path": "metadata.source"
      },
      {
        "type": "filter",
        "path": "metadata.timestamp"
      }
    ]
  }
}
```

## numCandidates Optimization

### The Golden Rule

```python
# numCandidates should be 10-20x the limit for good recall
NUM_CANDIDATES_MULTIPLIER = 20

def calculate_num_candidates(top_k: int, multiplier: int = 20) -> int:
    """Calculate optimal numCandidates for vector search."""
    return top_k * multiplier

# Examples:
# top_k=10  → numCandidates=200
# top_k=50  → numCandidates=1000
# top_k=100 → numCandidates=2000
```

### Trade-offs

| numCandidates | Recall | Latency | Memory |
|---------------|--------|---------|--------|
| 5x limit | ~85% | Low | Low |
| 10x limit | ~92% | Medium | Medium |
| 20x limit | ~97% | Higher | Higher |
| 50x limit | ~99% | High | High |

### Adaptive numCandidates

```python
def adaptive_num_candidates(
    top_k: int,
    collection_size: int,
    precision_required: str = "high"
) -> int:
    """Adapt numCandidates based on collection size and precision needs."""

    base_multiplier = {
        "low": 10,
        "medium": 15,
        "high": 20,
        "maximum": 50
    }.get(precision_required, 20)

    # Scale down for very large collections
    if collection_size > 10_000_000:
        base_multiplier = min(base_multiplier, 15)

    return min(top_k * base_multiplier, 10000)  # Cap at 10000
```

## Similarity Functions

### Cosine Similarity (Recommended)

```javascript
{
  "similarity": "cosine"
}
```

- **Best for**: Normalized embeddings (Voyage AI, OpenAI)
- **Range**: -1 to 1 (1 = identical)
- **Use when**: Direction matters more than magnitude

### Euclidean Distance

```javascript
{
  "similarity": "euclidean"
}
```

- **Best for**: Spatial data, unnormalized vectors
- **Range**: 0 to ∞ (0 = identical)
- **Use when**: Absolute distance matters

### Dot Product

```javascript
{
  "similarity": "dotProduct"
}
```

- **Best for**: Pre-normalized vectors, when magnitude matters
- **Range**: -∞ to ∞
- **Use when**: Combining direction and magnitude

## Vector Quantization

### Scalar Quantization (MongoDB 8.0+)

Reduce storage and improve performance with minimal recall loss:

```javascript
{
  "name": "vector_quantized_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [{
      "type": "vector",
      "path": "vector",
      "numDimensions": 1024,
      "similarity": "cosine",
      "quantization": "scalar"  // 4x storage reduction
    }]
  }
}
```

### Quantization Trade-offs

| Method | Storage | Recall | Latency |
|--------|---------|--------|---------|
| None (float32) | 100% | 100% | Baseline |
| Scalar (int8) | ~25% | ~98% | ~1.5x faster |
| Binary | ~3% | ~90% | ~4x faster |

### When to Use Quantization

```python
def should_use_quantization(
    collection_size: int,
    recall_requirement: float,
    latency_target_ms: float
) -> str:
    """Recommend quantization strategy."""

    if collection_size < 100_000:
        return "none"  # Full precision for small collections

    if recall_requirement > 0.99:
        return "none"  # Maximum recall needed

    if latency_target_ms < 50 and collection_size > 1_000_000:
        return "scalar"  # Balance speed and recall

    if latency_target_ms < 20 and recall_requirement < 0.92:
        return "binary"  # Maximum speed

    return "scalar"  # Default for most cases
```

## Pre-Filtering Strategies

### 1. Standard MQL Filters ($vectorSearch)

```python
from hybridrag.enhancements.filters import (
    VectorSearchFilterConfig,
    build_vector_search_filters
)

config = VectorSearchFilterConfig(
    equality_filters={
        "metadata.category": "documentation"
    },
    in_filters={
        "metadata.source": ["docs", "api-reference"]
    },
    range_filters={
        "metadata.timestamp": {
            "$gte": "2024-01-01T00:00:00Z"
        }
    }
)

filters = build_vector_search_filters(config)

pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_knn_index",
            "path": "vector",
            "queryVector": query_embedding,
            "numCandidates": 200,
            "limit": 10,
            "filter": filters
        }
    }
]
```

### 2. Lexical Prefilters ($search.vectorSearch)

For advanced filtering, see [Recipe 02: Lexical Prefilters](./02-lexical-prefilters.md).

### Pre-Filter vs Post-Filter

| Approach | When to Use | Performance |
|----------|-------------|-------------|
| Pre-filter | Highly selective filters | Fast (narrow search space) |
| Post-filter | Loose filters | Slower (search then filter) |

```python
def choose_filter_strategy(
    filter_selectivity: float,  # 0-1, lower = more selective
    collection_size: int
) -> str:
    """Choose between pre-filter and post-filter."""

    if filter_selectivity < 0.1:
        # Very selective: pre-filter saves time
        return "pre-filter"

    if filter_selectivity > 0.5:
        # Loose filter: post-filter may be faster
        return "post-filter"

    # Medium selectivity: depends on collection size
    if collection_size > 1_000_000:
        return "pre-filter"
    return "post-filter"
```

## Cosine Threshold Tuning

### Setting Minimum Similarity

```python
from hybridrag.enhancements import MongoDBHybridSearchConfig

config = MongoDBHybridSearchConfig(
    cosine_threshold=0.3  # Minimum similarity score
)

# In pipeline
pipeline = [
    {"$vectorSearch": {...}},
    {"$match": {"similarity": {"$gte": 0.3}}}  # Filter low-quality results
]
```

### Threshold Guidelines

| Threshold | Quality | Use Case |
|-----------|---------|----------|
| 0.8+ | Excellent | Deduplication, exact matches |
| 0.6-0.8 | Good | High-precision RAG |
| 0.4-0.6 | Fair | General RAG queries |
| 0.2-0.4 | Low | Exploratory, recall-focused |
| <0.2 | Poor | Usually noise |

## Embedding Model Optimization

### Voyage AI Configuration

```python
from voyageai import Client

client = Client()

# Use voyage-4-large for best quality (1024 dimensions)
embeddings = client.embed(
    texts=documents,
    model="voyage-4-large",
    input_type="document"  # Important: "document" for indexing
)

# For queries, use "query" type
query_embedding = client.embed(
    texts=[query],
    model="voyage-4-large",
    input_type="query"  # Important: "query" for searching
)
```

### Batching for Performance

```python
async def embed_in_batches(
    texts: list[str],
    batch_size: int = 128,
    model: str = "voyage-4-large"
) -> list[list[float]]:
    """Embed texts in batches for efficiency."""

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = await client.embed(
            texts=batch,
            model=model,
            input_type="document"
        )
        embeddings.extend(result.embeddings)

    return embeddings
```

## Performance Monitoring

### Query Explain

```python
# Explain vector search performance
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_knn_index",
            "path": "vector",
            "queryVector": query_embedding,
            "numCandidates": 200,
            "limit": 10
        }
    }
]

explain_result = await collection.aggregate(pipeline).explain()
print(f"Execution time: {explain_result['executionStats']['executionTimeMillis']}ms")
```

### Logging Search Metrics

```python
import time
import logging

async def search_with_metrics(
    collection,
    query_vector,
    top_k: int,
    config: MongoDBHybridSearchConfig
) -> tuple[list, dict]:
    """Search with performance metrics."""

    start = time.perf_counter()

    results = await vector_only_search(
        collection, query_vector, top_k, config
    )

    elapsed = time.perf_counter() - start

    metrics = {
        "latency_ms": elapsed * 1000,
        "results_count": len(results),
        "top_k_requested": top_k,
        "num_candidates": config.vector_num_candidates or top_k * 20,
        "threshold": config.cosine_threshold,
        "top_score": results[0].similarity if results else 0,
    }

    logging.info(f"Vector search metrics: {metrics}")

    return results, metrics
```

## Index Maintenance

### Check Index Status

```python
async def check_vector_index_status(collection, index_name: str) -> dict:
    """Check vector index status and health."""

    indexes = await collection.list_search_indexes().to_list(length=None)

    for index in indexes:
        if index.get("name") == index_name:
            return {
                "name": index_name,
                "status": index.get("status"),
                "queryable": index.get("queryable", False),
                "type": index.get("type"),
                "fields": index.get("latestDefinition", {}).get("fields", [])
            }

    return {"name": index_name, "status": "NOT_FOUND"}
```

### Index Warming

```python
async def warm_vector_index(collection, config: MongoDBHybridSearchConfig):
    """Warm up vector index with sample queries."""

    # Generate random vector for warming
    import random
    warm_vector = [random.random() for _ in range(1024)]

    # Run a few queries to warm the index
    for _ in range(3):
        await vector_only_search(
            collection, warm_vector, top_k=10, config=config
        )

    logging.info("Vector index warmed up")
```

## Best Practices Summary

| Aspect | Recommendation |
|--------|---------------|
| numCandidates | 10-20x limit |
| Similarity | Cosine for normalized embeddings |
| Quantization | Scalar for >100K docs |
| Threshold | 0.3-0.5 for general RAG |
| Batch size | 128 for embeddings |
| Filter fields | Index all filter paths |

## Troubleshooting

### Low Recall

1. Increase numCandidates
2. Lower cosine threshold
3. Check embedding model consistency
4. Verify index is queryable

### High Latency

1. Enable quantization
2. Reduce numCandidates
3. Add pre-filters to narrow search
4. Check index status

### Inconsistent Results

1. Ensure consistent input_type (document vs query)
2. Check for index rebuilding
3. Verify vector dimensions match

---

**Previous**: [Recipe 03: Conversation Memory](./03-conversation-memory.md)
**Next**: [Recipe 05: Knowledge Graph](./05-knowledge-graph.md)
