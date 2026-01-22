# Recipe 01: Native Hybrid Search with $rankFusion

Master MongoDB's native hybrid search combining vector similarity and full-text keyword search.

## Overview

Hybrid search addresses a fundamental limitation: neither vector search nor keyword search alone provides optimal results for all queries. By combining both approaches with Reciprocal Rank Fusion (RRF), you get the best of both worlds.

## The Problem

| Query Type | Vector Search | Text Search | Hybrid |
|------------|--------------|-------------|--------|
| "smart thermostat" | Finds semantic matches | Exact match | Both |
| "it's too warm at home" | Understands intent | No matches | Vector helps |
| "themrostat" (typo) | No match | Fuzzy finds it | Text helps |
| Technical acronyms | May miss context | Exact match | Text helps |

## MongoDB's Solution: $rankFusion

MongoDB 8.0+ provides native `$rankFusion` that:
1. Runs vector and text search pipelines in parallel
2. Applies Reciprocal Rank Fusion algorithm
3. Returns unified, relevance-ranked results

### The RRF Formula

```
score(d) = Σ (1 / (k + rank_i(d)))
```

Where:
- `k` = 60 (default constant, configurable)
- `rank_i(d)` = position of document d in result list i

## Implementation

### Basic Hybrid Search

```python
from hybridrag.enhancements import (
    hybrid_search_with_rank_fusion,
    MongoDBHybridSearchConfig,
)

# Configure hybrid search
config = MongoDBHybridSearchConfig(
    vector_index_name="vector_knn_index",
    text_index_name="text_search_index",
    vector_path="vector",
    text_search_path="content",
    vector_weight=0.6,  # Semantic similarity weight
    text_weight=0.4,    # Keyword matching weight
)

# Run hybrid search
results = await hybrid_search_with_rank_fusion(
    collection=chunks_collection,
    query_text="How do I configure authentication?",
    query_vector=query_embedding,
    top_k=10,
    config=config,
)
```

### Full Pipeline Example

```python
# The complete $rankFusion pipeline
pipeline = [
    {
        "$rankFusion": {
            "input": {
                "pipelines": {
                    # Pipeline 1: Vector Search (semantic similarity)
                    "vector": [
                        {
                            "$vectorSearch": {
                                "index": "vector_knn_index",
                                "path": "vector",
                                "queryVector": query_embedding,
                                "numCandidates": 200,  # 20x limit for good recall
                                "limit": 20,
                            }
                        }
                    ],
                    # Pipeline 2: Full-Text Search (keyword matching)
                    "text": [
                        {
                            "$search": {
                                "index": "text_search_index",
                                "compound": {
                                    "must": [{
                                        "text": {
                                            "query": query_text,
                                            "path": "content",
                                            "fuzzy": {
                                                "maxEdits": 2,
                                                "prefixLength": 3
                                            }
                                        }
                                    }]
                                }
                            }
                        },
                        {"$limit": 20}
                    ]
                }
            },
            # Configurable weights for each pipeline
            "combination": {
                "weights": {
                    "vector": 0.6,
                    "text": 0.4
                }
            },
            # CRITICAL: Enable for debugging
            "scoreDetails": True
        }
    },
    # Extract scores for analysis
    {
        "$addFields": {
            "hybrid_score": {"$meta": "rankFusionScore"},
            "score_details": {"$meta": "scoreDetails"}
        }
    },
    {"$limit": 10},
    {"$project": {"vector": 0}}  # Exclude large vector field
]

results = await collection.aggregate(pipeline).to_list(length=None)
```

### Extracting Per-Pipeline Scores

```python
def extract_pipeline_score(score_details: dict, pipeline_name: str) -> float:
    """Extract individual pipeline contribution from scoreDetails."""
    if not score_details or "details" not in score_details:
        return 0.0

    for detail in score_details.get("details", []):
        if detail.get("inputPipelineName") == pipeline_name:
            return detail.get("value", 0.0)

    return 0.0

# Usage
for result in results:
    vector_score = extract_pipeline_score(result["score_details"], "vector")
    text_score = extract_pipeline_score(result["score_details"], "text")
    print(f"Total: {result['hybrid_score']:.4f} (vector: {vector_score:.4f}, text: {text_score:.4f})")
```

## Alternative: $scoreFusion

For weighted score combination instead of rank fusion:

```python
pipeline = [
    {
        "$scoreFusion": {
            "input": {
                "pipelines": {...},
                "normalization": "sigmoid"  # Required inside input
            },
            "combination": {
                "weights": {
                    "vector": 0.6,
                    "text": 0.4
                }
            },
            "scoreDetails": True
        }
    }
]
```

### When to Use Each

| Use Case | Algorithm | Reasoning |
|----------|-----------|-----------|
| General RAG | $rankFusion | Rank-based, position-aware |
| Similar score ranges | $scoreFusion | Direct score combination |
| M0/M2 tiers | Manual RRF | Native fusion not available |

## Manual RRF Fallback

For MongoDB Atlas free tier (M0) where $rankFusion is not available:

```python
from hybridrag.enhancements import (
    manual_hybrid_search_with_rrf,
    reciprocal_rank_fusion,
)

# Run both searches concurrently
vector_results, text_results = await asyncio.gather(
    vector_only_search(collection, query_vector, top_k * 2, config),
    text_only_search(collection, query_text, top_k * 2, config),
)

# Merge with RRF
merged_results = reciprocal_rank_fusion(
    [vector_results, text_results],
    k=60  # RRF constant
)

final_results = merged_results[:top_k]
```

## Multi-Field Weighted Text Search

Search across multiple fields with different importance:

```python
config = MongoDBHybridSearchConfig(
    text_search_path_weights={
        "content": 10,      # Main content - highest weight
        "title": 5,         # Title - medium weight
        "metadata.tags": 3, # Tags - lower weight
    }
)

# Generates compound query with score boosting
{
    "compound": {
        "should": [
            {
                "text": {
                    "query": query_text,
                    "path": "content",
                    "fuzzy": {"maxEdits": 2, "prefixLength": 3},
                    "score": {"boost": {"value": 10}}
                }
            },
            {
                "text": {
                    "query": query_text,
                    "path": "title",
                    "fuzzy": {"maxEdits": 2, "prefixLength": 3},
                    "score": {"boost": {"value": 5}}
                }
            }
        ],
        "minimumShouldMatch": 1
    }
}
```

## Index Requirements

### Vector Search Index

```javascript
{
  "name": "vector_knn_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [{
      "type": "vector",
      "path": "vector",
      "numDimensions": 1024,
      "similarity": "cosine"
    }]
  }
}
```

### Text Search Index

```javascript
{
  "name": "text_search_index",
  "type": "search",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "content": {
          "type": "string",
          "analyzer": "lucene.standard"
        }
      }
    }
  }
}
```

## Best Practices

### 1. numCandidates Sizing

```python
# Always use 10-20x the limit for good recall
NUM_CANDIDATES_MULTIPLIER = 20

def calculate_num_candidates(top_k: int) -> int:
    return top_k * NUM_CANDIDATES_MULTIPLIER

# For top_k=10, use numCandidates=200
```

### 2. Weight Tuning

| Query Type | Vector Weight | Text Weight |
|------------|--------------|-------------|
| Natural language questions | 0.7 | 0.3 |
| Technical documentation | 0.5 | 0.5 |
| Code search | 0.3 | 0.7 |
| Proper nouns/names | 0.4 | 0.6 |

### 3. Error Handling

```python
async def robust_hybrid_search(collection, query_text, query_vector, top_k):
    """Hybrid search with graceful fallbacks."""
    try:
        # Try native $rankFusion first
        return await hybrid_search_with_rank_fusion(
            collection, query_text, query_vector, top_k
        )
    except Exception as e:
        logger.warning(f"$rankFusion failed: {e}")

        try:
            # Fallback to manual RRF
            return await manual_hybrid_search_with_rrf(
                collection, query_text, query_vector, top_k
            )
        except Exception as e2:
            logger.warning(f"Manual RRF failed: {e2}")

            # Last resort: vector-only search
            return await vector_only_search(
                collection, query_vector, top_k
            )
```

## Performance Considerations

| Factor | Recommendation |
|--------|---------------|
| numCandidates | 10-20x limit |
| Pipeline limit | 2x final limit (overfetch) |
| Index warming | Pre-query on startup |
| Result caching | Cache common queries |

## HybridRAG Integration

```python
from hybridrag import HybridRAG

rag = HybridRAG()
await rag.initialize()

# Uses hybrid search by default with mode="mix"
response = await rag.query(
    "What authentication methods are supported?",
    mode="mix"  # Combines vector + text + knowledge graph
)
```

## References

- [MongoDB $rankFusion Documentation](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/)
- [Hybrid Search Tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/hybrid-search/)
- [MongoDB Blog: Harness the Power of $rankFusion](https://www.mongodb.com/company/blog/technical/harness-power-atlas-search-vector-search-with-rankfusion)

---

**Next**: [Recipe 02: Lexical Prefilters for Vector Search](./02-lexical-prefilters.md)
