# Recipe 05: Knowledge Graph-Enhanced RAG

Augment vector search with entity relationships using MongoDB's graph capabilities.

## Overview

Knowledge graphs add structured relationships to RAG systems, enabling:

- Entity-centric retrieval
- Multi-hop reasoning
- Relationship-aware context
- Better handling of complex queries

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Query                        │
│        "What products are related to X?"            │
└─────────────────────────┬───────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Vector    │   │   Graph     │   │   Entity    │
│   Search    │   │  Traversal  │   │   Lookup    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │   $rankFusion   │
              │  Mix Mode RAG   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Entity-Boosted │
              │    Reranking    │
              └─────────────────┘
```

## Schema Design

### Entities Collection

```javascript
// Collection: entities
{
  "_id": ObjectId("..."),
  "name": "MongoDB Atlas",
  "normalized_name": "mongodb atlas",  // For case-insensitive lookup
  "type": "product",
  "description": "Fully managed cloud database service",
  "aliases": ["Atlas", "MongoDB Cloud"],
  "properties": {
    "vendor": "MongoDB Inc.",
    "category": "database",
    "launched": "2016"
  },
  "vector": [0.1, 0.2, ...],  // Entity embedding
  "metadata": {
    "source": "official_docs",
    "confidence": 0.95
  }
}
```

### Relationships Collection

```javascript
// Collection: relationships
{
  "_id": ObjectId("..."),
  "source_entity": "MongoDB Atlas",
  "target_entity": "Vector Search",
  "relationship_type": "HAS_FEATURE",
  "weight": 0.9,  // Relationship strength
  "properties": {
    "since_version": "6.0",
    "description": "Native vector search capability"
  },
  "bidirectional": false
}
```

### Text Chunks with Entity References

```javascript
// Collection: text_chunks
{
  "_id": ObjectId("..."),
  "content": "MongoDB Atlas provides vector search...",
  "vector": [...],
  "entities": ["MongoDB Atlas", "Vector Search"],  // Extracted entities
  "entity_mentions": [
    {"entity": "MongoDB Atlas", "start": 0, "end": 13},
    {"entity": "Vector Search", "start": 23, "end": 36}
  ],
  "document_id": ObjectId("...")
}
```

## Graph Traversal with $graphLookup

### Basic Entity Graph Traversal

```python
from hybridrag.enhancements import (
    graph_traversal,
    GraphTraversalConfig,
)

config = GraphTraversalConfig(
    max_depth=2,           # How many hops to traverse
    max_nodes=50,          # Maximum nodes to return
    relationships_collection="relationships",
    entities_collection="entities",
    source_field="source_entity",
    target_field="target_entity",
    relationship_type_field="relationship_type",
    weight_field="weight",
    min_weight=0.5         # Filter weak relationships
)

# Traverse from a starting entity
result = await graph_traversal(
    db=database,
    start_entity="MongoDB Atlas",
    config=config
)

# Returns GraphTraversalResult:
# {
#   "related_entities": ["Vector Search", "Atlas Search", ...],
#   "edges": [
#     {"source": "MongoDB Atlas", "target": "Vector Search", "type": "HAS_FEATURE"},
#     ...
#   ],
#   "max_depth_reached": 2
# }
```

### The $graphLookup Pipeline

```python
pipeline = [
    {"$match": {"name": start_entity}},
    {
        "$graphLookup": {
            "from": "relationships",
            "startWith": "$name",
            "connectFromField": "target_entity",  # For forward traversal
            "connectToField": "source_entity",
            "as": "graph_path",
            "maxDepth": 2,
            "depthField": "depth",
            "restrictSearchWithMatch": {
                "weight": {"$gte": 0.5}  # Filter by relationship strength
            }
        }
    },
    {"$unwind": "$graph_path"},
    {
        "$project": {
            "source": "$graph_path.source_entity",
            "target": "$graph_path.target_entity",
            "relationship_type": "$graph_path.relationship_type",
            "weight": "$graph_path.weight",
            "depth": "$graph_path.depth"
        }
    },
    {"$limit": 50}
]
```

### Bidirectional Traversal

```python
async def bidirectional_graph_traversal(
    db,
    start_entity: str,
    config: GraphTraversalConfig
) -> GraphTraversalResult:
    """Traverse graph in both directions from entity."""

    # Forward traversal (entity → related)
    forward_pipeline = [
        {"$match": {"source_entity": start_entity}},
        {
            "$graphLookup": {
                "from": "relationships",
                "startWith": "$target_entity",
                "connectFromField": "target_entity",
                "connectToField": "source_entity",
                "as": "forward_path",
                "maxDepth": config.max_depth
            }
        }
    ]

    # Backward traversal (related → entity)
    backward_pipeline = [
        {"$match": {"target_entity": start_entity}},
        {
            "$graphLookup": {
                "from": "relationships",
                "startWith": "$source_entity",
                "connectFromField": "source_entity",
                "connectToField": "target_entity",
                "as": "backward_path",
                "maxDepth": config.max_depth
            }
        }
    ]

    # Run both in parallel
    forward_results, backward_results = await asyncio.gather(
        db.relationships.aggregate(forward_pipeline).to_list(length=None),
        db.relationships.aggregate(backward_pipeline).to_list(length=None)
    )

    # Merge results
    all_edges = set()
    for result in forward_results + backward_results:
        for edge in result.get("forward_path", []) + result.get("backward_path", []):
            all_edges.add((edge["source_entity"], edge["target_entity"]))

    return GraphTraversalResult(
        related_entities=[e[1] for e in all_edges],
        edges=list(all_edges)
    )
```

## Entity-Boosted Reranking

### The Entity Boosting Algorithm

```python
from hybridrag.enhancements import EntityBoostingReranker

reranker = EntityBoostingReranker(
    base_reranker=voyage_rerank_func,
    boost_weight=0.2  # How much to boost entity matches
)

# Query entities are extracted from the user query
query_entities = ["MongoDB", "vector search"]

# Boost chunks that mention query entities
boosted_results = await reranker.rerank(
    query=user_query,
    results=search_results,
    query_entities=query_entities
)
```

### Entity Overlap Calculation

```python
def calculate_entity_boost(
    chunk_entities: list[str],
    query_entities: list[str],
    boost_weight: float = 0.2
) -> float:
    """Calculate boost based on entity overlap."""

    if not query_entities:
        return 0.0

    # Normalize entities for matching
    chunk_normalized = {e.lower() for e in chunk_entities}
    query_normalized = {e.lower() for e in query_entities}

    # Calculate overlap
    overlap = chunk_normalized & query_normalized
    overlap_ratio = len(overlap) / len(query_normalized)

    return overlap_ratio * boost_weight


# Example:
# Query entities: ["MongoDB Atlas", "vector search"]
# Chunk entities: ["MongoDB Atlas", "indexing"]
# Overlap: 1/2 = 0.5
# Boost: 0.5 * 0.2 = 0.1 added to score
```

## Mix Mode Search

### Combining All Signals

```python
from hybridrag.enhancements import (
    mix_mode_search,
    MixModeConfig,
)

config = MixModeConfig(
    # Hybrid search settings
    hybrid_config=MongoDBHybridSearchConfig(
        vector_weight=0.6,
        text_weight=0.4
    ),
    # Graph traversal settings
    graph_config=GraphTraversalConfig(
        max_depth=2,
        max_nodes=50
    ),
    # Feature toggles
    enable_graph_traversal=True,
    enable_entity_boosting=True,
    enable_reranking=True,
    # Weights
    entity_boost_weight=0.2,
    entity_only_weight=0.5  # For entity-only results
)

results = await mix_mode_search(
    db=database,
    chunks_collection=chunks,
    query_text=user_query,
    query_vector=query_embedding,
    query_entities=["MongoDB Atlas"],  # Extracted from query
    top_k=10,
    config=config
)
```

### Mix Mode Pipeline

```
1. Extract entities from query
2. Run hybrid search ($rankFusion)
3. Traverse knowledge graph from query entities
4. Find chunks mentioning related entities
5. Combine with entity boosting
6. Final reranking with cross-encoder
```

## Entity Extraction

### Basic NER with SpaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> list[str]:
    """Extract named entities from text."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECHNOLOGY"]:
            entities.append(ent.text)
    return entities
```

### LLM-Based Entity Extraction

```python
ENTITY_EXTRACTION_PROMPT = """Extract key entities from the following text.
Focus on products, technologies, organizations, and concepts.
Return as a JSON array of strings.

Text: {text}

Entities:"""

async def extract_entities_llm(text: str, llm_func) -> list[str]:
    """Extract entities using LLM."""
    response = await llm_func(
        ENTITY_EXTRACTION_PROMPT.format(text=text)
    )
    return json.loads(response)
```

## Building the Knowledge Graph

### Entity-Relationship Extraction Pipeline

```python
async def build_knowledge_graph(
    chunks: list[dict],
    llm_func,
    db
) -> None:
    """Build knowledge graph from document chunks."""

    all_entities = {}
    all_relationships = []

    for chunk in chunks:
        # Extract entities
        entities = await extract_entities_llm(chunk["content"], llm_func)

        for entity in entities:
            normalized = entity.lower()
            if normalized not in all_entities:
                all_entities[normalized] = {
                    "name": entity,
                    "normalized_name": normalized,
                    "type": "concept",
                    "mentions": 0
                }
            all_entities[normalized]["mentions"] += 1

        # Extract relationships between entities
        if len(entities) >= 2:
            relationships = await extract_relationships_llm(
                chunk["content"], entities, llm_func
            )
            all_relationships.extend(relationships)

    # Store in MongoDB
    await db.entities.insert_many(list(all_entities.values()))
    await db.relationships.insert_many(all_relationships)
```

### Relationship Extraction Prompt

```python
RELATIONSHIP_EXTRACTION_PROMPT = """Given these entities: {entities}
And this text: {text}

Extract relationships between the entities.
Return as JSON array: [{"source": "X", "target": "Y", "type": "RELATIONSHIP_TYPE"}]

Common relationship types: HAS_FEATURE, PART_OF, RELATED_TO, COMPETES_WITH, USED_BY

Relationships:"""
```

## Index Configuration

### Entity Index

```javascript
// Compound index for entity lookup
db.entities.createIndex(
  { "normalized_name": 1 },
  { unique: true }
)

// Text index for entity search
db.entities.createIndex(
  { "name": "text", "aliases": "text" }
)
```

### Relationship Index

```javascript
// Indexes for graph traversal
db.relationships.createIndex({ "source_entity": 1 })
db.relationships.createIndex({ "target_entity": 1 })
db.relationships.createIndex(
  { "source_entity": 1, "target_entity": 1 },
  { unique: true }
)
```

### Chunk Entity Index

```javascript
// Multikey index for entity-based chunk retrieval
db.text_chunks.createIndex({ "entities": 1 })
```

## Query Patterns

### Find Chunks for Entities

```python
async def get_chunks_for_entities(
    db,
    entities: list[str],
    limit: int = 20
) -> list[dict]:
    """Find chunks mentioning specific entities."""

    pipeline = [
        {
            "$match": {
                "entities": {"$in": entities}
            }
        },
        {
            "$addFields": {
                "entity_count": {
                    "$size": {
                        "$setIntersection": ["$entities", entities]
                    }
                }
            }
        },
        {"$sort": {"entity_count": -1}},  # Most entity overlap first
        {"$limit": limit}
    ]

    return await db.text_chunks.aggregate(pipeline).to_list(length=None)
```

### Entity-Aware Hybrid Search

```python
async def entity_aware_search(
    db,
    query: str,
    query_vector: list[float],
    top_k: int = 10
) -> list[dict]:
    """Search that leverages both vectors and entity relationships."""

    # 1. Extract entities from query
    query_entities = await extract_entities_llm(query, llm_func)

    # 2. Expand entities via graph
    expanded_entities = set(query_entities)
    for entity in query_entities:
        related = await graph_traversal(db, entity, config)
        expanded_entities.update(related.related_entities[:5])  # Top 5 related

    # 3. Hybrid search with entity filter
    results = await hybrid_search_with_rank_fusion(
        collection=db.text_chunks,
        query_text=query,
        query_vector=query_vector,
        top_k=top_k * 2,
        config=config
    )

    # 4. Boost results with entity overlap
    for result in results:
        chunk_entities = result.get("entities", [])
        boost = calculate_entity_boost(chunk_entities, list(expanded_entities))
        result["score"] = result.get("score", 0) + boost

    # 5. Re-sort by boosted score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]
```

## Best Practices

| Aspect | Recommendation |
|--------|---------------|
| Graph depth | 2-3 hops maximum |
| Entity normalization | Case-insensitive, strip whitespace |
| Relationship weights | 0-1 scale, filter < 0.3 |
| Entity boost | 0.1-0.3 weight |
| Extraction model | Domain-specific fine-tuning helps |

## References

- [MongoDB $graphLookup Documentation](https://www.mongodb.com/docs/manual/reference/operator/aggregation/graphLookup/)
- [Knowledge Graphs for RAG](https://www.mongodb.com/developer/products/atlas/knowledge-graphs-rag/)

---

**Previous**: [Recipe 04: Vector Search Optimization](./04-vector-search-optimization.md)
**Next**: [Recipe 06: Filtering Strategies](./06-filtering-strategies.md)
