# ADR-0007: Collection Naming Strategy

## Status

Accepted

## Context

The HybridRAG database currently has 29 collections with two distinct patterns:

1. **Core collections** (shared): `ingested_documents`, `ingested_chunks`, `conversation_sessions`
2. **Engine-prefixed collections** (per-engine): `engine-bench-001_chunks`, `smoketest-codex_entities`, etc.

The engine-prefixed pattern creates ~10 collections per RAG engine instance, leading to collection proliferation (MongoDB Anti-Pattern 1.4: "Reduce Unnecessary Collections").

### Current Collection Count by Pattern

| Engine Prefix | Collections |
|--------------|-------------|
| `engine-bench-001_*` | 12 |
| `smoketest-codex_*` | 12 |
| Core (no prefix) | 5 |
| **Total** | 29 |

### Problems with Current Approach

1. **Collection Proliferation**: Each new engine creates 12 new collections
2. **Index Duplication**: Same indexes must be created for each engine's collections
3. **Management Overhead**: Difficult to query across engines
4. **No Shared Indexes**: Cannot leverage compound indexes across engines

## Decision

**Adopt a tenant-field pattern instead of collection-per-tenant:**

### Recommended Schema

```javascript
// Single chunks collection with engine_id field
{
  _id: "...",
  engine_id: "engine-bench-001",  // Tenant discriminator
  content: "...",
  vector: [...],
  full_doc_id: "...",
  created_at: 1234567890
}

// Compound index covers all engines efficiently
db.chunks.createIndex({ engine_id: 1, full_doc_id: 1 })
db.chunks.createIndex({ engine_id: 1, created_at: -1 })
```

### Migration Path

For existing deployments, keep both patterns temporarily:

1. **Phase 1**: New engines use shared collections with `engine_id` field
2. **Phase 2**: Migrate existing prefixed collections to shared pattern
3. **Phase 3**: Drop legacy prefixed collections

### Collection Consolidation Map

| Current Pattern | New Pattern |
|-----------------|-------------|
| `{engine}_chunks` | `rag_chunks` |
| `{engine}_entities` | `rag_entities` |
| `{engine}_relationships` | `rag_relationships` |
| `{engine}_full_docs` | `rag_documents` |
| `{engine}_full_entities` | `rag_entity_summaries` |
| `{engine}_full_relations` | `rag_relation_summaries` |
| `{engine}_text_chunks` | `rag_text_chunks` |
| `{engine}_entity_chunks` | `rag_entity_chunks` |
| `{engine}_relation_chunks` | `rag_relation_chunks` |
| `{engine}_chunk_entity_relation` | `rag_chunk_entity_relations` |
| `{engine}_chunk_entity_relation_edges` | `rag_chunk_entity_edges` |
| `{engine}_doc_status` | `rag_doc_status` |
| `{engine}_llm_response_cache` | `rag_llm_cache` |

**Result**: 12 collections per engine → 13 shared collections total

## Consequences

### Positive

- **Reduced Collections**: 29 → ~18 collections (with 2 engines)
- **Shared Indexes**: One set of indexes serves all engines
- **Cross-Engine Queries**: Easy to query/compare across engines
- **Simpler Management**: Fewer collections to backup, monitor, maintain
- **Better RAM Usage**: Shared indexes fit better in working set

### Negative

- **Migration Required**: Existing data needs migration
- **Larger Collections**: Single collection holds all engine data
- **Query Complexity**: Must always filter by `engine_id`

### Mitigation Strategies

- **Compound Indexes**: Always prefix indexes with `engine_id` for query efficiency
- **TTL Indexes**: Add TTL on test/staging engines for automatic cleanup
- **Partial Indexes**: Use partial indexes for engine-specific queries if needed

## Implementation

### Required Indexes for Shared Collections

```javascript
// rag_chunks
db.rag_chunks.createIndex({ engine_id: 1, full_doc_id: 1 })
db.rag_chunks.createIndex({ engine_id: 1, file_path: 1 })

// rag_entities
db.rag_entities.createIndex({ engine_id: 1, entity_name: 1 })
db.rag_entities.createIndex({ engine_id: 1, source_id: 1 })

// rag_relationships
db.rag_relationships.createIndex({ engine_id: 1, src_id: 1, tgt_id: 1 })

// rag_llm_cache (with TTL for automatic cleanup)
db.rag_llm_cache.createIndex({ engine_id: 1, query_hash: 1 })
db.rag_llm_cache.createIndex(
  { created_at: 1 },
  { expireAfterSeconds: 86400 * 7 }  // 7-day TTL
)
```

## References

- MongoDB Schema Design Anti-Pattern 1.4: "Reduce Unnecessary Collections"
- MongoDB Schema Design Rule 2.2: "Store Data That's Accessed Together"
- [MongoDB Multi-Tenancy Patterns](https://www.mongodb.com/docs/atlas/app-services/mongodb/multi-tenant/)

## Date

2026-01-29
