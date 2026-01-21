# ADR-0001: MongoDB Single Database Architecture

## Status

Accepted

## Context

RAG systems typically use multiple databases:
- Vector databases (Pinecone, Weaviate) for embeddings
- Graph databases (Neo4j) for knowledge graphs
- Document stores (PostgreSQL, Elasticsearch) for full-text search
- Cache layers (Redis) for performance

This creates operational complexity:
- Multiple connection strings and credentials
- Data synchronization challenges
- Increased infrastructure costs
- Complex failure scenarios
- Difficult atomic operations

## Decision

**Use MongoDB Atlas as the single database for all RAG components:**
- Vector search via `$vectorSearch` aggregation stage
- Full-text search via Atlas Search
- Knowledge graphs via native document relationships and `$graphLookup`
- Document storage via MongoDB collections
- Caching via in-memory data structures (Python-side)

## Consequences

### Positive

- **Atomic Operations**: Single transaction boundary for updates
- **Simplified Operations**: One database to manage, monitor, backup
- **Cost Reduction**: No need for multiple database subscriptions
- **Data Consistency**: No synchronization lag between systems
- **Hybrid Search**: Native `$rankFusion` for combining vector and text results

### Negative

- **MongoDB Atlas Requirement**: Requires M10+ cluster for full features
- **Vendor Lock-in**: Tight coupling to MongoDB ecosystem
- **Learning Curve**: Team must learn MongoDB-specific patterns
- **Free Tier Limitations**: M0/M2 tiers lack `$rankFusion` support

### Mitigation Strategies

- **Fallback Chain**: Manual RRF for M0/M2 tiers when `$rankFusion` unavailable
- **Clear Documentation**: Extensive docs and examples
- **Abstraction Layer**: Storage interfaces allow future swapping if needed

## References

- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [MongoDB Graph Lookups](https://www.mongodb.com/docs/manual/reference/operator/aggregation/graphLookup/)
- [MongoDB $rankFusion](https://www.mongodb.com/docs/atlas/atlas-search/rank-fusion/)

## Date

2026-01-20
