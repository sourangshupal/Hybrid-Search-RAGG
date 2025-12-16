"""
MongoDB Native Hybrid Search using $rankFusion.

This module implements proper rank fusion for combining vector similarity
search with full-text keyword search using MongoDB Atlas's native $rankFusion
aggregation stage.

Why it matters:
    Simple interleaving of results from different sources using round-robin
    provides no actual fusion. This is inadequate for production RAG systems
    that need proper relevance scoring.

    MongoDB's $rankFusion uses Reciprocal Rank Fusion (RRF):
    score = Σ (1 / (60 + rank_i)) for each input pipeline

    This provides mathematically sound fusion of multiple retrieval signals.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection
    from pymongo.asynchronous.database import AsyncDatabase

logger = logging.getLogger("hybridrag.mongodb_hybrid")
logger.setLevel(logging.INFO)

# Default RRF constant (MongoDB default is 60, but configurable)
DEFAULT_RRF_CONSTANT = 60


class SearchResult(BaseModel):
    """
    Type-safe model for search results.

    Provides consistent structure for all search types (semantic, text, hybrid).
    Uses Pydantic for validation and serialization.
    """

    chunk_id: str = Field(..., description="MongoDB ObjectId of chunk as string")
    document_id: str = Field(default="", description="Parent document ObjectId as string")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1 for vector, RRF score for hybrid)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata (source, page, etc.)")
    document_title: str = Field(default="", description="Title from document lookup")
    document_source: str = Field(default="", description="Source path from document lookup")
    search_type: str = Field(default="unknown", description="Type of search that produced this result")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()


@dataclass
class MongoDBHybridSearchConfig:
    """Configuration for MongoDB hybrid search."""

    # Vector search settings
    vector_index_name: str = "vector_knn_index"
    vector_path: str = "vector"
    vector_num_candidates: int = 100

    # Full-text search settings
    text_index_name: str = "text_search_index"
    text_search_path: str = "content"  # Field(s) to search

    # Fuzzy search settings (for text search)
    fuzzy_max_edits: int = 2  # Max character edits for fuzzy matching
    fuzzy_prefix_length: int = 3  # Chars that must match exactly at start

    # Hybrid search settings
    vector_weight: float = 0.6  # Weight for vector search in score fusion
    text_weight: float = 0.4  # Weight for text search in score fusion
    use_rank_fusion: bool = True  # Use $rankFusion (RRF) instead of $scoreFusion

    # Document lookup settings
    documents_collection: str = "documents"  # Collection for document metadata
    enable_document_lookup: bool = True  # Join with documents for metadata

    # Filtering
    cosine_threshold: float = 0.3


async def create_text_search_index_if_not_exists(
    collection: "AsyncCollection",
    index_name: str = "text_search_index",
    search_fields: list[str] | None = None,
) -> bool:
    """
    Create a MongoDB Atlas Search index for full-text search.

    Args:
        collection: MongoDB collection
        index_name: Name for the search index
        search_fields: Fields to index for text search (default: ["content"])

    Returns:
        bool: True if index was created, False if it already exists
    """
    if search_fields is None:
        search_fields = ["content"]

    try:
        # Check if index already exists
        indexes_cursor = await collection.list_search_indexes()
        indexes = await indexes_cursor.to_list(length=None)

        for index in indexes:
            if index.get("name") == index_name:
                logger.info(f"Text search index '{index_name}' already exists")
                return False

        # Create the search index definition
        # Using "lucene.standard" analyzer for general text search
        field_mappings = {}
        for field_name in search_fields:
            field_mappings[field_name] = {
                "type": "string",
                "analyzer": "lucene.standard",
            }

        from pymongo.operations import SearchIndexModel

        search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": field_mappings,
                }
            },
            name=index_name,
            type="search",
        )

        await collection.create_search_index(search_index_model)
        logger.info(f"Text search index '{index_name}' created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating text search index '{index_name}': {e}")
        raise


async def hybrid_search_with_rank_fusion(
    collection: "AsyncCollection",
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search using MongoDB's $rankFusion.

    This combines:
    1. Vector similarity search ($vectorSearch)
    2. Full-text keyword search ($search)

    Using Reciprocal Rank Fusion (RRF) formula:
    score = Σ (1 / (60 + rank_i))

    Args:
        collection: MongoDB collection with both vector and text indexes
        query_text: The search query text
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Hybrid search configuration

    Returns:
        List of documents with fused relevance scores
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    logger.info(
        f"[HYBRID_SEARCH] Starting $rankFusion search: "
        f"query='{query_text[:50]}...', top_k={top_k}"
    )

    # Build the hybrid search pipeline using $rankFusion
    # $rankFusion accepts an array of input pipelines
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        # Pipeline 1: Vector similarity search
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": config.vector_index_name,
                                    "path": config.vector_path,
                                    "queryVector": query_vector,
                                    "numCandidates": config.vector_num_candidates,
                                    "limit": top_k * 2,  # Get more candidates for fusion
                                }
                            }
                        ],
                        # Pipeline 2: Full-text search
                        "text": [
                            {
                                "$search": {
                                    "index": config.text_index_name,
                                    "text": {
                                        "query": query_text,
                                        "path": config.text_search_path,
                                    },
                                }
                            },
                            {"$limit": top_k * 2},
                        ],
                    }
                },
                "combination": {"rrf": {}},  # Reciprocal Rank Fusion - must be object
            }
        },
        # Add fusion score to results
        {"$addFields": {"hybrid_score": {"$meta": "rankFusionScore"}}},
        # Limit final results
        {"$limit": top_k},
        # Project out the vector field to reduce response size
        {"$project": {"vector": 0}},
    ]

    try:
        cursor = await collection.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)

        logger.info(f"[HYBRID_SEARCH] $rankFusion returned {len(results)} results")

        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append(
                {
                    **doc,
                    "id": doc.get("_id"),
                    "score": doc.get("hybrid_score"),
                    "search_type": "hybrid_rrf",
                }
            )

        if formatted_results:
            top_score = formatted_results[0].get("score", 0)
            logger.info(f"[HYBRID_SEARCH] Top result score: {top_score:.4f}")

        return formatted_results

    except Exception as e:
        logger.error(f"[HYBRID_SEARCH] $rankFusion failed: {e}")
        # Fall back to manual RRF (works on M0/M2 tiers)
        logger.warning("[HYBRID_SEARCH] Falling back to manual RRF search")
        try:
            return await manual_hybrid_search_with_rrf(
                collection, query_text, query_vector, top_k, config
            )
        except Exception as rrf_err:
            # Last resort: vector-only search
            logger.error(f"[HYBRID_SEARCH] Manual RRF also failed: {rrf_err}")
            logger.warning("[HYBRID_SEARCH] Last resort: vector-only search")
            return await vector_only_search(
                collection, query_vector, top_k, config
            )


async def hybrid_search_with_score_fusion(
    collection: "AsyncCollection",
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search using MongoDB's $scoreFusion with custom weights.

    This allows explicit weighting of vector vs text search scores:
    final_score = (vector_weight * vector_score) + (text_weight * text_score)

    Args:
        collection: MongoDB collection
        query_text: The search query text
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Hybrid search configuration

    Returns:
        List of documents with weighted fusion scores
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    logger.info(
        f"[HYBRID_SEARCH] Starting $scoreFusion search: "
        f"weights=[vector:{config.vector_weight}, text:{config.text_weight}]"
    )

    # Build the score fusion pipeline
    pipeline = [
        {
            "$scoreFusion": {
                "input": {
                    "pipelines": {
                        # Pipeline 1: Vector search with sigmoid normalization
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": config.vector_index_name,
                                    "path": config.vector_path,
                                    "queryVector": query_vector,
                                    "numCandidates": config.vector_num_candidates,
                                    "limit": top_k * 2,
                                }
                            }
                        ],
                        # Pipeline 2: Full-text search with sigmoid normalization
                        "text": [
                            {
                                "$search": {
                                    "index": config.text_index_name,
                                    "text": {
                                        "query": query_text,
                                        "path": config.text_search_path,
                                    },
                                }
                            },
                            {"$limit": top_k * 2},
                        ],
                    }
                },
                "combination": {
                    # Weighted sum with custom expression
                    "weights": {
                        "vector": config.vector_weight,
                        "text": config.text_weight,
                    }
                },
                "normalization": {
                    # Sigmoid normalization for score scaling
                    "sigmoid": {}
                },
            }
        },
        {"$addFields": {"fusion_score": {"$meta": "scoreFusionScore"}}},
        {"$limit": top_k},
        {"$project": {"vector": 0}},
    ]

    try:
        cursor = await collection.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)

        logger.info(f"[HYBRID_SEARCH] $scoreFusion returned {len(results)} results")

        formatted_results = []
        for doc in results:
            formatted_results.append(
                {
                    **doc,
                    "id": doc.get("_id"),
                    "score": doc.get("fusion_score"),
                    "search_type": "hybrid_score_fusion",
                }
            )

        return formatted_results

    except Exception as e:
        logger.error(f"[HYBRID_SEARCH] $scoreFusion failed: {e}")
        # Fall back to manual RRF (works on M0/M2 tiers)
        logger.warning("[HYBRID_SEARCH] Falling back to manual RRF search")
        try:
            return await manual_hybrid_search_with_rrf(
                collection, query_text, query_vector, top_k, config
            )
        except Exception as rrf_err:
            # Last resort: vector-only search
            logger.error(f"[HYBRID_SEARCH] Manual RRF also failed: {rrf_err}")
            logger.warning("[HYBRID_SEARCH] Last resort: vector-only search")
            return await vector_only_search(
                collection, query_vector, top_k, config
            )


async def text_only_search(
    collection: "AsyncCollection",
    query_text: str,
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: "AsyncDatabase | None" = None,
) -> list[SearchResult]:
    """
    Full-text search using MongoDB Atlas Search with fuzzy matching.

    Uses $search operator for keyword matching with fuzzy matching support.
    Optionally joins with documents collection for metadata.
    Works on all Atlas tiers including M0 (free tier).

    Args:
        collection: MongoDB collection with text search index
        query_text: The search query text
        top_k: Number of results to return
        config: Search configuration
        db: Database instance for $lookup (optional)

    Returns:
        List of SearchResult objects ordered by text relevance
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    # Build pipeline with fuzzy matching
    pipeline: list[dict[str, Any]] = [
        {
            "$search": {
                "index": config.text_index_name,
                "text": {
                    "query": query_text,
                    "path": config.text_search_path,
                    "fuzzy": {
                        "maxEdits": config.fuzzy_max_edits,
                        "prefixLength": config.fuzzy_prefix_length,
                    },
                },
            }
        },
        {"$limit": top_k * 2},  # Over-fetch for better RRF results
    ]

    # Add $lookup for document metadata if enabled and db provided
    if config.enable_document_lookup and db is not None:
        pipeline.extend([
            {
                "$lookup": {
                    "from": config.documents_collection,
                    "localField": "document_id",
                    "foreignField": "_id",
                    "as": "document_info",
                }
            },
            {
                "$unwind": {
                    "path": "$document_info",
                    "preserveNullAndEmptyArrays": True,
                }
            },
        ])

    # Project final fields
    pipeline.append({
        "$project": {
            "chunk_id": "$_id",
            "document_id": 1,
            "content": 1,
            "similarity": {"$meta": "searchScore"},
            "metadata": 1,
            "document_title": {"$ifNull": ["$document_info.title", ""]},
            "document_source": {"$ifNull": ["$document_info.source", ""]},
            "vector": 0,
        }
    })

    try:
        cursor = await collection.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                chunk_id=str(doc.get("chunk_id", doc.get("_id", ""))),
                document_id=str(doc.get("document_id", "")),
                content=doc.get("content", ""),
                similarity=doc.get("similarity", 0.0),
                metadata=doc.get("metadata", {}),
                document_title=doc.get("document_title", ""),
                document_source=doc.get("document_source", ""),
                search_type="text_only",
            )
            for doc in results
        ]

        logger.info(
            f"[TEXT_SEARCH] Completed: query='{query_text[:50]}...', results={len(search_results)}"
        )

        return search_results

    except Exception as e:
        logger.warning(f"[TEXT_SEARCH] Text search failed: {e}")
        return []


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = DEFAULT_RRF_CONSTANT,
) -> list[SearchResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF is a simple yet effective algorithm for combining results from different
    search methods. It works by scoring each document based on its rank position
    in each result list.

    Args:
        result_lists: List of ranked SearchResult lists from different searches
        k: RRF constant (default: 60, standard in literature)

    Returns:
        Unified list of SearchResult sorted by combined RRF score

    Algorithm:
        For each document d appearing in result lists:
            RRF_score(d) = Σ(1 / (k + rank_i(d)))
        Where rank_i(d) is the position of document d in result list i.

    References:
        - Cormack et al. (2009): "Reciprocal Rank Fusion outperforms the best system"
        - Standard k=60 performs well across various datasets
    """
    # Build score dictionary by chunk_id
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    # Process each search result list
    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id

            # Calculate RRF contribution: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            # Accumulate score (automatic deduplication)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = result

    # Sort by combined RRF score (descending)
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build final result list with updated similarity scores
    merged_results = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_map[chunk_id]
        # Create new SearchResult with updated similarity (RRF score)
        merged_result = SearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            content=result.content,
            similarity=rrf_score,  # Combined RRF score
            metadata=result.metadata,
            document_title=result.document_title,
            document_source=result.document_source,
            search_type="hybrid_rrf",
        )
        merged_results.append(merged_result)

    logger.info(
        f"[RRF] Merged {len(result_lists)} result lists into "
        f"{len(merged_results)} unique results"
    )

    return merged_results


async def manual_hybrid_search_with_rrf(
    collection: "AsyncCollection",
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: "AsyncDatabase | None" = None,
) -> list[SearchResult]:
    """
    Manual RRF implementation for M0/M2 tiers.

    This function is used when MongoDB's native $rankFusion is not available
    (e.g., on free tier M0 or shared M2 clusters).

    It runs semantic (vector) search and text search concurrently, then merges
    results using Reciprocal Rank Fusion (RRF).

    Works on all Atlas tiers including M0 (free tier) - no M10+ required!

    Args:
        collection: MongoDB collection with both vector and text indexes
        query_text: The search query text
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Hybrid search configuration
        db: Database instance for $lookup (optional)

    Returns:
        List of SearchResult with fused RRF scores

    Algorithm:
        1. Run semantic search (vector similarity)
        2. Run text search (keyword/fuzzy matching)
        3. Merge results using Reciprocal Rank Fusion
        4. Return top N results by combined score
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    logger.info(
        f"[MANUAL_HYBRID] Starting manual RRF search: "
        f"query='{query_text[:50]}...', top_k={top_k}"
    )

    # Over-fetch for better RRF results (2x requested count)
    fetch_count = top_k * 2

    # Run both searches concurrently for performance
    vector_results, text_results = await asyncio.gather(
        vector_only_search(collection, query_vector, fetch_count, config, db),
        text_only_search(collection, query_text, fetch_count, config, db),
        return_exceptions=True,
    )

    # Handle errors gracefully
    if isinstance(vector_results, Exception):
        logger.warning(f"[MANUAL_HYBRID] Vector search failed: {vector_results}")
        vector_results = []

    if isinstance(text_results, Exception):
        logger.warning(f"[MANUAL_HYBRID] Text search failed: {text_results}")
        text_results = []

    # If both failed, return empty list
    if not vector_results and not text_results:
        logger.error("[MANUAL_HYBRID] Both searches failed, returning empty results")
        return []

    # If only one succeeded, return those results directly
    if not vector_results:
        logger.info("[MANUAL_HYBRID] Only text search succeeded, returning text results")
        return text_results[:top_k]

    if not text_results:
        logger.info("[MANUAL_HYBRID] Only vector search succeeded, returning vector results")
        return vector_results[:top_k]

    # Merge results using Reciprocal Rank Fusion
    merged = reciprocal_rank_fusion(
        [vector_results, text_results],
        k=DEFAULT_RRF_CONSTANT,
    )

    # Return top N results
    final_results = merged[:top_k]

    logger.info(
        f"[MANUAL_HYBRID] Completed: "
        f"semantic={len(vector_results)}, text={len(text_results)}, "
        f"merged={len(merged)}, returned={len(final_results)}"
    )

    return final_results


async def vector_only_search(
    collection: "AsyncCollection",
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: "AsyncDatabase | None" = None,
) -> list[SearchResult]:
    """
    Perform semantic vector search using MongoDB Atlas Vector Search.

    Args:
        collection: MongoDB collection with vector search index
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Search configuration
        db: Database instance for $lookup (optional)

    Returns:
        List of SearchResult objects ordered by vector similarity
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    # Build pipeline
    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": config.vector_index_name,
                "path": config.vector_path,
                "queryVector": query_vector,
                "numCandidates": config.vector_num_candidates,
                "limit": top_k,
            }
        },
    ]

    # Add $lookup for document metadata if enabled and db provided
    if config.enable_document_lookup and db is not None:
        pipeline.extend([
            {
                "$lookup": {
                    "from": config.documents_collection,
                    "localField": "document_id",
                    "foreignField": "_id",
                    "as": "document_info",
                }
            },
            {
                "$unwind": {
                    "path": "$document_info",
                    "preserveNullAndEmptyArrays": True,
                }
            },
        ])

    # Project final fields
    pipeline.append({
        "$project": {
            "chunk_id": "$_id",
            "document_id": 1,
            "content": 1,
            "similarity": {"$meta": "vectorSearchScore"},
            "metadata": 1,
            "document_title": {"$ifNull": ["$document_info.title", ""]},
            "document_source": {"$ifNull": ["$document_info.source", ""]},
            "vector": 0,
        }
    })

    # Filter by cosine threshold
    pipeline.append({"$match": {"similarity": {"$gte": config.cosine_threshold}}})

    try:
        cursor = await collection.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                chunk_id=str(doc.get("chunk_id", doc.get("_id", ""))),
                document_id=str(doc.get("document_id", "")),
                content=doc.get("content", ""),
                similarity=doc.get("similarity", 0.0),
                metadata=doc.get("metadata", {}),
                document_title=doc.get("document_title", ""),
                document_source=doc.get("document_source", ""),
                search_type="vector_only",
            )
            for doc in results
        ]

        logger.info(
            f"[VECTOR_SEARCH] Completed: results={len(search_results)}, threshold={config.cosine_threshold}"
        )

        return search_results

    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] Failed: {e}")
        return []


class MongoDBHybridSearcher:
    """
    High-level interface for MongoDB hybrid search operations.

    This class manages the setup and execution of hybrid searches
    across multiple collections (chunks, entities, relationships).
    """

    def __init__(
        self,
        db: "AsyncDatabase",
        workspace: str = "",
        config: MongoDBHybridSearchConfig | None = None,
    ):
        self.db = db
        self.workspace = workspace
        self.config = config or MongoDBHybridSearchConfig()
        self._initialized_collections: set[str] = set()

    def _get_collection_name(self, namespace: str) -> str:
        """Get the full collection name with workspace prefix."""
        if self.workspace:
            return f"{self.workspace}_{namespace}"
        return namespace

    async def ensure_text_index(self, namespace: str, search_fields: list[str] | None = None) -> None:
        """
        Ensure text search index exists for a collection.

        Args:
            namespace: Collection namespace (e.g., "text_chunks")
            search_fields: Fields to index (default: ["content"])
        """
        collection_name = self._get_collection_name(namespace)

        if collection_name in self._initialized_collections:
            return

        collection = self.db[collection_name]

        # Determine index name based on workspace
        if self.workspace:
            index_name = f"text_search_index_{collection_name}"
        else:
            index_name = f"text_search_index_{namespace}"

        await create_text_search_index_if_not_exists(
            collection,
            index_name=index_name,
            search_fields=search_fields or ["content"],
        )

        self._initialized_collections.add(collection_name)

    async def hybrid_search(
        self,
        namespace: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        use_rank_fusion: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search on a collection.

        Args:
            namespace: Collection namespace
            query_text: Search query text
            query_vector: Query embedding vector
            top_k: Number of results
            use_rank_fusion: Use RRF (True) or weighted score fusion (False)

        Returns:
            List of search results with fusion scores
        """
        collection_name = self._get_collection_name(namespace)
        collection = self.db[collection_name]

        # Update config with workspace-specific index names
        config = MongoDBHybridSearchConfig(
            vector_index_name=f"vector_knn_index_{collection_name}" if self.workspace else "vector_knn_index",
            text_index_name=f"text_search_index_{collection_name}" if self.workspace else f"text_search_index_{namespace}",
            vector_weight=self.config.vector_weight,
            text_weight=self.config.text_weight,
            cosine_threshold=self.config.cosine_threshold,
        )

        if use_rank_fusion:
            return await hybrid_search_with_rank_fusion(
                collection, query_text, query_vector, top_k, config
            )
        else:
            return await hybrid_search_with_score_fusion(
                collection, query_text, query_vector, top_k, config
            )


# Factory function for easy integration
def create_hybrid_searcher(
    db: "AsyncDatabase",
    workspace: str = "",
    vector_weight: float = 0.6,
    text_weight: float = 0.4,
    cosine_threshold: float = 0.3,
) -> MongoDBHybridSearcher:
    """
    Create a MongoDB hybrid searcher with custom configuration.

    Args:
        db: MongoDB database instance
        workspace: Workspace prefix for collections
        vector_weight: Weight for vector search (0.0 to 1.0)
        text_weight: Weight for text search (0.0 to 1.0)
        cosine_threshold: Minimum cosine similarity threshold

    Returns:
        Configured MongoDBHybridSearcher instance
    """
    config = MongoDBHybridSearchConfig(
        vector_weight=vector_weight,
        text_weight=text_weight,
        cosine_threshold=cosine_threshold,
    )

    return MongoDBHybridSearcher(db, workspace, config)
