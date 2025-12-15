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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection
    from pymongo.asynchronous.database import AsyncDatabase

logger = logging.getLogger("hybridrag.mongodb_hybrid")
logger.setLevel(logging.INFO)

# Default RRF constant (MongoDB default is 60, but configurable)
DEFAULT_RRF_CONSTANT = 60


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

    # Hybrid search settings
    vector_weight: float = 0.6  # Weight for vector search in score fusion
    text_weight: float = 0.4  # Weight for text search in score fusion
    use_rank_fusion: bool = True  # Use $rankFusion (RRF) instead of $scoreFusion

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
        # Fall back to vector-only search
        logger.warning("[HYBRID_SEARCH] Falling back to vector-only search")
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
        logger.warning("[HYBRID_SEARCH] Falling back to vector-only search")
        return await vector_only_search(
            collection, query_vector, top_k, config
        )


async def vector_only_search(
    collection: "AsyncCollection",
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Fallback vector-only search when hybrid search is not available.
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    pipeline = [
        {
            "$vectorSearch": {
                "index": config.vector_index_name,
                "path": config.vector_path,
                "queryVector": query_vector,
                "numCandidates": config.vector_num_candidates,
                "limit": top_k,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$match": {"score": {"$gte": config.cosine_threshold}}},
        {"$project": {"vector": 0}},
    ]

    cursor = await collection.aggregate(pipeline, allowDiskUse=True)
    results = await cursor.to_list(length=None)

    return [
        {
            **doc,
            "id": doc.get("_id"),
            "score": doc.get("score"),
            "search_type": "vector_only",
        }
        for doc in results
    ]


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
