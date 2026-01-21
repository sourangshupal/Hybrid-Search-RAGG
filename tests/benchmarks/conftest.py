"""
Benchmark configuration.

Provides fixtures for performance benchmarking.
"""

import pytest


@pytest.fixture(scope="session")
def benchmark_queries():
    """Sample queries for benchmarking."""
    return [
        "What is MongoDB Atlas?",
        "How does vector search work?",
        "mongodb hybrid search configuration",
        "atlas search full text features",
        "knowledge graph mongodb",
    ]
