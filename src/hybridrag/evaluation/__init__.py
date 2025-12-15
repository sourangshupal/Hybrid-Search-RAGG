"""
HybridRAG Evaluation Framework.

Provides RAGAS-based evaluation for measuring RAG quality:
- Faithfulness: Is the answer factually accurate based on context?
- Answer Relevance: Is the answer relevant to the question?
- Context Recall: Is all relevant information retrieved?
- Context Precision: Is retrieved context clean without noise?

Usage:
    from hybridrag.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    results = await evaluator.run()
"""

from .ragas_eval import RAGEvaluator, run_evaluation

__all__ = ["RAGEvaluator", "run_evaluation"]
