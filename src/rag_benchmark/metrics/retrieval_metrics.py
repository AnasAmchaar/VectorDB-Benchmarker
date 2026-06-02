"""Retrieval quality metrics for evaluation."""

from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Calculate retrieval quality metrics."""
    
    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs in order
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        if not relevant:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        hits = len(retrieved_k & relevant)
        return hits / len(relevant)
    
    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs in order
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        hits = sum(1 for doc in retrieved_k if doc in relevant)
        return hits / k
    
    @staticmethod
    def mrr(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank (for single query).
        
        Args:
            retrieved: List of retrieved document IDs in order
            relevant: Set of relevant document IDs
            
        Returns:
            Reciprocal rank (0.0 to 1.0)
        """
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate NDCG@K (binary relevance).
        
        Args:
            retrieved: List of retrieved document IDs in order
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG score (0.0 to 1.0)
        """
        import math
        
        if not relevant:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], 1):
            if doc in relevant:
                dcg += 1.0 / math.log2(i + 1)
        
        # Calculate ideal DCG
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_all(
        retrieved: List[str],
        relevant: Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Calculate all retrieval metrics.
        
        Args:
            retrieved: List of retrieved document IDs in order
            relevant: Set of relevant document IDs
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        for k in k_values:
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k)
        
        metrics["mrr"] = RetrievalMetrics.mrr(retrieved, relevant)
        
        return metrics
    
    @staticmethod
    def average_metrics(
        all_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Average metrics across multiple queries.
        
        Args:
            all_metrics: List of metric dictionaries
            
        Returns:
            Averaged metrics
        """
        if not all_metrics:
            return {}
        
        keys = all_metrics[0].keys()
        return {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in keys
        }
