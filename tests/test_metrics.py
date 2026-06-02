import sys
import os
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_benchmark.metrics import RetrievalMetrics

def test_recall():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    relevant = {"doc_3", "doc_5"}
    
    metrics = RetrievalMetrics.calculate_all(retrieved, relevant, [3, 5])
    
    # Recall@3: doc_3 is in top 3. We have 2 relevant total. Recall = 1/2 = 0.5
    assert metrics["recall@3"] == 0.5
    
    # Recall@5: both are in top 5. Recall = 2/2 = 1.0
    assert metrics["recall@5"] == 1.0

def test_mrr():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = {"doc_2"}
    
    metrics = RetrievalMetrics.calculate_all(retrieved, relevant, [3])
    
    # MRR: relevant doc_2 is at rank 2 (1-indexed). MRR = 1/2 = 0.5
    assert metrics["mrr"] == 0.5

def test_no_relevant():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = {"doc_4"}
    
    metrics = RetrievalMetrics.calculate_all(retrieved, relevant, [3])
    
    assert metrics["recall@3"] == 0.0
    assert metrics["mrr"] == 0.0

if __name__ == "__main__":
    test_recall()
    test_mrr()
    test_no_relevant()
    print("All tests passed!")
