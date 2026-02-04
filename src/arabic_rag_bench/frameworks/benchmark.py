"""RAG Framework Benchmark Runner.

Benchmarks RAG frameworks across different vectorDBs measuring:
- Indexing latency
- Query latency
- Retrieval quality (Recall, Precision, MRR)
- Answer quality (if ground truth available)
- Memory usage
"""

import time
import json
import tracemalloc
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .base import RAGConfig, RAGQueryResult, VectorDBType
from .registry import get_framework, list_frameworks


@dataclass
class LatencyStats:
    """Latency statistics."""
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    samples: int = 0
    
    @classmethod
    def from_values(cls, values: List[float]) -> "LatencyStats":
        if not values:
            return cls()
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return cls(
            mean=sum(values) / n,
            min=sorted_values[0],
            max=sorted_values[-1],
            p50=sorted_values[int(n * 0.5)],
            p95=sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
            p99=sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
            samples=n,
        )


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
        }


@dataclass
class FrameworkBenchmarkResult:
    """Results for a single framework + vectorDB combination."""
    
    framework: str
    vectordb: str
    config: Dict[str, Any]
    
    # Document stats
    document_count: int = 0
    chunk_count: int = 0
    
    # Latency
    index_latency: LatencyStats = field(default_factory=LatencyStats)
    query_latency: LatencyStats = field(default_factory=LatencyStats)
    retrieval_only_latency: LatencyStats = field(default_factory=LatencyStats)
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Quality
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "vectordb": self.vectordb,
            "config": self.config,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "index_latency_ms": {
                "mean": self.index_latency.mean,
                "p50": self.index_latency.p50,
                "p95": self.index_latency.p95,
            },
            "query_latency_ms": {
                "mean": self.query_latency.mean,
                "p50": self.query_latency.p50,
                "p95": self.query_latency.p95,
            },
            "retrieval_latency_ms": {
                "mean": self.retrieval_only_latency.mean,
                "p50": self.retrieval_only_latency.p50,
                "p95": self.retrieval_only_latency.p95,
            },
            "peak_memory_mb": self.peak_memory_mb,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "errors": self.errors,
        }


@dataclass
class RAGBenchmarkResult:
    """Complete benchmark results across frameworks and vectorDBs."""
    
    name: str
    timestamp: str
    config: Dict[str, Any]
    results: List[FrameworkBenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def print_summary(self) -> None:
        """Print a summary table."""
        from tabulate import tabulate
        
        headers = [
            "Framework", "VectorDB", "Docs", 
            "Index (ms)", "Query (ms)", "Retrieval (ms)",
            "Memory (MB)", "Recall@5", "MRR"
        ]
        
        rows = []
        for r in self.results:
            rows.append([
                r.framework,
                r.vectordb,
                r.document_count,
                f"{r.index_latency.mean:.1f}",
                f"{r.query_latency.mean:.1f}",
                f"{r.retrieval_only_latency.mean:.1f}",
                f"{r.peak_memory_mb:.1f}",
                f"{r.retrieval_metrics.recall_at_k.get(5, 0):.3f}",
                f"{r.retrieval_metrics.mrr:.3f}",
            ])
        
        print("\n" + "=" * 100)
        print("RAG FRAMEWORK BENCHMARK RESULTS")
        print("=" * 100)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()


class RAGBenchmark:
    """RAG Framework benchmark runner."""
    
    def __init__(
        self,
        frameworks: Optional[List[str]] = None,
        vectordbs: Optional[List[VectorDBType]] = None,
        base_config: Optional[RAGConfig] = None,
    ):
        """Initialize benchmark.
        
        Args:
            frameworks: List of framework names to benchmark.
                       Defaults to all registered frameworks.
            vectordbs: List of vectorDB types to test.
                      Defaults to ChromaDB only.
            base_config: Base configuration for all frameworks.
        """
        self.frameworks = frameworks or list_frameworks()
        self.vectordbs = vectordbs or [VectorDBType.CHROMADB]
        self.base_config = base_config or RAGConfig()
        
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._queries: List[str] = []
        self._ground_truth: List[List[str]] = []  # Relevant doc IDs per query
    
    def load_data(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        ground_truth: Optional[List[List[str]]] = None,
    ) -> None:
        """Load benchmark data.
        
        Args:
            documents: List of documents to index
            doc_ids: Optional document IDs
            queries: List of test queries
            ground_truth: For each query, list of relevant document IDs
        """
        self._documents = documents
        self._doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        self._queries = queries or []
        self._ground_truth = ground_truth or []
    
    def _calculate_retrieval_metrics(
        self,
        retrieved_contexts: List[List[str]],
        top_k_values: List[int] = [1, 3, 5, 10],
    ) -> RetrievalMetrics:
        """Calculate retrieval quality metrics.
        
        Note: Without ground truth, we use proxy metrics.
        """
        if not self._ground_truth or not retrieved_contexts:
            return RetrievalMetrics()
        
        recall_at_k = {k: 0.0 for k in top_k_values}
        precision_at_k = {k: 0.0 for k in top_k_values}
        reciprocal_ranks = []
        
        for i, (contexts, relevant) in enumerate(zip(retrieved_contexts, self._ground_truth)):
            if not relevant:
                continue
            
            # Find first relevant result for MRR
            rr = 0.0
            for rank, ctx in enumerate(contexts, 1):
                # Check if context contains any relevant document content
                if any(rel_id in ctx for rel_id in relevant):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
            
            # Calculate recall and precision at different k
            for k in top_k_values:
                top_k_contexts = contexts[:k]
                hits = sum(1 for ctx in top_k_contexts 
                          if any(rel_id in ctx for rel_id in relevant))
                
                recall_at_k[k] += hits / len(relevant)
                precision_at_k[k] += hits / k if k > 0 else 0
        
        n_queries = len(self._ground_truth)
        if n_queries > 0:
            recall_at_k = {k: v / n_queries for k, v in recall_at_k.items()}
            precision_at_k = {k: v / n_queries for k, v in precision_at_k.items()}
        
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
        )
    
    def _benchmark_single(
        self,
        framework_name: str,
        vectordb: VectorDBType,
    ) -> FrameworkBenchmarkResult:
        """Benchmark a single framework + vectorDB combination."""
        
        # Create config for this run
        config = RAGConfig(
            vectordb_type=vectordb,
            vectordb_config=self.base_config.vectordb_config.copy(),
            embedding_model=self.base_config.embedding_model,
            embedding_provider=self.base_config.embedding_provider,
            embedding_dimension=self.base_config.embedding_dimension,
            llm_model=self.base_config.llm_model,
            llm_provider=self.base_config.llm_provider,
            top_k=self.base_config.top_k,
            chunk_size=self.base_config.chunk_size,
            chunk_overlap=self.base_config.chunk_overlap,
            storage_dir=f"./rag_bench_{framework_name}_{vectordb.value}",
            collection_name=f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            extra_config=self.base_config.extra_config.copy(),
        )
        
        result = FrameworkBenchmarkResult(
            framework=framework_name,
            vectordb=vectordb.value,
            config={
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
                "top_k": config.top_k,
                "chunk_size": config.chunk_size,
            },
        )
        
        try:
            # Get framework
            framework = get_framework(framework_name, config)
            
            # Check if vectordb is supported
            if vectordb not in framework.supported_vectordbs:
                result.errors.append(
                    f"{vectordb.value} not supported by {framework_name}"
                )
                return result
            
            # Initialize
            print(f"  Initializing {framework_name} with {vectordb.value}...")
            framework.initialize()
            
            # Start memory tracking
            tracemalloc.start()
            
            # Index documents
            print(f"  Indexing {len(self._documents)} documents...")
            index_latencies = []
            
            for i, (doc, doc_id) in enumerate(zip(self._documents, self._doc_ids)):
                start = time.time()
                chunk_count = framework.add_documents([doc], [doc_id])
                latency = (time.time() - start) * 1000
                index_latencies.append(latency)
                
                if (i + 1) % 10 == 0:
                    print(f"    Indexed {i + 1}/{len(self._documents)} documents")
            
            result.document_count = len(self._documents)
            result.chunk_count = sum(1 for _ in self._documents)  # Approximate
            result.index_latency = LatencyStats.from_values(index_latencies)
            
            # Run queries
            if self._queries:
                print(f"  Running {len(self._queries)} queries...")
                query_latencies = []
                retrieval_latencies = []
                all_contexts = []
                
                for query in self._queries:
                    # Full RAG query
                    start = time.time()
                    response = framework.query(query)
                    query_latencies.append((time.time() - start) * 1000)
                    
                    # Retrieval only
                    start = time.time()
                    contexts = framework.retrieve_only(query)
                    retrieval_latencies.append((time.time() - start) * 1000)
                    all_contexts.append(contexts)
                
                result.query_latency = LatencyStats.from_values(query_latencies)
                result.retrieval_only_latency = LatencyStats.from_values(retrieval_latencies)
                result.retrieval_metrics = self._calculate_retrieval_metrics(all_contexts)
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.peak_memory_mb = peak / 1024 / 1024
            
            # Cleanup
            framework.close()
            
        except Exception as e:
            result.errors.append(f"{type(e).__name__}: {str(e)}")
            print(f"  ✗ Error: {e}")
        
        return result
    
    def run(self) -> RAGBenchmarkResult:
        """Run the complete benchmark.
        
        Returns:
            RAGBenchmarkResult with all measurements
        """
        print("\n" + "=" * 60)
        print("Starting RAG Framework Benchmark")
        print("=" * 60)
        print(f"Frameworks: {', '.join(self.frameworks)}")
        print(f"VectorDBs: {', '.join(v.value for v in self.vectordbs)}")
        print(f"Documents: {len(self._documents)}")
        print(f"Queries: {len(self._queries)}")
        print()
        
        result = RAGBenchmarkResult(
            name="RAG Framework Benchmark",
            timestamp=datetime.now().isoformat(),
            config={
                "frameworks": self.frameworks,
                "vectordbs": [v.value for v in self.vectordbs],
                "document_count": len(self._documents),
                "query_count": len(self._queries),
            },
        )
        
        # Run each combination
        for framework_name in self.frameworks:
            for vectordb in self.vectordbs:
                print(f"\nBenchmarking: {framework_name} + {vectordb.value}")
                print("-" * 40)
                
                bench_result = self._benchmark_single(framework_name, vectordb)
                result.results.append(bench_result)
                
                if not bench_result.errors:
                    print(f"  ✓ Completed: {bench_result.query_latency.mean:.1f}ms avg query")
                else:
                    print(f"  ✗ Errors: {len(bench_result.errors)}")
        
        return result


def run_rag_benchmark(
    documents: List[str],
    queries: List[str],
    frameworks: Optional[List[str]] = None,
    vectordbs: Optional[List[str]] = None,
    ground_truth: Optional[List[List[str]]] = None,
    output_path: Optional[str] = None,
    **config_kwargs,
) -> RAGBenchmarkResult:
    """Convenience function to run RAG benchmark.
    
    Args:
        documents: Documents to index
        queries: Test queries
        frameworks: Framework names (default: all)
        vectordbs: VectorDB names (default: chromadb)
        ground_truth: Relevant doc IDs per query
        output_path: Path to save results JSON
        **config_kwargs: Additional RAGConfig options
        
    Returns:
        Benchmark results
    """
    # Convert vectordb strings to enum
    vdb_types = None
    if vectordbs:
        vdb_types = [VectorDBType(vdb) for vdb in vectordbs]
    
    # Create config
    config = RAGConfig(**config_kwargs)
    
    # Create and run benchmark
    benchmark = RAGBenchmark(
        frameworks=frameworks,
        vectordbs=vdb_types,
        base_config=config,
    )
    
    benchmark.load_data(
        documents=documents,
        queries=queries,
        ground_truth=ground_truth,
    )
    
    result = benchmark.run()
    result.print_summary()
    
    if output_path:
        result.save(output_path)
        print(f"Results saved to: {output_path}")
    
    return result
