"""Main benchmark engine for VectorDB Benchmarker."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import BenchmarkConfig, VectorDBConfig
from .data import DataLoader, Corpus
from .metrics import BenchmarkMetrics, RetrievalMetrics
from .metrics.benchmark_metrics import MetricsCollector

# Import to register all implementations
from .vectordb import (  # noqa: F401
    chromadb, faiss_db, pinecone_db, milvus_db, weaviate_db,
    qdrant_db, vald_db, turbopuffer_db, pgvector_db, redis_db, elasticsearch_db
)
from .embeddings import gemini, openai_embed, sentence_transformers_embed  # noqa: F401

from .vectordb import get_vectordb
from .embeddings import get_embedding


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    
    name: str
    timestamp: str
    config: Dict[str, Any]
    results: List[BenchmarkMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [r.to_dict() for r in self.results]
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def print_summary(self) -> None:
        """Print a summary table of results."""
        from tabulate import tabulate
        
        headers = [
            "Database", "Docs", "Index (ms)", 
            "Query Mean (ms)", "Query P95 (ms)",
            "Memory (MB)", "Recall@5", "MRR"
        ]
        
        rows = []
        for r in self.results:
            rows.append([
                r.db_name,
                r.document_count,
                f"{r.index_latency.mean:.2f}",
                f"{r.query_latency.mean:.2f}",
                f"{r.query_latency.p95:.2f}",
                f"{r.memory.peak_mb:.2f}",
                f"{r.retrieval_metrics.get('recall@5', 0):.3f}",
                f"{r.retrieval_metrics.get('mrr', 0):.3f}",
            ])
        
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS - VectorDB Benchmarker")
        print("=" * 80)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()


class Benchmark:
    """Main benchmark runner."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark.
        
        Args:
            config: Benchmark configuration. Uses default if not provided.
        """
        self.config = config or BenchmarkConfig.default()
        self._corpus: Optional[Corpus] = None
        self._embeddings: Optional[List[List[float]]] = None
        self._query_embeddings: Optional[List[List[float]]] = None
    
    def load_data(self) -> Corpus:
        """Load benchmark data."""
        if self._corpus is None:
            self._corpus = DataLoader.load(
                source=self.config.data.source,
                file_path=self.config.data.file_path
            )
        return self._corpus
    
    def generate_embeddings(self, force: bool = False) -> None:
        """Generate embeddings for documents and queries.
        
        Args:
            force: Force regeneration even if already cached
        """
        if self._embeddings is not None and not force:
            return
        
        corpus = self.load_data()
        embedding_provider = get_embedding(
            self.config.embedding.provider,
            {
                "model": self.config.embedding.model,
                "batch_size": self.config.embedding.batch_size,
            }
        )
        
        print(f"Generating embeddings using {self.config.embedding.provider}...")
        print(f"  Model: {self.config.embedding.model}")
        print(f"  Documents: {len(corpus.documents)}")
        
        doc_texts = [doc.get("text", "") for doc in corpus.documents]
        self._embeddings = embedding_provider.embed_documents(doc_texts)
        self._query_embeddings = embedding_provider.embed_queries(corpus.get_query_texts())
        
        # Update config with actual dimension
        self.config.embedding.dimension = len(self._embeddings[0])
        print(f"  Dimension: {self.config.embedding.dimension}")
    
    def run(self, databases: Optional[List[str]] = None) -> BenchmarkResult:
        """Run the benchmark.
        
        Args:
            databases: List of database names to benchmark. 
                      Uses config if not provided.
                      
        Returns:
            BenchmarkResult with all metrics
        """
        # Load data and generate embeddings
        corpus = self.load_data()
        self.generate_embeddings()
        
        # Determine which databases to test
        if databases:
            db_configs = [
                VectorDBConfig(name=db, dimension=len(self._embeddings[0]))
                for db in databases
            ]
        else:
            db_configs = self.config.vectordbs
        
        if not db_configs:
            db_configs = [
                VectorDBConfig(name="chromadb"),
                VectorDBConfig(name="faiss"),
            ]
        
        # Create result
        result = BenchmarkResult(
            name=self.config.name,
            timestamp=datetime.now().isoformat(),
            config={
                "embedding": {
                    "provider": self.config.embedding.provider,
                    "model": self.config.embedding.model,
                    "dimension": len(self._embeddings[0]),
                },
                "data": {
                    "source": self.config.data.source,
                    "num_documents": len(corpus.documents),
                    "num_queries": len(corpus.queries),
                }
            }
        )
        
        # Run benchmark for each database
        for db_config in db_configs:
            print(f"\nBenchmarking {db_config.name}...")
            
            try:
                metrics = self._benchmark_database(db_config, corpus)
                result.results.append(metrics)
                print(f"  ✓ Completed: {metrics.query_latency.mean:.2f}ms avg query")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        return result
    
    def _benchmark_database(
        self, 
        db_config: VectorDBConfig,
        corpus: Corpus
    ) -> BenchmarkMetrics:
        """Benchmark a single database."""
        collector = MetricsCollector(db_config.name)
        dimension = len(self._embeddings[0])
        
        db = get_vectordb(db_config.name, db_config.connection)
        
        collection_name = f"vectordb-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            db.connect()
            
            collector.start_memory_tracking()
            
            db.create_collection(collection_name, dimension)
            
            doc_ids = [f"doc_{i}" for i in range(len(corpus.documents))]
            
            doc_texts = [doc.get("text", "") for doc in corpus.documents]
            doc_metas = [doc.get("metadata", {}) for doc in corpus.documents]
            
            _, index_latency = collector.measure_latency(
                db.add_documents,
                collection_name,
                doc_ids,
                self._embeddings,
                doc_texts,
                doc_metas
            )
            collector.record_index_latency(index_latency)
            
            all_query_results = []
            num_iterations = self.config.metrics.num_query_iterations
            
            for iteration in range(num_iterations):
                for i, query_emb in enumerate(self._query_embeddings):
                    results, latency = collector.measure_latency(
                        db.search,
                        collection_name,
                        query_emb,
                        max(self.config.metrics.top_k_values)
                    )
                    collector.record_query_latency(latency)
                    
                    if iteration == 0:
                        all_query_results.append(results)
            
            collector.stop_memory_tracking()
            
            relevance = corpus.get_relevance_judgments()
            all_metrics = []
            
            for i, results in enumerate(all_query_results):
                retrieved_ids = [r.id for r in results]
                relevant_ids = {f"doc_{idx}" for idx in relevance.get(i, [])}
                
                query_metrics = RetrievalMetrics.calculate_all(
                    retrieved_ids,
                    relevant_ids,
                    self.config.metrics.top_k_values
                )
                all_metrics.append(query_metrics)
            
            avg_metrics = RetrievalMetrics.average_metrics(all_metrics)
            collector.set_retrieval_metrics(avg_metrics)
            collector.set_document_info(len(corpus.documents), dimension)
            
            db.delete_collection(collection_name)
            
        finally:
            db.disconnect()
        
        return collector.get_metrics()
    
    def save_results(self, result: BenchmarkResult, filename: Optional[str] = None) -> str:
        """Save benchmark results to file.
        
        Args:
            result: Benchmark results
            filename: Output filename. Auto-generated if not provided.
            
        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        output_path = output_dir / filename
        result.save(str(output_path))
        
        print(f"\nResults saved to: {output_path}")
        return str(output_path)
