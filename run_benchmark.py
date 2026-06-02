"""Quick benchmark runner script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from rag_benchmark import Benchmark, BenchmarkConfig

# All available vector databases
ALL_VECTORDBS = [
    "chromadb",      # Local, always available
    "faiss",         # Local, always available
    "pinecone",      # Cloud, needs API key
    "milvus",        # Needs server or Zilliz Cloud
    "weaviate",      # Needs server or Weaviate Cloud
    "qdrant",        # Needs server or Qdrant Cloud
    "vald",          # Needs server
    "turbopuffer",   # Cloud, needs API key
    "pgvector",      # Needs PostgreSQL with pgvector
    "redis",         # Needs Redis 8 with RediSearch
    "elasticsearch", # Needs Elasticsearch server
]

LOCAL_VECTORDBS = ["chromadb", "faiss"]

def main():
    print("=" * 70)
    print("VectorDB Benchmarker (VectorDB Benchmark)")
    print("=" * 70)
    
    config = BenchmarkConfig.default()
    config.embedding.provider = "sentence-transformers"
    config.embedding.model = "paraphrase-multilingual-MiniLM-L12-v2"
    
    benchmark = Benchmark(config)
    
    result = benchmark.run(databases=["chromadb", "faiss"])
    
    result.print_summary()
    
    benchmark.save_results(result)

if __name__ == "__main__":
    main()
