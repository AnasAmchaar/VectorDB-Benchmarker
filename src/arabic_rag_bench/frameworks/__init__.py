"""RAG Framework connectors for benchmarking.

Supported frameworks:
- LightRAG: Knowledge graph-based RAG
- RAGAnything: Multimodal RAG framework  
- LlamaIndex: Popular RAG framework with many integrations
- LangChain: Flexible RAG chains

Each framework can be tested with different vectorDBs to measure
end-to-end RAG performance.
"""

from .base import BaseRAGFramework, RAGConfig, RAGQueryResult, VectorDBType
from .registry import get_framework, list_frameworks, register_framework
from .benchmark import RAGBenchmark, RAGBenchmarkResult, run_rag_benchmark

# Import framework implementations to register them
from . import llamaindex_framework  # noqa: F401
from . import langchain_framework  # noqa: F401
from . import lightrag_framework  # noqa: F401
from . import raganything_framework  # noqa: F401

__all__ = [
    # Base classes
    "BaseRAGFramework",
    "RAGConfig", 
    "RAGQueryResult",
    "VectorDBType",
    # Registry
    "get_framework",
    "list_frameworks",
    "register_framework",
    # Benchmark
    "RAGBenchmark",
    "RAGBenchmarkResult", 
    "run_rag_benchmark",
]
