"""Arabic RAG Benchmark - Extensible benchmarking tool for VectorDBs and RAG systems."""

__version__ = "0.1.0"
__author__ = "Arabic RAG Bench Team"

from .benchmark import Benchmark
from .config import BenchmarkConfig

__all__ = ["Benchmark", "BenchmarkConfig", "__version__"]
