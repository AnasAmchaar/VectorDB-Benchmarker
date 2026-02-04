"""Benchmark metrics collection and reporting."""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from statistics import mean, stdev
import json


@dataclass
class LatencyMetrics:
    """Latency measurements."""
    values_ms: List[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        return mean(self.values_ms) if self.values_ms else 0.0
    
    @property
    def std(self) -> float:
        return stdev(self.values_ms) if len(self.values_ms) > 1 else 0.0
    
    @property
    def min(self) -> float:
        return min(self.values_ms) if self.values_ms else 0.0
    
    @property
    def max(self) -> float:
        return max(self.values_ms) if self.values_ms else 0.0
    
    @property
    def p50(self) -> float:
        return self._percentile(50)
    
    @property
    def p95(self) -> float:
        return self._percentile(95)
    
    @property
    def p99(self) -> float:
        return self._percentile(99)
    
    def _percentile(self, p: int) -> float:
        if not self.values_ms:
            return 0.0
        sorted_vals = sorted(self.values_ms)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": round(self.mean, 3),
            "std_ms": round(self.std, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
        }


@dataclass
class MemoryMetrics:
    """Memory usage measurements."""
    peak_mb: float = 0.0
    current_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "peak_mb": round(self.peak_mb, 3),
            "current_mb": round(self.current_mb, 3),
        }


@dataclass
class BenchmarkMetrics:
    """Complete benchmark metrics for a vector database."""
    
    db_name: str
    index_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    query_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    document_count: int = 0
    dimension: int = 0
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.db_name,
            "document_count": self.document_count,
            "dimension": self.dimension,
            "index_latency": self.index_latency.to_dict(),
            "query_latency": self.query_latency.to_dict(),
            "memory": self.memory.to_dict(),
            "retrieval": self.retrieval_metrics,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class MetricsCollector:
    """Collect benchmark metrics during execution."""
    
    def __init__(self, db_name: str):
        self.metrics = BenchmarkMetrics(db_name=db_name)
        self._tracking_memory = False
    
    def start_memory_tracking(self) -> None:
        """Start tracking memory usage."""
        tracemalloc.start()
        self._tracking_memory = True
    
    def stop_memory_tracking(self) -> None:
        """Stop tracking and record memory usage."""
        if self._tracking_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.metrics.memory = MemoryMetrics(
                peak_mb=peak / (1024 * 1024),
                current_mb=current / (1024 * 1024)
            )
            self._tracking_memory = False
    
    def measure_latency(self, func: Callable, *args, **kwargs) -> tuple:
        """Measure latency of a function call.
        
        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        return result, latency_ms
    
    def record_index_latency(self, latency_ms: float) -> None:
        """Record an indexing latency measurement."""
        self.metrics.index_latency.values_ms.append(latency_ms)
    
    def record_query_latency(self, latency_ms: float) -> None:
        """Record a query latency measurement."""
        self.metrics.query_latency.values_ms.append(latency_ms)
    
    def set_document_info(self, count: int, dimension: int) -> None:
        """Set document count and dimension."""
        self.metrics.document_count = count
        self.metrics.dimension = dimension
    
    def set_retrieval_metrics(self, metrics: Dict[str, float]) -> None:
        """Set retrieval quality metrics."""
        self.metrics.retrieval_metrics = metrics
    
    def get_metrics(self) -> BenchmarkMetrics:
        """Get the collected metrics."""
        return self.metrics
