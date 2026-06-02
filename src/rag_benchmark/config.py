"""Configuration management for VectorDB Benchmarker."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: str = "gemini"  # gemini, openai, sentence-transformers
    model: str = "text-embedding-004"
    api_key_env: str = "GOOGLE_API_KEY"
    batch_size: int = 20
    
    
@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    name: str = "chromadb"
    connection: Dict[str, Any] = field(default_factory=dict)
    index_name: str = "benchmark"
    dimension: int = 768  # Gemini embedding dimension
    

@dataclass
class DataConfig:
    """Configuration for benchmark data."""
    source: str = "synthetic"  # builtin, file, huggingface
    file_path: Optional[str] = None
    dataset_name: Optional[str] = None
    num_documents: int = 50
    num_queries: int = 10
    language: str = "ar"


@dataclass 
class MetricsConfig:
    """Configuration for benchmark metrics."""
    measure_latency: bool = True
    measure_memory: bool = True
    measure_recall: bool = True
    num_query_iterations: int = 10
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    

@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    name: str = "vectordb-benchmark"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectordbs: List[VectorDBConfig] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        config = cls()
        config.name = data.get("name", config.name)
        config.output_dir = data.get("output_dir", config.output_dir)
        
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
            
        if "vectordbs" in data:
            config.vectordbs = [
                VectorDBConfig(**db) for db in data["vectordbs"]
            ]
            
        if "data" in data:
            config.data = DataConfig(**data["data"])
            
        if "metrics" in data:
            config.metrics = MetricsConfig(**data["metrics"])
            
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(i) for i in obj]
            return obj
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(to_dict(self), f, allow_unicode=True, default_flow_style=False)
    
    @classmethod
    def default(cls) -> "BenchmarkConfig":
        """Create default configuration with common VectorDBs."""
        config = cls()
        config.vectordbs = [
            VectorDBConfig(name="chromadb", connection={"persist_directory": "./data/chromadb"}),
            VectorDBConfig(name="faiss", connection={"index_path": "./data/faiss"}),
            VectorDBConfig(name="pinecone", connection={"index_name": "benchmark"}),
        ]
        return config
