"""Base class for RAG framework connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class VectorDBType(str, Enum):
    """Supported vector databases."""
    CHROMADB = "chromadb"
    FAISS = "faiss"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    VALD = "vald"
    TURBOPUFFER = "turbopuffer"
    PGVECTOR = "pgvector"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


@dataclass
class RAGConfig:
    """Configuration for RAG framework."""
    
    # Vector database settings
    vectordb_type: VectorDBType = VectorDBType.CHROMADB
    vectordb_config: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding settings
    embedding_model: str = "text-embedding-004"
    embedding_provider: str = "gemini"
    embedding_dimension: int = 768
    
    # LLM settings
    llm_model: str = "gemini-2.0-flash"
    llm_provider: str = "gemini"
    
    # Retrieval settings
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Storage
    storage_dir: str = "./rag_storage"
    collection_name: str = "arabic_bench"
    
    # Framework-specific settings
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGQueryResult:
    """Result from a RAG query."""
    
    query: str
    answer: str
    contexts: List[str]  # Retrieved context chunks
    source_docs: List[str] = field(default_factory=list)  # Source document IDs
    scores: List[float] = field(default_factory=list)  # Retrieval scores
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRAGFramework(ABC):
    """Abstract base class for RAG framework connectors.
    
    Each framework implementation should:
    1. Initialize with a specific vectorDB backend
    2. Support document ingestion with chunking
    3. Support retrieval-augmented generation queries
    4. Expose retrieval contexts separately for evaluation
    
    To add a new framework:
    1. Create a new file in frameworks/ directory
    2. Inherit from BaseRAGFramework
    3. Implement all abstract methods
    4. Register with @register_framework("name") decorator
    """
    
    def __init__(self, config: RAGConfig):
        """Initialize the RAG framework.
        
        Args:
            config: RAG configuration including vectordb settings
        """
        self.config = config
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if framework is initialized."""
        return self._initialized
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the framework name."""
        pass
    
    @property
    @abstractmethod
    def supported_vectordbs(self) -> List[VectorDBType]:
        """Return list of supported vector databases."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the framework with configured vectorDB.
        
        This should:
        - Set up the vectorDB connection
        - Initialize embedding model
        - Initialize LLM
        - Create necessary indexes/collections
        """
        pass
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add documents to the RAG system.
        
        Documents will be chunked and indexed automatically.
        
        Args:
            documents: List of document texts
            doc_ids: Optional document IDs
            metadatas: Optional metadata for each document
            
        Returns:
            Number of chunks created
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """Query the RAG system.
        
        Args:
            query: The query text
            top_k: Number of contexts to retrieve (uses config default if None)
            
        Returns:
            RAGQueryResult with answer and retrieved contexts
        """
        pass
    
    @abstractmethod
    def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Retrieve contexts without generating an answer.
        
        Useful for evaluating retrieval quality separately.
        
        Args:
            query: The query text
            top_k: Number of contexts to retrieve
            
        Returns:
            List of retrieved context chunks
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the index."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get framework statistics.
        
        Returns:
            Dictionary with stats like document count, chunk count, etc.
        """
        return {
            "framework": self.name,
            "vectordb": self.config.vectordb_type.value,
            "initialized": self._initialized,
        }
