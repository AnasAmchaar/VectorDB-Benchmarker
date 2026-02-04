"""Base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers.
    
    To add a new embedding provider:
    1. Create a new file in embeddings/ directory
    2. Inherit from BaseEmbedding
    3. Implement all abstract methods
    4. Register with @register_embedding("name") decorator
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = config.get("model", "default")
        self._dimension: int = 0
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple queries.
        
        Args:
            texts: List of queries to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.embed_query(text) for text in texts]
