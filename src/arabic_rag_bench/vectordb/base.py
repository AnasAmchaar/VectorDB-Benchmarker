"""Base class for vector database connectors."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from a vector search."""
    id: str
    score: float
    document: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseVectorDB(ABC):
    """Abstract base class for vector database connectors.
    
    To add a new vector database:
    1. Create a new file in vectordb/ directory
    2. Inherit from BaseVectorDB
    3. Implement all abstract methods
    4. Register with @register_vectordb("name") decorator
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector database connector.
        
        Args:
            config: Configuration dictionary containing connection details
        """
        self.config = config
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection/index.
        
        Args:
            name: Name of the collection
            dimension: Dimension of the vectors
        """
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection/index.
        
        Args:
            name: Name of the collection to delete
        """
        pass
    
    @abstractmethod
    def add_documents(
        self, 
        collection: str,
        ids: List[str],
        embeddings: List[List[float]], 
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents with their embeddings to the database.
        
        Args:
            collection: Name of the collection
            ids: Unique identifiers for each document
            embeddings: Vector embeddings for each document
            documents: Original text documents (optional)
            metadatas: Metadata for each document (optional)
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        collection: str,
        query_embedding: List[float], 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents.
        
        Args:
            collection: Name of the collection
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def count(self, collection: str) -> int:
        """Return the number of documents in a collection.
        
        Args:
            collection: Name of the collection
            
        Returns:
            Number of documents
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
