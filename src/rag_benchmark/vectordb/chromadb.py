"""ChromaDB connector for VectorDB Benchmarker."""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("chromadb")
class ChromaDBConnector(BaseVectorDB):
    """ChromaDB vector database connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.persist_directory = config.get("persist_directory", "./data/chromadb")
        self._collections: Dict[str, Any] = {}
    
    def connect(self) -> None:
        """Connect to ChromaDB."""
        import chromadb
        
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self._collections.clear()
        self.client = None
        self._connected = False
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create or get a collection."""
        # Delete if exists to ensure clean state
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        
        collection = self.client.create_collection(
            name=name,
            metadata={"dimension": dimension, "hnsw:space": "cosine"}
        )
        self._collections[name] = collection
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            self._collections.pop(name, None)
        except Exception:
            pass
    
    def _get_collection(self, name: str):
        """Get or load a collection."""
        if name not in self._collections:
            self._collections[name] = self.client.get_collection(name)
        return self._collections[name]
    
    def add_documents(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to collection."""
        coll = self._get_collection(collection)
        
        kwargs = {
            "ids": ids,
            "embeddings": embeddings,
        }
        if documents:
            kwargs["documents"] = documents
        if metadatas:
            kwargs["metadatas"] = metadatas
            
        coll.add(**kwargs)
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        coll = self._get_collection(collection)
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filters:
            kwargs["where"] = filters
            
        results = coll.query(**kwargs)
        
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                id=results["ids"][0][i],
                score=1 - results["distances"][0][i],  # Convert distance to similarity
                document=results["documents"][0][i] if results["documents"] else None,
                metadata=results["metadatas"][0][i] if results["metadatas"] else None
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        coll = self._get_collection(collection)
        return coll.count()
