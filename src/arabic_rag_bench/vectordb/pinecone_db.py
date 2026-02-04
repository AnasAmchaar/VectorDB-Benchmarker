"""Pinecone connector for Arabic RAG Benchmark."""

import os
import time
from typing import List, Dict, Any, Optional

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("pinecone")
class PineconeConnector(BaseVectorDB):
    """Pinecone vector database connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("PINECONE_API_KEY")
        self.cloud = config.get("cloud", "aws")
        self.region = config.get("region", "us-east-1")
        self.pc = None
        self._indexes: Dict[str, Any] = {}
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Pinecone."""
        from pinecone import Pinecone
        
        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
        
        self.pc = Pinecone(api_key=self.api_key)
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._indexes.clear()
        self._dimensions.clear()
        self.pc = None
        self._connected = False
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a Pinecone index."""
        from pinecone import ServerlessSpec
        
        # Check if index exists
        existing = [idx.name for idx in self.pc.list_indexes()]
        
        if name in existing:
            # Delete existing to ensure clean state
            self.pc.delete_index(name)
            time.sleep(5)  # Wait for deletion
        
        # Create new index
        self.pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self.cloud,
                region=self.region
            )
        )
        
        # Wait for index to be ready
        while not self.pc.describe_index(name).status["ready"]:
            time.sleep(1)
        
        self._indexes[name] = self.pc.Index(name)
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete a Pinecone index."""
        try:
            self.pc.delete_index(name)
            self._indexes.pop(name, None)
            self._dimensions.pop(name, None)
        except Exception:
            pass
    
    def _get_index(self, name: str):
        """Get or connect to an index."""
        if name not in self._indexes:
            self._indexes[name] = self.pc.Index(name)
        return self._indexes[name]
    
    def add_documents(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to Pinecone index."""
        index = self._get_index(collection)
        
        # Prepare vectors with metadata
        vectors = []
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            metadata = {}
            if metadatas and i < len(metadatas):
                metadata.update(metadatas[i])
            if documents and i < len(documents):
                # Store document text in metadata (Pinecone doesn't have separate doc storage)
                metadata["_document"] = documents[i][:1000]  # Limit size
            
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        index = self._get_index(collection)
        
        kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        if filters:
            kwargs["filter"] = filters
        
        response = index.query(**kwargs)
        
        results = []
        for match in response.matches:
            metadata = match.metadata or {}
            document = metadata.pop("_document", None)
            
            results.append(SearchResult(
                id=match.id,
                score=match.score,
                document=document,
                metadata=metadata if metadata else None
            ))
        
        return results
    
    def count(self, collection: str) -> int:
        """Count documents in index."""
        index = self._get_index(collection)
        stats = index.describe_index_stats()
        return stats.total_vector_count
