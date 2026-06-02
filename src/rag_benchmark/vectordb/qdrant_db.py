"""Qdrant connector for VectorDB Benchmarker."""

import os
from typing import List, Dict, Any, Optional

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("qdrant")
class QdrantConnector(BaseVectorDB):
    """Qdrant vector database connector.
    
    Qdrant is a high-performance vector similarity search engine.
    Supports both self-hosted and Qdrant Cloud.
    
    Config options:
        - url: Qdrant server URL (default: "http://localhost:6333")
        - api_key: API key for Qdrant Cloud
        - prefer_grpc: Use gRPC instead of REST (default: False)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.api_key = config.get("api_key", os.getenv("QDRANT_API_KEY"))
        self.prefer_grpc = config.get("prefer_grpc", False)
        self.client = None
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Qdrant."""
        from qdrant_client import QdrantClient
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
        )
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        if self.client:
            self.client.close()
        self.client = None
        self._dimensions.clear()
        self._connected = False
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.models import Distance, VectorParams
        
        # Delete if exists
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        
        # Create collection
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )
        
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection."""
        try:
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
            self._dimensions.pop(name, None)
        except Exception:
            pass
    
    def add_documents(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to Qdrant collection."""
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            payload = {}
            if documents and i < len(documents):
                payload["document"] = documents[i]
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            
            # Qdrant requires numeric IDs, so we hash the string ID
            numeric_id = abs(hash(doc_id)) % (2**63)
            payload["_original_id"] = doc_id
            
            points.append(PointStruct(
                id=numeric_id,
                vector=embedding,
                payload=payload,
            ))
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection, points=batch)
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(FieldCondition(
                    key=k,
                    match=MatchValue(value=v),
                ))
            query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(SearchResult(
                id=payload.get("_original_id", str(hit.id)),
                score=hit.score,
                document=payload.get("document"),
                metadata={k: v for k, v in payload.items() 
                         if k not in ["document", "_original_id"]},
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        info = self.client.get_collection(collection)
        return info.points_count or 0
