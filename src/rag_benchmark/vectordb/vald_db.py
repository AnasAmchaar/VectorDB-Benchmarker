"""Vald connector for VectorDB Benchmarker."""

import os
from typing import List, Dict, Any, Optional
import json

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("vald")
class ValdConnector(BaseVectorDB):
    """Vald vector database connector.
    
    Vald is a highly scalable distributed vector search engine.
    Designed for billion-scale vector search.
    
    Config options:
        - host: Vald gateway host (default: "localhost")
        - port: Vald gateway port (default: 8081)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", os.getenv("VALD_HOST", "localhost"))
        self.port = config.get("port", int(os.getenv("VALD_PORT", "8081")))
        self.channel = None
        self.insert_client = None
        self.search_client = None
        self.remove_client = None
        self._dimensions: Dict[str, int] = {}
        self._documents: Dict[str, Dict[str, Dict[str, Any]]] = {}  # collection -> {id: data}
    
    def connect(self) -> None:
        """Connect to Vald."""
        import grpc
        from vald.v1.vald import insert_pb2_grpc, search_pb2_grpc, remove_pb2_grpc
        
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.insert_client = insert_pb2_grpc.InsertStub(self.channel)
        self.search_client = search_pb2_grpc.SearchStub(self.channel)
        self.remove_client = remove_pb2_grpc.RemoveStub(self.channel)
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Vald."""
        if self.channel:
            self.channel.close()
        self.channel = None
        self.insert_client = None
        self.search_client = None
        self.remove_client = None
        self._dimensions.clear()
        self._documents.clear()
        self._connected = False
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Initialize collection tracking (Vald doesn't have explicit collections)."""
        # Vald doesn't have explicit collection management
        # We track collections locally
        self._dimensions[name] = dimension
        self._documents[name] = {}
    
    def delete_collection(self, name: str) -> None:
        """Delete all vectors in collection."""
        from vald.v1.payload import payload_pb2
        
        if name in self._documents:
            # Remove all vectors in this collection
            for doc_id in list(self._documents[name].keys()):
                try:
                    request = payload_pb2.Remove.Request(
                        id=payload_pb2.Object.ID(id=f"{name}_{doc_id}")
                    )
                    self.remove_client.Remove(request)
                except Exception:
                    pass
            
            self._documents.pop(name, None)
            self._dimensions.pop(name, None)
    
    def add_documents(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to Vald."""
        from vald.v1.payload import payload_pb2
        
        if collection not in self._documents:
            self._documents[collection] = {}
        
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            # Vald uses global IDs, so prefix with collection name
            vald_id = f"{collection}_{doc_id}"
            
            # Store document locally (Vald only stores vectors)
            self._documents[collection][doc_id] = {
                "document": documents[i] if documents and i < len(documents) else None,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else None,
            }
            
            # Insert vector
            request = payload_pb2.Insert.Request(
                vector=payload_pb2.Object.Vector(
                    id=vald_id,
                    vector=embedding,
                ),
                config=payload_pb2.Insert.Config(
                    skip_strict_exist_check=True,
                ),
            )
            
            self.insert_client.Insert(request)
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        from vald.v1.payload import payload_pb2
        
        request = payload_pb2.Search.Request(
            vector=query_embedding,
            config=payload_pb2.Search.Config(
                num=top_k,
                radius=-1,  # No radius limit
                epsilon=0.1,
                timeout=3000000000,  # 3 seconds in nanoseconds
            ),
        )
        
        response = self.search_client.Search(request)
        
        search_results = []
        collection_prefix = f"{collection}_"
        
        for result in response.results:
            # Filter to only this collection
            if not result.id.startswith(collection_prefix):
                continue
            
            doc_id = result.id[len(collection_prefix):]
            doc_data = self._documents.get(collection, {}).get(doc_id, {})
            
            # Apply filters if provided
            if filters and doc_data.get("metadata"):
                match = all(
                    doc_data["metadata"].get(k) == v 
                    for k, v in filters.items()
                )
                if not match:
                    continue
            
            search_results.append(SearchResult(
                id=doc_id,
                score=1 - result.distance,  # Convert distance to similarity
                document=doc_data.get("document"),
                metadata=doc_data.get("metadata"),
            ))
        
        return search_results[:top_k]
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        return len(self._documents.get(collection, {}))
