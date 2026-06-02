"""Turbopuffer connector for VectorDB Benchmarker."""

import os
from typing import List, Dict, Any, Optional

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("turbopuffer")
class TurbopufferConnector(BaseVectorDB):
    """Turbopuffer vector database connector.
    
    Turbopuffer is a serverless vector database optimized for speed.
    
    Config options:
        - api_key: Turbopuffer API key (or TURBOPUFFER_API_KEY env var)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", os.getenv("TURBOPUFFER_API_KEY"))
        self._namespaces: Dict[str, Any] = {}
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Turbopuffer."""
        import turbopuffer as tpuf
        
        if not self.api_key:
            raise ValueError("Turbopuffer API key not provided")
        
        tpuf.api_key = self.api_key
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Turbopuffer."""
        self._namespaces.clear()
        self._dimensions.clear()
        self._connected = False
    
    def _get_namespace(self, name: str):
        """Get or create a namespace."""
        import turbopuffer as tpuf
        
        if name not in self._namespaces:
            self._namespaces[name] = tpuf.Namespace(name)
        return self._namespaces[name]
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a Turbopuffer namespace (collection)."""
        # Delete if exists
        try:
            ns = self._get_namespace(name)
            ns.delete_all()
        except Exception:
            pass
        
        self._dimensions[name] = dimension
        self._namespaces[name] = None  # Will be recreated on first use
    
    def delete_collection(self, name: str) -> None:
        """Delete a Turbopuffer namespace."""
        try:
            ns = self._get_namespace(name)
            ns.delete_all()
            self._namespaces.pop(name, None)
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
        """Add documents to Turbopuffer."""
        ns = self._get_namespace(collection)
        
        # Prepare data for upsert
        upsert_ids = []
        upsert_vectors = []
        upsert_attributes = {}
        
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            upsert_ids.append(doc_id)
            upsert_vectors.append(embedding)
            
            # Collect attributes
            if documents and i < len(documents):
                if "document" not in upsert_attributes:
                    upsert_attributes["document"] = []
                upsert_attributes["document"].append(documents[i][:10000])  # Limit size
            
            if metadatas and i < len(metadatas):
                for k, v in metadatas[i].items():
                    if k not in upsert_attributes:
                        upsert_attributes[k] = []
                    upsert_attributes[k].append(v)
        
        ns.upsert(
            ids=upsert_ids,
            vectors=upsert_vectors,
            attributes=upsert_attributes if upsert_attributes else None,
        )
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        ns = self._get_namespace(collection)
        
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_attributes": ["document"],
        }
        
        if filters:
            # Convert to Turbopuffer filter format
            query_params["filters"] = filters
        
        results = ns.query(**query_params)
        
        search_results = []
        for row in results:
            attrs = row.attributes or {}
            search_results.append(SearchResult(
                id=row.id,
                score=row.dist,
                document=attrs.get("document"),
                metadata={k: v for k, v in attrs.items() if k != "document"},
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        try:
            ns = self._get_namespace(collection)
            # Turbopuffer doesn't have a direct count API
            # Use a dummy query to check
            return len(ns.query(vector=[0.0] * self._dimensions.get(collection, 768), top_k=10000))
        except Exception:
            return 0
