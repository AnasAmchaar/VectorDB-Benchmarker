"""Milvus connector for VectorDB Benchmarker."""

import os
from typing import List, Dict, Any, Optional

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("milvus")
class MilvusConnector(BaseVectorDB):
    """Milvus vector database connector.
    
    Milvus is a cloud-native vector database designed for scalability.
    Supports both Milvus standalone and Zilliz Cloud.
    
    Config options:
        - uri: Milvus server URI (default: "http://localhost:19530")
        - token: Authentication token (for Zilliz Cloud)
        - db_name: Database name (default: "default")
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.uri = config.get("uri", os.getenv("MILVUS_URI", "http://localhost:19530"))
        self.token = config.get("token", os.getenv("MILVUS_TOKEN", ""))
        self.db_name = config.get("db_name", "default")
        self.client = None
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Milvus."""
        from pymilvus import MilvusClient
        
        self.client = MilvusClient(
            uri=self.uri,
            token=self.token,
            db_name=self.db_name,
        )
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self.client:
            self.client.close()
        self.client = None
        self._dimensions.clear()
        self._connected = False
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a Milvus collection."""
        # Drop if exists
        if self.client.has_collection(name):
            self.client.drop_collection(name)
        
        # Create collection with auto-generated schema
        self.client.create_collection(
            collection_name=name,
            dimension=dimension,
            metric_type="COSINE",
            auto_id=False,
            id_type="string",
            max_length=256,
        )
        
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete a Milvus collection."""
        try:
            if self.client.has_collection(name):
                self.client.drop_collection(name)
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
        """Add documents to Milvus collection."""
        data = []
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            record = {
                "id": doc_id,
                "vector": embedding,
            }
            if documents and i < len(documents):
                record["document"] = documents[i][:65535]  # Milvus varchar limit
            if metadatas and i < len(metadatas):
                # Flatten metadata into record
                for k, v in metadatas[i].items():
                    if isinstance(v, (str, int, float, bool)):
                        record[f"meta_{k}"] = v
            data.append(record)
        
        self.client.insert(collection_name=collection, data=data)
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        search_params = {
            "collection_name": collection,
            "data": [query_embedding],
            "limit": top_k,
            "output_fields": ["document"],
        }
        
        if filters:
            # Convert filters to Milvus filter expression
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, str):
                    filter_parts.append(f'meta_{k} == "{v}"')
                else:
                    filter_parts.append(f"meta_{k} == {v}")
            if filter_parts:
                search_params["filter"] = " and ".join(filter_parts)
        
        results = self.client.search(**search_params)
        
        search_results = []
        for hit in results[0]:
            search_results.append(SearchResult(
                id=hit["id"],
                score=hit["distance"],
                document=hit.get("entity", {}).get("document"),
                metadata=None,
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        stats = self.client.get_collection_stats(collection)
        return stats.get("row_count", 0)
