"""Weaviate connector for VectorDB Benchmarker."""

import os
from typing import List, Dict, Any, Optional

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("weaviate")
class WeaviateConnector(BaseVectorDB):
    """Weaviate vector database connector.
    
    Weaviate is an open-source vector database with GraphQL API.
    Supports both self-hosted and Weaviate Cloud.
    
    Config options:
        - url: Weaviate server URL (default: "http://localhost:8080")
        - api_key: API key for Weaviate Cloud
        - grpc_port: gRPC port (default: 50051)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", os.getenv("WEAVIATE_URL", "http://localhost:8080"))
        self.api_key = config.get("api_key", os.getenv("WEAVIATE_API_KEY"))
        self.grpc_port = config.get("grpc_port", 50051)
        self.client = None
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Weaviate."""
        import weaviate
        from weaviate.classes.init import Auth
        
        if self.api_key:
            # Weaviate Cloud connection
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
            )
        else:
            # Local connection
            # Parse host and port from URL
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080
            
            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=self.grpc_port,
            )
        
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self.client:
            self.client.close()
        self.client = None
        self._dimensions.clear()
        self._connected = False
    
    def _get_class_name(self, name: str) -> str:
        """Convert collection name to Weaviate class name (PascalCase, starts with uppercase)."""
        # Weaviate class names must start with uppercase
        return name.replace("-", "_").replace(".", "_").title().replace("_", "")
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a Weaviate class (collection)."""
        from weaviate.classes.config import Configure, Property, DataType
        
        class_name = self._get_class_name(name)
        
        # Delete if exists
        if self.client.collections.exists(class_name):
            self.client.collections.delete(class_name)
        
        # Create collection with vector config
        self.client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="document", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
        )
        
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete a Weaviate class."""
        try:
            class_name = self._get_class_name(name)
            if self.client.collections.exists(class_name):
                self.client.collections.delete(class_name)
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
        """Add documents to Weaviate."""
        import json
        
        class_name = self._get_class_name(collection)
        coll = self.client.collections.get(class_name)
        
        with coll.batch.dynamic() as batch:
            for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
                properties = {
                    "doc_id": doc_id,
                    "document": documents[i] if documents and i < len(documents) else "",
                    "metadata": json.dumps(metadatas[i]) if metadatas and i < len(metadatas) else "{}",
                }
                
                batch.add_object(
                    properties=properties,
                    vector=embedding,
                    uuid=None,  # Auto-generate UUID
                )
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        import json
        
        class_name = self._get_class_name(collection)
        coll = self.client.collections.get(class_name)
        
        results = coll.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=["distance"],
        )
        
        search_results = []
        for obj in results.objects:
            metadata = {}
            try:
                metadata = json.loads(obj.properties.get("metadata", "{}"))
            except Exception:
                pass
            
            search_results.append(SearchResult(
                id=obj.properties.get("doc_id", str(obj.uuid)),
                score=1 - (obj.metadata.distance or 0),  # Convert distance to similarity
                document=obj.properties.get("document"),
                metadata=metadata,
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        class_name = self._get_class_name(collection)
        coll = self.client.collections.get(class_name)
        result = coll.aggregate.over_all(total_count=True)
        return result.total_count or 0
