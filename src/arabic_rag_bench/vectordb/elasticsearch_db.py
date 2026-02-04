"""Elasticsearch connector for Arabic RAG Benchmark."""

import os
from typing import List, Dict, Any, Optional

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("elasticsearch")
class ElasticsearchConnector(BaseVectorDB):
    """Elasticsearch vector database connector.
    
    Elasticsearch supports dense vector fields and kNN search.
    
    Config options:
        - hosts: List of Elasticsearch hosts (default: ["http://localhost:9200"])
        - api_key: API key for Elastic Cloud
        - cloud_id: Cloud ID for Elastic Cloud
        - username: Basic auth username
        - password: Basic auth password
        - verify_certs: Verify SSL certificates (default: True)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hosts = config.get("hosts", [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")])
        self.api_key = config.get("api_key", os.getenv("ELASTICSEARCH_API_KEY"))
        self.cloud_id = config.get("cloud_id", os.getenv("ELASTIC_CLOUD_ID"))
        self.username = config.get("username", os.getenv("ELASTICSEARCH_USERNAME"))
        self.password = config.get("password", os.getenv("ELASTICSEARCH_PASSWORD"))
        self.verify_certs = config.get("verify_certs", True)
        self.client = None
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Elasticsearch."""
        from elasticsearch import Elasticsearch
        
        if self.cloud_id:
            # Elastic Cloud connection
            self.client = Elasticsearch(
                cloud_id=self.cloud_id,
                api_key=self.api_key,
            )
        elif self.api_key:
            # API key auth
            self.client = Elasticsearch(
                hosts=self.hosts,
                api_key=self.api_key,
                verify_certs=self.verify_certs,
            )
        elif self.username and self.password:
            # Basic auth
            self.client = Elasticsearch(
                hosts=self.hosts,
                basic_auth=(self.username, self.password),
                verify_certs=self.verify_certs,
            )
        else:
            # No auth
            self.client = Elasticsearch(
                hosts=self.hosts,
                verify_certs=self.verify_certs,
            )
        
        # Verify connection
        self.client.info()
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Elasticsearch."""
        if self.client:
            self.client.close()
        self.client = None
        self._dimensions.clear()
        self._connected = False
    
    def _index_name(self, collection: str) -> str:
        """Get Elasticsearch index name (lowercase, no special chars)."""
        return collection.lower().replace(" ", "_").replace("-", "_")
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create an Elasticsearch index with vector mapping."""
        index_name = self._index_name(name)
        
        # Delete if exists
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
        
        # Create index with dense_vector mapping
        mappings = {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine",
                },
                "document": {
                    "type": "text",
                    "analyzer": "arabic",  # Arabic analyzer for better tokenization
                },
                "metadata": {
                    "type": "object",
                    "enabled": True,
                },
            }
        }
        
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
        
        self.client.indices.create(
            index=index_name,
            mappings=mappings,
            settings=settings,
        )
        
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete an Elasticsearch index."""
        try:
            index_name = self._index_name(name)
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
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
        """Add documents to Elasticsearch."""
        from elasticsearch.helpers import bulk
        
        index_name = self._index_name(collection)
        
        actions = []
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            doc = {
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "embedding": embedding,
                }
            }
            
            if documents and i < len(documents):
                doc["_source"]["document"] = documents[i]
            
            if metadatas and i < len(metadatas):
                doc["_source"]["metadata"] = metadatas[i]
            
            actions.append(doc)
        
        # Bulk index
        bulk(self.client, actions, refresh=True)
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents using kNN."""
        index_name = self._index_name(collection)
        
        # Build kNN query
        knn = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10,
        }
        
        # Add filter if provided
        if filters:
            filter_clauses = []
            for k, v in filters.items():
                filter_clauses.append({"term": {f"metadata.{k}": v}})
            knn["filter"] = {"bool": {"must": filter_clauses}}
        
        response = self.client.search(
            index=index_name,
            knn=knn,
            source=["document", "metadata"],
        )
        
        search_results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            search_results.append(SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                document=source.get("document"),
                metadata=source.get("metadata"),
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        try:
            index_name = self._index_name(collection)
            response = self.client.count(index=index_name)
            return response["count"]
        except Exception:
            return 0
