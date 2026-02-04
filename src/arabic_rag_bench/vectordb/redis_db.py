"""Redis 8 with vector search connector for Arabic RAG Benchmark."""

import os
from typing import List, Dict, Any, Optional
import json

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("redis")
class RedisVectorConnector(BaseVectorDB):
    """Redis 8 vector search connector.
    
    Redis 8 includes native vector search capabilities (RediSearch module).
    
    Config options:
        - host: Redis host (default: "localhost")
        - port: Redis port (default: 6379)
        - password: Redis password (or REDIS_PASSWORD env var)
        - ssl: Use SSL connection (default: False)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", os.getenv("REDIS_HOST", "localhost"))
        self.port = config.get("port", int(os.getenv("REDIS_PORT", "6379")))
        self.password = config.get("password", os.getenv("REDIS_PASSWORD"))
        self.ssl = config.get("ssl", False)
        self.client = None
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to Redis."""
        import redis
        
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            ssl=self.ssl,
            decode_responses=False,  # We need bytes for vectors
        )
        
        # Verify connection
        self.client.ping()
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            self.client.close()
        self.client = None
        self._dimensions.clear()
        self._connected = False
    
    def _index_name(self, collection: str) -> str:
        """Get Redis index name."""
        return f"idx:{collection}"
    
    def _key_prefix(self, collection: str) -> str:
        """Get key prefix for documents."""
        return f"doc:{collection}:"
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a Redis search index for vectors."""
        from redis.commands.search.field import VectorField, TextField, TagField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        index_name = self._index_name(name)
        prefix = self._key_prefix(name)
        
        # Drop existing index if exists
        try:
            self.client.ft(index_name).dropindex(delete_documents=True)
        except Exception:
            pass
        
        # Create index schema
        schema = [
            VectorField(
                "embedding",
                "FLAT",  # FLAT or HNSW
                {
                    "TYPE": "FLOAT32",
                    "DIM": dimension,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
            TextField("document"),
            TagField("doc_id"),
        ]
        
        # Create index
        definition = IndexDefinition(
            prefix=[prefix],
            index_type=IndexType.HASH,
        )
        
        self.client.ft(index_name).create_index(
            schema,
            definition=definition,
        )
        
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete a Redis search index."""
        try:
            index_name = self._index_name(name)
            self.client.ft(index_name).dropindex(delete_documents=True)
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
        """Add documents to Redis."""
        import numpy as np
        
        prefix = self._key_prefix(collection)
        
        pipeline = self.client.pipeline()
        
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            key = f"{prefix}{doc_id}"
            
            # Convert embedding to bytes
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            fields = {
                "embedding": embedding_bytes,
                "doc_id": doc_id,
            }
            
            if documents and i < len(documents):
                fields["document"] = documents[i]
            
            if metadatas and i < len(metadatas):
                fields["metadata"] = json.dumps(metadatas[i])
            
            pipeline.hset(key, mapping=fields)
        
        pipeline.execute()
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        import numpy as np
        from redis.commands.search.query import Query
        
        index_name = self._index_name(collection)
        
        # Convert query embedding to bytes
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Build query
        filter_str = "*"
        if filters:
            filter_parts = []
            for k, v in filters.items():
                filter_parts.append(f"@{k}:{{{v}}}")
            filter_str = " ".join(filter_parts)
        
        query = (
            Query(f"({filter_str})=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("doc_id", "document", "metadata", "score")
            .dialect(2)
        )
        
        results = self.client.ft(index_name).search(
            query,
            query_params={"vec": query_bytes}
        )
        
        search_results = []
        for doc in results.docs:
            metadata = {}
            if hasattr(doc, "metadata") and doc.metadata:
                try:
                    metadata = json.loads(doc.metadata)
                except Exception:
                    pass
            
            # Redis returns distance, convert to similarity
            score = 1 - float(doc.score) if hasattr(doc, "score") else 0
            
            search_results.append(SearchResult(
                id=doc.doc_id if hasattr(doc, "doc_id") else doc.id,
                score=score,
                document=doc.document if hasattr(doc, "document") else None,
                metadata=metadata,
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        try:
            index_name = self._index_name(collection)
            info = self.client.ft(index_name).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0
