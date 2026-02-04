"""Vector database connectors for Arabic RAG Benchmark.

Supported databases:
- ChromaDB: Lightweight, in-process vector store
- FAISS: Facebook AI Similarity Search
- Pinecone: Managed cloud vector database
- Milvus: Cloud-native vector database
- Weaviate: Open-source vector database with GraphQL
- Qdrant: High-performance vector search engine
- Vald: Distributed vector search engine
- Turbopuffer: Serverless vector database
- pgvector: PostgreSQL with vector extensions
- Redis: Redis 8 with vector search
- Elasticsearch: Distributed search with vector support
"""

from .base import BaseVectorDB, SearchResult
from .registry import VectorDBRegistry, register_vectordb, get_vectordb

# Import all connectors to register them
from . import chromadb  # noqa: F401
from . import faiss_db  # noqa: F401
from . import pinecone_db  # noqa: F401
from . import milvus_db  # noqa: F401
from . import weaviate_db  # noqa: F401
from . import qdrant_db  # noqa: F401
from . import vald_db  # noqa: F401
from . import turbopuffer_db  # noqa: F401
from . import pgvector_db  # noqa: F401
from . import redis_db  # noqa: F401
from . import elasticsearch_db  # noqa: F401

__all__ = [
    "BaseVectorDB",
    "SearchResult",
    "VectorDBRegistry",
    "register_vectordb",
    "get_vectordb",
]
