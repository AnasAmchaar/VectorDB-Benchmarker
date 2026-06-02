"""LlamaIndex RAG framework connector.

LlamaIndex is a popular data framework for LLM applications.
It provides a simple interface for RAG with many vectorDB integrations.

Supported VectorDBs: ChromaDB, FAISS, Pinecone, Qdrant, Milvus, Weaviate
"""

import os
import time
from typing import List, Dict, Any, Optional

from .base import BaseRAGFramework, RAGConfig, RAGQueryResult, VectorDBType
from .registry import register_framework


@register_framework("llamaindex")
class LlamaIndexFramework(BaseRAGFramework):
    """LlamaIndex RAG framework implementation.
    
    LlamaIndex offers:
    - Simple document ingestion with automatic chunking
    - Multiple vectorDB backends
    - Flexible query engines
    - Built-in evaluation tools
    """
    
    VECTORDB_MAPPING = {
        VectorDBType.CHROMADB: "chromadb",
        VectorDBType.FAISS: "faiss", 
        VectorDBType.PINECONE: "pinecone",
        VectorDBType.QDRANT: "qdrant",
        VectorDBType.MILVUS: "milvus",
        VectorDBType.WEAVIATE: "weaviate",
    }
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._index = None
        self._query_engine = None
        self._retriever = None
        self._embed_model = None
        self._llm = None
        self._vector_store = None
        self._storage_context = None
        self._doc_count = 0
    
    @property
    def name(self) -> str:
        return "LlamaIndex"
    
    @property
    def supported_vectordbs(self) -> List[VectorDBType]:
        return list(self.VECTORDB_MAPPING.keys())
    
    def _setup_embedding_model(self):
        """Set up the embedding model based on config."""
        from llama_index.core import Settings
        
        if self.config.embedding_provider == "gemini":
            from llama_index.embeddings.gemini import GeminiEmbedding
            self._embed_model = GeminiEmbedding(
                model_name=f"models/{self.config.embedding_model}",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            )
        elif self.config.embedding_provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            self._embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.config.embedding_provider == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self._embed_model = HuggingFaceEmbedding(
                model_name=self.config.embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
        
        Settings.embed_model = self._embed_model
    
    def _setup_llm(self):
        """Set up the LLM based on config."""
        from llama_index.core import Settings
        
        if self.config.llm_provider == "gemini":
            from llama_index.llms.gemini import Gemini
            self._llm = Gemini(
                model=f"models/{self.config.llm_model}",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            )
        elif self.config.llm_provider == "openai":
            from llama_index.llms.openai import OpenAI
            self._llm = OpenAI(
                model=self.config.llm_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.config.llm_provider == "ollama":
            from llama_index.llms.ollama import Ollama
            self._llm = Ollama(
                model=self.config.llm_model,
                base_url=self.config.extra_config.get("ollama_url", "http://localhost:11434"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        Settings.llm = self._llm
    
    def _setup_vector_store(self):
        """Set up the vector store based on config."""
        from llama_index.core import StorageContext
        
        vdb_type = self.config.vectordb_type
        vdb_config = self.config.vectordb_config
        
        if vdb_type == VectorDBType.CHROMADB:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
            
            persist_dir = vdb_config.get("persist_dir", f"{self.config.storage_dir}/chroma")
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._vector_store = ChromaVectorStore(chroma_collection=collection)
            
        elif vdb_type == VectorDBType.FAISS:
            import faiss
            from llama_index.vector_stores.faiss import FaissVectorStore
            
            d = self.config.embedding_dimension
            faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
            self._vector_store = FaissVectorStore(faiss_index=faiss_index)
            
        elif vdb_type == VectorDBType.PINECONE:
            from pinecone import Pinecone
            from llama_index.vector_stores.pinecone import PineconeVectorStore
            
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = vdb_config.get("index_name", self.config.collection_name)
            
            # Create index if it doesn't exist
            if index_name not in [idx.name for idx in pc.list_indexes()]:
                pc.create_index(
                    name=index_name,
                    dimension=self.config.embedding_dimension,
                    metric="cosine",
                    spec=vdb_config.get("spec", {"serverless": {"cloud": "aws", "region": "us-east-1"}})
                )
            
            index = pc.Index(index_name)
            self._vector_store = PineconeVectorStore(pinecone_index=index)
            
        elif vdb_type == VectorDBType.QDRANT:
            from qdrant_client import QdrantClient
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            
            url = vdb_config.get("url", "http://localhost:6333")
            client = QdrantClient(url=url)
            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
            )
            
        elif vdb_type == VectorDBType.MILVUS:
            from llama_index.vector_stores.milvus import MilvusVectorStore
            
            self._vector_store = MilvusVectorStore(
                uri=vdb_config.get("uri", "http://localhost:19530"),
                collection_name=self.config.collection_name,
                dim=self.config.embedding_dimension,
            )
            
        elif vdb_type == VectorDBType.WEAVIATE:
            import weaviate
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            
            client = weaviate.Client(url=vdb_config.get("url", "http://localhost:8080"))
            self._vector_store = WeaviateVectorStore(
                weaviate_client=client,
                index_name=self.config.collection_name,
            )
        else:
            raise ValueError(f"Unsupported vectordb: {vdb_type}")
        
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
    
    def initialize(self) -> None:
        """Initialize LlamaIndex with configured components."""
        from llama_index.core import Settings
        
        # Set chunk size
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap
        
        self._setup_embedding_model()
        self._setup_llm()
        self._setup_vector_store()
        
        self._initialized = True
        print(f"✓ LlamaIndex initialized with {self.config.vectordb_type.value}")
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add documents to the index."""
        from llama_index.core import Document, VectorStoreIndex
        
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")
        
        # Create LlamaIndex documents
        llama_docs = []
        for i, text in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            metadata = metadatas[i] if metadatas else {}
            metadata["doc_id"] = doc_id
            
            llama_docs.append(Document(
                text=text,
                doc_id=doc_id,
                metadata=metadata,
            ))
        
        # Create or update index
        if self._index is None:
            self._index = VectorStoreIndex.from_documents(
                llama_docs,
                storage_context=self._storage_context,
                show_progress=True,
            )
        else:
            for doc in llama_docs:
                self._index.insert(doc)
        
        # Create query engine and retriever
        self._query_engine = self._index.as_query_engine(
            similarity_top_k=self.config.top_k
        )
        self._retriever = self._index.as_retriever(
            similarity_top_k=self.config.top_k
        )
        
        self._doc_count += len(documents)
        return len(llama_docs)  # Approximate chunk count
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """Query the RAG system."""
        if not self._initialized or self._query_engine is None:
            raise RuntimeError("Framework not initialized or no documents added.")
        
        k = top_k or self.config.top_k
        
        # Update retriever if top_k changed
        if k != self.config.top_k:
            self._query_engine = self._index.as_query_engine(similarity_top_k=k)
        
        start_time = time.time()
        response = self._query_engine.query(query)
        latency = (time.time() - start_time) * 1000
        
        # Extract contexts from source nodes
        contexts = []
        source_docs = []
        scores = []
        
        for node in response.source_nodes:
            contexts.append(node.text)
            source_docs.append(node.node_id)
            scores.append(node.score if node.score else 0.0)
        
        return RAGQueryResult(
            query=query,
            answer=str(response),
            contexts=contexts,
            source_docs=source_docs,
            scores=scores,
            latency_ms=latency,
        )
    
    def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Retrieve contexts without generating an answer."""
        if not self._initialized or self._retriever is None:
            raise RuntimeError("Framework not initialized or no documents added.")
        
        k = top_k or self.config.top_k
        
        # Update retriever if top_k changed
        if k != self.config.top_k:
            self._retriever = self._index.as_retriever(similarity_top_k=k)
        
        nodes = self._retriever.retrieve(query)
        return [node.text for node in nodes]
    
    def clear(self) -> None:
        """Clear all documents."""
        # Reset index
        self._index = None
        self._query_engine = None
        self._retriever = None
        self._doc_count = 0
        
        # Reinitialize vector store
        self._setup_vector_store()
    
    def close(self) -> None:
        """Clean up resources."""
        self._index = None
        self._query_engine = None
        self._retriever = None
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get framework statistics."""
        stats = super().get_stats()
        stats.update({
            "document_count": self._doc_count,
            "has_index": self._index is not None,
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
        })
        return stats
