"""LightRAG framework connector.

LightRAG is a knowledge graph-based RAG framework that builds
entity-relationship graphs for enhanced retrieval.

Supported VectorDBs: Built-in nano-vectordb (ChromaDB-like)
Note: LightRAG uses its own internal vector storage by default.
"""

import os
import time
import asyncio
import shutil
from typing import List, Dict, Any, Optional

from .base import BaseRAGFramework, RAGConfig, RAGQueryResult, VectorDBType
from .registry import register_framework


@register_framework("lightrag")
class LightRAGFramework(BaseRAGFramework):
    """LightRAG framework implementation.
    
    LightRAG features:
    - Knowledge graph-based entity extraction
    - Multiple query modes: naive, local, global, hybrid
    - Graph-enhanced retrieval
    
    Note: LightRAG uses its own internal vector storage (nano-vectordb).
    VectorDB configuration is limited to storage path.
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._rag = None
        self._doc_count = 0
        self._query_mode = config.extra_config.get("query_mode", "hybrid")
    
    @property
    def name(self) -> str:
        return "LightRAG"
    
    @property
    def supported_vectordbs(self) -> List[VectorDBType]:
        # LightRAG uses its own internal vector storage
        return [VectorDBType.CHROMADB]  # Conceptually similar
    
    def _get_embedding_func(self):
        """Create embedding function for LightRAG."""
        from lightrag.llm.gemini import gemini_embed
        from lightrag.utils import EmbeddingFunc
        import numpy as np
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        async def embedding_wrapper(texts: List[str]) -> np.ndarray:
            """Wrapper that returns proper numpy array format."""
            from google import genai
            
            client = genai.Client(api_key=api_key)
            embeddings = []
            
            for text in texts:
                result = client.models.embed_content(
                    model=self.config.embedding_model,
                    contents=text
                )
                embeddings.append(result.embeddings[0].values)
            
            return np.array(embeddings)
        
        return EmbeddingFunc(
            embedding_dim=self.config.embedding_dimension,
            max_token_size=8192,
            func=embedding_wrapper,
        )
    
    def _get_llm_func(self):
        """Create LLM function for LightRAG."""
        from lightrag.llm.gemini import gemini_model_complete
        
        return gemini_model_complete
    
    def initialize(self) -> None:
        """Initialize LightRAG."""
        from lightrag import LightRAG, QueryParam
        
        storage_dir = self.config.storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self._rag = LightRAG(
            working_dir=storage_dir,
            llm_model_func=self._get_llm_func(),
            llm_model_name=self.config.llm_model,
            embedding_func=self._get_embedding_func(),
        )
        
        self._initialized = True
        print(f"✓ LightRAG initialized with storage at {storage_dir}")
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add documents to LightRAG."""
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")
        
        # LightRAG expects documents as single string or list
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            
            # Run async insert
            asyncio.run(self._rag.ainsert(
                doc,
                ids=[doc_id],
            ))
        
        self._doc_count += len(documents)
        return len(documents)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """Query LightRAG."""
        from lightrag import QueryParam
        
        if not self._initialized:
            raise RuntimeError("Framework not initialized.")
        
        k = top_k or self.config.top_k
        
        start_time = time.time()
        
        # Query with specified mode
        param = QueryParam(
            mode=self._query_mode,
            top_k=k,
        )
        
        result = asyncio.run(self._rag.aquery(query, param=param))
        
        latency = (time.time() - start_time) * 1000
        
        # Extract contexts - LightRAG embeds context in the response
        # We need to retrieve separately for evaluation
        contexts = self.retrieve_only(query, top_k=k)
        
        return RAGQueryResult(
            query=query,
            answer=result,
            contexts=contexts,
            source_docs=[],
            scores=[],
            latency_ms=latency,
            metadata={"mode": self._query_mode}
        )
    
    def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Retrieve contexts using naive mode (pure vector search)."""
        from lightrag import QueryParam
        
        if not self._initialized:
            raise RuntimeError("Framework not initialized.")
        
        k = top_k or self.config.top_k
        
        # Use naive mode for pure retrieval
        param = QueryParam(mode="naive", top_k=k)
        
        try:
            result = asyncio.run(self._rag.aquery(query, param=param))
            
            # Parse contexts from the response
            # LightRAG returns context within the answer
            if "Sources:" in result:
                contexts = result.split("Sources:")[1].strip().split("\n\n")
            else:
                contexts = [result]
            
            return contexts[:k]
        except Exception:
            return []
    
    def clear(self) -> None:
        """Clear LightRAG storage."""
        if os.path.exists(self.config.storage_dir):
            shutil.rmtree(self.config.storage_dir)
        
        self._doc_count = 0
        self._rag = None
        
        # Reinitialize
        self.initialize()
    
    def close(self) -> None:
        """Clean up resources."""
        self._rag = None
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get framework statistics."""
        stats = super().get_stats()
        stats.update({
            "document_count": self._doc_count,
            "query_mode": self._query_mode,
            "storage_dir": self.config.storage_dir,
        })
        return stats
