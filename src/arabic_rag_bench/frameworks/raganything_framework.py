"""RAGAnything framework connector.

RAGAnything is a multimodal RAG framework that can process
text, images, audio, and video for comprehensive retrieval.

Supported VectorDBs: ChromaDB, FAISS (via internal configuration)
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional

from .base import BaseRAGFramework, RAGConfig, RAGQueryResult, VectorDBType
from .registry import register_framework


@register_framework("raganything")
class RAGAnythingFramework(BaseRAGFramework):
    """RAGAnything framework implementation.
    
    RAGAnything features:
    - Multimodal document processing (text, images, audio, video)
    - Automatic modality detection
    - Unified retrieval across modalities
    - Built on LightRAG's knowledge graph
    
    Note: For this benchmark, we focus on text-only processing.
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._rag = None
        self._doc_count = 0
    
    @property
    def name(self) -> str:
        return "RAGAnything"
    
    @property
    def supported_vectordbs(self) -> List[VectorDBType]:
        # RAGAnything uses LightRAG internally
        return [VectorDBType.CHROMADB]
    
    def _get_embedding_func(self):
        """Create embedding function for RAGAnything."""
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
        """Create LLM function."""
        from lightrag.llm.gemini import gemini_model_complete
        return gemini_model_complete
    
    def initialize(self) -> None:
        """Initialize RAGAnything."""
        try:
            from raganything import RAGAnything
        except ImportError:
            raise ImportError(
                "RAGAnything not installed. Install with: pip install raganything"
            )
        
        storage_dir = self.config.storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self._rag = RAGAnything(
            working_dir=storage_dir,
            llm_model_func=self._get_llm_func(),
            llm_model_name=self.config.llm_model,
            embedding_func=self._get_embedding_func(),
        )
        
        self._initialized = True
        print(f"✓ RAGAnything initialized with storage at {storage_dir}")
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add text documents to RAGAnything."""
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")
        
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            
            # RAGAnything has insert_text method for plain text
            asyncio.run(self._rag.insert_text(
                text=doc,
                doc_id=doc_id,
            ))
        
        self._doc_count += len(documents)
        return len(documents)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """Query RAGAnything."""
        if not self._initialized:
            raise RuntimeError("Framework not initialized.")
        
        k = top_k or self.config.top_k
        
        start_time = time.time()
        
        # RAGAnything query
        result = asyncio.run(self._rag.query(
            query=query,
            top_k=k,
        ))
        
        latency = (time.time() - start_time) * 1000
        
        # Extract answer and contexts
        answer = result if isinstance(result, str) else result.get("answer", "")
        contexts = result.get("contexts", []) if isinstance(result, dict) else []
        
        return RAGQueryResult(
            query=query,
            answer=answer,
            contexts=contexts,
            source_docs=[],
            scores=[],
            latency_ms=latency,
        )
    
    def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Retrieve contexts without generating answer."""
        if not self._initialized:
            raise RuntimeError("Framework not initialized.")
        
        k = top_k or self.config.top_k
        
        try:
            # Use underlying LightRAG's naive retrieval
            result = asyncio.run(self._rag.lightrag.aquery(
                query,
                param={"mode": "naive", "top_k": k}
            ))
            
            if isinstance(result, str):
                return [result]
            return result.get("contexts", [])[:k]
        except Exception:
            return []
    
    def clear(self) -> None:
        """Clear storage."""
        import shutil
        
        if os.path.exists(self.config.storage_dir):
            shutil.rmtree(self.config.storage_dir)
        
        self._doc_count = 0
        self._rag = None
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
            "storage_dir": self.config.storage_dir,
        })
        return stats
