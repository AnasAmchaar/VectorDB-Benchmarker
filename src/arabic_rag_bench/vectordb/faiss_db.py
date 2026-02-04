"""FAISS connector for Arabic RAG Benchmark."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("faiss")
class FAISSConnector(BaseVectorDB):
    """FAISS vector database connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index_path = Path(config.get("index_path", "./data/faiss"))
        self._indexes: Dict[str, Any] = {}
        self._documents: Dict[str, Dict[str, Any]] = {}  # collection -> {id: {doc, metadata}}
        self._id_maps: Dict[str, List[str]] = {}  # collection -> [id1, id2, ...]
    
    def connect(self) -> None:
        """Initialize FAISS."""
        import faiss  # noqa: F401 - verify installation
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._connected = True
    
    def disconnect(self) -> None:
        """Clean up FAISS indexes."""
        self._indexes.clear()
        self._documents.clear()
        self._id_maps.clear()
        self._connected = False
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new FAISS index."""
        import faiss
        
        # Use IndexFlatIP for cosine similarity (normalize vectors first)
        index = faiss.IndexFlatIP(dimension)
        self._indexes[name] = index
        self._documents[name] = {}
        self._id_maps[name] = []
    
    def delete_collection(self, name: str) -> None:
        """Delete a FAISS index."""
        self._indexes.pop(name, None)
        self._documents.pop(name, None)
        self._id_maps.pop(name, None)
        
        # Remove persisted files
        index_file = self.index_path / f"{name}.index"
        meta_file = self.index_path / f"{name}.meta.json"
        if index_file.exists():
            index_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
    
    def add_documents(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to FAISS index."""
        if collection not in self._indexes:
            raise ValueError(f"Collection '{collection}' does not exist")
        
        index = self._indexes[collection]
        
        # Normalize embeddings for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss_module = __import__("faiss")
        faiss_module.normalize_L2(vectors)
        
        # Add to index
        index.add(vectors)
        
        # Store documents and metadata
        for i, doc_id in enumerate(ids):
            self._id_maps[collection].append(doc_id)
            self._documents[collection][doc_id] = {
                "document": documents[i] if documents else None,
                "metadata": metadatas[i] if metadatas else None
            }
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if collection not in self._indexes:
            raise ValueError(f"Collection '{collection}' does not exist")
        
        index = self._indexes[collection]
        faiss_module = __import__("faiss")
        
        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        faiss_module.normalize_L2(query)
        
        # Search
        scores, indices = index.search(query, min(top_k, index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
                
            doc_id = self._id_maps[collection][idx]
            doc_data = self._documents[collection].get(doc_id, {})
            
            # Apply filters if provided
            if filters and doc_data.get("metadata"):
                match = all(
                    doc_data["metadata"].get(k) == v 
                    for k, v in filters.items()
                )
                if not match:
                    continue
            
            results.append(SearchResult(
                id=doc_id,
                score=float(score),
                document=doc_data.get("document"),
                metadata=doc_data.get("metadata")
            ))
        
        return results
    
    def count(self, collection: str) -> int:
        """Count documents in index."""
        if collection not in self._indexes:
            return 0
        return self._indexes[collection].ntotal
    
    def save(self, collection: str) -> None:
        """Save index to disk."""
        import faiss
        
        if collection not in self._indexes:
            return
            
        index_file = self.index_path / f"{collection}.index"
        meta_file = self.index_path / f"{collection}.meta.json"
        
        faiss.write_index(self._indexes[collection], str(index_file))
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "id_map": self._id_maps[collection],
                "documents": self._documents[collection]
            }, f, ensure_ascii=False)
    
    def load(self, collection: str) -> None:
        """Load index from disk."""
        import faiss
        
        index_file = self.index_path / f"{collection}.index"
        meta_file = self.index_path / f"{collection}.meta.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        self._indexes[collection] = faiss.read_index(str(index_file))
        
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self._id_maps[collection] = meta["id_map"]
            self._documents[collection] = meta["documents"]
