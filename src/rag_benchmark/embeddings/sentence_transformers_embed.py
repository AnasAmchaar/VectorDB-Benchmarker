"""Sentence Transformers embedding provider (local, no API needed)."""

from typing import List, Dict, Any

from .base import BaseEmbedding
from .registry import register_embedding


@register_embedding("sentence-transformers")
class SentenceTransformersEmbedding(BaseEmbedding):
    """Sentence Transformers embedding provider (local)."""
    
    # Multilingual-supporting models
    MODELS = {
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "LaBSE": 768,
        "distiluse-base-multilingual-cased-v1": 512,
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "paraphrase-multilingual-MiniLM-L12-v2")
        self._dimension = self.MODELS.get(self.model_name, 384)
        self._model = None
    
    def _get_model(self):
        """Get or load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Sentence Transformers."""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        model = self._get_model()
        embedding = model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
