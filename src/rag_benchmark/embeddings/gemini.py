"""Google Gemini embedding provider."""

import os
from typing import List, Dict, Any

from .base import BaseEmbedding
from .registry import register_embedding


@register_embedding("gemini")
class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embedding provider."""
    
    MODELS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.model = config.get("model", "text-embedding-004")
        self._dimension = self.MODELS.get(self.model, 768)
        self._client = None
    
    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Gemini API."""
        client = self._get_client()
        embeddings = []
        
        for text in texts:
            result = client.models.embed_content(
                model=self.model,
                contents=text
            )
            embeddings.append(list(result.embeddings[0].values))
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        client = self._get_client()
        result = client.models.embed_content(
            model=self.model,
            contents=text
        )
        return list(result.embeddings[0].values)
