"""OpenAI embedding provider."""

import os
from typing import List, Dict, Any

from .base import BaseEmbedding
from .registry import register_embedding


@register_embedding("openai")
class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding provider."""
    
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model", "text-embedding-3-small")
        self._dimension = self.MODELS.get(self.model, 1536)
        self.batch_size = config.get("batch_size", 20)
        self._client = None
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI API."""
        client = self._get_client()
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch
            )
            for item in response.data:
                embeddings.append(item.embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding
