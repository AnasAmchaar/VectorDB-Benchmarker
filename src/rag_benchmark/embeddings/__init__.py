"""Embedding providers for VectorDB Benchmarker.

Supported providers:
- Gemini: Google's text-embedding models
- OpenAI: OpenAI embedding models
- Sentence Transformers: Local HuggingFace models (no API key needed)
"""

from .base import BaseEmbedding
from .registry import EmbeddingRegistry, register_embedding, get_embedding

# Import all providers to register them
from . import gemini  # noqa: F401
from . import openai_embed  # noqa: F401
from . import sentence_transformers_embed  # noqa: F401

__all__ = [
    "BaseEmbedding",
    "EmbeddingRegistry",
    "register_embedding",
    "get_embedding",
]
