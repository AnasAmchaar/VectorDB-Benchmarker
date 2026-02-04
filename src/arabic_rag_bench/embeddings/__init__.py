"""Embedding providers for Arabic RAG Benchmark."""

from .base import BaseEmbedding
from .registry import EmbeddingRegistry, register_embedding, get_embedding

__all__ = ["BaseEmbedding", "EmbeddingRegistry", "register_embedding", "get_embedding"]
