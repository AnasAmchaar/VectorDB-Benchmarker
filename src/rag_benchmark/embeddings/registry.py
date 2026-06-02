"""Registry for embedding providers."""

from typing import Dict, Type, Any
from .base import BaseEmbedding


class EmbeddingRegistry:
    """Registry for embedding implementations."""
    
    _registry: Dict[str, Type[BaseEmbedding]] = {}
    
    @classmethod
    def register(cls, name: str, embedding_class: Type[BaseEmbedding]) -> None:
        """Register an embedding implementation."""
        cls._registry[name.lower()] = embedding_class
    
    @classmethod
    def get(cls, name: str, config: Dict[str, Any]) -> BaseEmbedding:
        """Get an instance of a registered embedding provider."""
        name = name.lower()
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown embedding provider: '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](config)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered embedding providers."""
        return list(cls._registry.keys())


def register_embedding(name: str):
    """Decorator to register an embedding implementation."""
    def decorator(cls: Type[BaseEmbedding]) -> Type[BaseEmbedding]:
        EmbeddingRegistry.register(name, cls)
        return cls
    return decorator


def get_embedding(name: str, config: Dict[str, Any]) -> BaseEmbedding:
    """Get an embedding provider instance by name."""
    return EmbeddingRegistry.get(name, config)
