"""Registry for RAG framework implementations."""

from typing import Dict, Type, List, Callable
from .base import BaseRAGFramework, RAGConfig


# Global registry of framework implementations
_FRAMEWORK_REGISTRY: Dict[str, Type[BaseRAGFramework]] = {}


def register_framework(name: str) -> Callable:
    """Decorator to register a RAG framework implementation.
    
    Usage:
        @register_framework("lightrag")
        class LightRAGFramework(BaseRAGFramework):
            ...
    
    Args:
        name: Unique name for the framework
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseRAGFramework]) -> Type[BaseRAGFramework]:
        if name in _FRAMEWORK_REGISTRY:
            raise ValueError(f"Framework '{name}' is already registered")
        _FRAMEWORK_REGISTRY[name] = cls
        return cls
    return decorator


def get_framework(name: str, config: RAGConfig) -> BaseRAGFramework:
    """Get a RAG framework instance by name.
    
    Args:
        name: Name of the framework (e.g., 'lightrag', 'llamaindex')
        config: RAG configuration
        
    Returns:
        Initialized framework instance
        
    Raises:
        ValueError: If framework is not registered
    """
    if name not in _FRAMEWORK_REGISTRY:
        available = ", ".join(_FRAMEWORK_REGISTRY.keys())
        raise ValueError(
            f"Unknown framework: {name}. "
            f"Available: {available}"
        )
    
    framework_cls = _FRAMEWORK_REGISTRY[name]
    return framework_cls(config)


def list_frameworks() -> List[str]:
    """List all registered frameworks.
    
    Returns:
        List of framework names
    """
    return list(_FRAMEWORK_REGISTRY.keys())
