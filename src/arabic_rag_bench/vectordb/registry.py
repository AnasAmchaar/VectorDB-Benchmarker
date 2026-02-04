"""Registry for vector database connectors."""

from typing import Dict, Type, Any
from .base import BaseVectorDB


class VectorDBRegistry:
    """Registry for vector database implementations."""
    
    _registry: Dict[str, Type[BaseVectorDB]] = {}
    
    @classmethod
    def register(cls, name: str, db_class: Type[BaseVectorDB]) -> None:
        """Register a vector database implementation.
        
        Args:
            name: Name to register the database under
            db_class: The database class to register
        """
        cls._registry[name.lower()] = db_class
    
    @classmethod
    def get(cls, name: str, config: Dict[str, Any]) -> BaseVectorDB:
        """Get an instance of a registered vector database.
        
        Args:
            name: Name of the registered database
            config: Configuration for the database
            
        Returns:
            Instance of the vector database
            
        Raises:
            ValueError: If database name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown vector database: '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](config)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered vector databases."""
        return list(cls._registry.keys())


def register_vectordb(name: str):
    """Decorator to register a vector database implementation.
    
    Usage:
        @register_vectordb("mydb")
        class MyVectorDB(BaseVectorDB):
            ...
    """
    def decorator(cls: Type[BaseVectorDB]) -> Type[BaseVectorDB]:
        VectorDBRegistry.register(name, cls)
        return cls
    return decorator


def get_vectordb(name: str, config: Dict[str, Any]) -> BaseVectorDB:
    """Get a vector database instance by name.
    
    Args:
        name: Name of the vector database
        config: Configuration dictionary
        
    Returns:
        Instance of the vector database
    """
    return VectorDBRegistry.get(name, config)
