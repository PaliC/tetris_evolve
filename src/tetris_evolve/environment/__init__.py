"""
Environment module: Generic environment wrapper and configuration for PufferLib/Gymnasium.

This module provides an abstract interface for different environments (Tetris, Atari, etc.)
and a generic wrapper for evaluation.

Usage:
    from tetris_evolve.environment import TetrisConfig, GenericEnvironmentWrapper

    # Create a Tetris environment
    config = TetrisConfig()
    wrapper = GenericEnvironmentWrapper(config)

    # Or use the factory
    config = create_environment_config("tetris")
"""
from typing import Dict, List, Type

from .base import EnvironmentConfig
from .wrapper import GenericEnvironmentWrapper
from .tetris import TetrisConfig

# Registry of available environment configurations
_ENVIRONMENT_REGISTRY: Dict[str, Type[EnvironmentConfig]] = {
    "tetris": TetrisConfig,
}


def get_available_environments() -> List[str]:
    """
    Get a list of available environment configuration names.

    Returns:
        List of environment names that can be passed to create_environment_config()
    """
    return list(_ENVIRONMENT_REGISTRY.keys())


def create_environment_config(name: str, **kwargs) -> EnvironmentConfig:
    """
    Create an environment configuration by name.

    Args:
        name: Environment name (e.g., "tetris", "atari")
        **kwargs: Additional arguments passed to the config constructor

    Returns:
        An EnvironmentConfig instance

    Raises:
        ValueError: If the environment name is not recognized
    """
    name_lower = name.lower()
    if name_lower not in _ENVIRONMENT_REGISTRY:
        available = ", ".join(get_available_environments())
        raise ValueError(
            f"Unknown environment: '{name}'. Available environments: {available}"
        )

    config_class = _ENVIRONMENT_REGISTRY[name_lower]
    return config_class(**kwargs)


def register_environment(name: str, config_class: Type[EnvironmentConfig]) -> None:
    """
    Register a new environment configuration.

    Args:
        name: Name to register the environment under
        config_class: EnvironmentConfig subclass
    """
    if not issubclass(config_class, EnvironmentConfig):
        raise TypeError(
            f"config_class must be a subclass of EnvironmentConfig, got {type(config_class)}"
        )
    _ENVIRONMENT_REGISTRY[name.lower()] = config_class


__all__ = [
    # Base classes
    "EnvironmentConfig",
    "GenericEnvironmentWrapper",
    # Implementations
    "TetrisConfig",
    # Registry functions
    "get_available_environments",
    "create_environment_config",
    "register_environment",
]
