"""
Environment module: Generic environment wrapper and configuration for PufferLib/Gymnasium.

This module provides an abstract interface for different environments (Tetris, Atari, etc.)
and a generic wrapper for evaluation.
"""

from tetris_evolve.environment.base import EnvironmentConfig
from tetris_evolve.environment.wrapper import GenericEnvironmentWrapper
from tetris_evolve.environment.tetris import TetrisConfig

__all__ = [
    "EnvironmentConfig",
    "GenericEnvironmentWrapper",
    "TetrisConfig",
]
