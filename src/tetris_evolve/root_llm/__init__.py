"""
Root LLM module: Root LLM logic and orchestration.

This module implements the Root LLM (Depth=0) that acts as the autonomous
evolutionary strategist, deciding what programs to evolve and how.

Usage:
    from tetris_evolve.root_llm import RootLLMFunctions

    functions = RootLLMFunctions(db, env_config)
    functions.inject_into_repl(repl)
"""
from .functions import (
    RootLLMFunctions,
    PerformanceAnalysis,
    HistoricalTrends,
    ResourceLimitError,
)

__all__ = [
    "RootLLMFunctions",
    "PerformanceAnalysis",
    "HistoricalTrends",
    "ResourceLimitError",
]
