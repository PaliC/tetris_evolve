"""
tetris_evolve: LLM-driven evolutionary code generation.

This package implements an evolutionary system for generating and optimizing
circle packing algorithms using LLMs.
"""

from .exceptions import (
    LLMEvolveError,
    BudgetExceededError,
    ConfigValidationError,
    CodeExtractionError,
    EvaluationError,
)
from .config import (
    Config,
    ExperimentConfig,
    LLMConfig,
    EvolutionConfig,
    BudgetConfig,
    EvaluationConfig,
    load_config,
    config_from_dict,
)
from .cost_tracker import CostTracker, TokenUsage, CostSummary
from .logger import ExperimentLogger
from .repl import REPLEnvironment, REPLResult

__all__ = [
    # Exceptions
    "LLMEvolveError",
    "BudgetExceededError",
    "ConfigValidationError",
    "CodeExtractionError",
    "EvaluationError",
    # Config
    "Config",
    "ExperimentConfig",
    "LLMConfig",
    "EvolutionConfig",
    "BudgetConfig",
    "EvaluationConfig",
    "load_config",
    "config_from_dict",
    # Cost tracking
    "CostTracker",
    "TokenUsage",
    "CostSummary",
    # Logging
    "ExperimentLogger",
    # REPL
    "REPLEnvironment",
    "REPLResult",
]

__version__ = "0.1.0"
