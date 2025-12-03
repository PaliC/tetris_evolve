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
    load_evaluator,
)
from .cost_tracker import CostTracker, TokenUsage, CostSummary
from .logger import ExperimentLogger
from .repl import REPLEnvironment, REPLResult
from .evolution_api import EvolutionAPI, TrialResult, GenerationSummary
from .llm import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    ROOT_LLM_SYSTEM_PROMPT,
    get_root_system_prompt,
)

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
    "load_evaluator",
    # Cost tracking
    "CostTracker",
    "TokenUsage",
    "CostSummary",
    # Logging
    "ExperimentLogger",
    # REPL
    "REPLEnvironment",
    "REPLResult",
    # Evolution API
    "EvolutionAPI",
    "TrialResult",
    "GenerationSummary",
    # LLM
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "ROOT_LLM_SYSTEM_PROMPT",
    "get_root_system_prompt",
]

__version__ = "0.1.0"
