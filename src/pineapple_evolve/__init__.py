"""
pineapple_evolve: LLM-driven evolutionary code generation.

This package implements an evolutionary system for generating and optimizing
circle packing algorithms using LLMs.
"""

from .config import (
    BudgetConfig,
    Config,
    EvaluationConfig,
    EvolutionConfig,
    ExperimentConfig,
    LLMConfig,
    config_from_dict,
    load_config,
    load_evaluator,
)
from .cost_tracker import CostSummary, CostTracker, TokenUsage
from .evolution_api import EvolutionAPI, GenerationSummary, TrialResult, TrialSelection
from .exceptions import (
    BudgetExceededError,
    CodeExtractionError,
    ConfigValidationError,
    EvaluationError,
    LLMEvolveError,
)
from .llm import (
    ROOT_LLM_SYSTEM_PROMPT_STATIC,
    ROOT_LLM_SYSTEM_PROMPT_DYNAMIC,
    LLMClient,
    LLMResponse,
    MockLLMClient,
    get_root_system_prompt,
)
from .logger import ExperimentLogger
from .repl import REPLEnvironment, REPLResult
from .root_llm import OrchestratorResult, RootLLMOrchestrator

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
    "TrialSelection",
    "GenerationSummary",
    # LLM
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "ROOT_LLM_SYSTEM_PROMPT_STATIC",
    "ROOT_LLM_SYSTEM_PROMPT_DYNAMIC",
    "get_root_system_prompt",
    # Root LLM Orchestrator
    "RootLLMOrchestrator",
    "OrchestratorResult",
]

__version__ = "0.1.0"
