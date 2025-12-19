"""
Custom exceptions for the pineapple_evolve project.

All exceptions inherit from LLMEvolveError for easy catching.
"""


class LLMEvolveError(Exception):
    """Base exception for all LLM-Evolve errors."""

    pass


class BudgetExceededError(LLMEvolveError):
    """Raised when the cost budget is exceeded."""

    pass


class ConfigValidationError(LLMEvolveError):
    """Raised when configuration validation fails."""

    pass


class CodeExtractionError(LLMEvolveError):
    """Raised when code cannot be extracted from LLM response."""

    pass


class EvaluationError(LLMEvolveError):
    """Raised when program evaluation fails."""

    pass


class GenerationLimitError(LLMEvolveError):
    """Raised when attempting to exceed max_generations limit."""

    pass


class ChildrenLimitError(LLMEvolveError):
    """Raised when attempting to exceed max_children_per_generation limit."""

    pass
