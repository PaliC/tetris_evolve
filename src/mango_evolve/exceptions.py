"""
Custom exceptions for the mango_evolve project.

All exceptions inherit from MangoEvolveError for easy catching.
"""


class MangoEvolveError(Exception):
    """Base exception for all MangoEvolve errors."""

    pass


class BudgetExceededError(MangoEvolveError):
    """Raised when the cost budget is exceeded."""

    pass


class ConfigValidationError(MangoEvolveError):
    """Raised when configuration validation fails."""

    pass


class CodeExtractionError(MangoEvolveError):
    """Raised when code cannot be extracted from LLM response."""

    pass


class EvaluationError(MangoEvolveError):
    """Raised when program evaluation fails."""

    pass


class GenerationLimitError(MangoEvolveError):
    """Raised when attempting to exceed max_generations limit."""

    pass


class ChildrenLimitError(MangoEvolveError):
    """Raised when attempting to exceed max_children_per_generation limit."""

    pass


class CalibrationBudgetError(MangoEvolveError):
    """Raised when calibration call budget is exhausted for a model."""

    pass


class ContextOverflowError(MangoEvolveError):
    """Raised when root LLM context size exceeds safe limits."""

    pass
