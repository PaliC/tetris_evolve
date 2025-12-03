"""
Evaluation modules for tetris_evolve.
"""

from .circle_packing import (
    CirclePackingEvaluator,
    PackingResult,
    validate_packing,
    evaluate_code,
    DEFAULT_TARGET,
    DEFAULT_N_CIRCLES,
)

__all__ = [
    "CirclePackingEvaluator",
    "PackingResult",
    "validate_packing",
    "evaluate_code",
    "DEFAULT_TARGET",
    "DEFAULT_N_CIRCLES",
]
