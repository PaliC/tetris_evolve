"""
Evaluation modules for tetris_evolve.
"""

from .circle_packing import (
    DEFAULT_N_CIRCLES,
    CirclePackingEvaluator,
    PackingResult,
    evaluate_code,
    validate_packing,
)

__all__ = [
    "CirclePackingEvaluator",
    "PackingResult",
    "validate_packing",
    "evaluate_code",
    "DEFAULT_N_CIRCLES",
]
