"""
Evaluation module: Player evaluation framework for evolution.

This module provides functions to evaluate evolved players by running them
through multiple episodes and collecting metrics.

Usage:
    from tetris_evolve.evaluation import evaluate_player, EvaluationResult

    result = evaluate_player(player, env_config, num_episodes=100)
    print(f"Mean score: {result.env_metrics['score'].mean}")
"""
from .evaluator import (
    evaluate_player,
    evaluate_player_from_code,
    EvaluationResult,
    MetricStats,
    PlayerExecutionError,
    Player,
)
from .parallel import (
    ParallelEvaluator,
    EvaluationCache,
)
from .diversity import (
    compute_code_diversity,
    select_diverse,
    code_similarity,
    compute_population_stats,
)

__all__ = [
    # Core evaluation
    "evaluate_player",
    "evaluate_player_from_code",
    "EvaluationResult",
    "MetricStats",
    "PlayerExecutionError",
    "Player",
    # Parallel evaluation
    "ParallelEvaluator",
    "EvaluationCache",
    # Diversity
    "compute_code_diversity",
    "select_diverse",
    "code_similarity",
    "compute_population_stats",
]
