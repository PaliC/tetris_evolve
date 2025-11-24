"""
Evolution module: Main evolution loop and orchestration.

This module implements the top-level evolution loop that coordinates the Root LLM,
Child RLLMs, evaluation, and persistence.

Usage:
    from tetris_evolve.evolution import EvolutionRunner, EvolutionConfig

    config = EvolutionConfig(max_generations=50)
    runner = EvolutionRunner(config, env_config, llm_client, run_dir)
    result = runner.run()
"""
from .loop import (
    EvolutionRunner,
    EvolutionConfig,
    EvolutionLogger,
    GenerationSummary,
)

__all__ = [
    "EvolutionRunner",
    "EvolutionConfig",
    "EvolutionLogger",
    "GenerationSummary",
]
