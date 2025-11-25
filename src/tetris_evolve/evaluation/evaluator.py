"""
Player evaluation framework.

This module provides functions to evaluate evolved players by running them
through multiple episodes and collecting metrics.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union
import traceback

import numpy as np

from ..environment.base import EnvironmentConfig


class Player(Protocol):
    """Protocol defining the player interface."""

    def select_action(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """Select an action given an observation."""
        ...

    def reset(self) -> None:
        """Reset player state for a new episode."""
        ...


class PlayerExecutionError(Exception):
    """Exception raised when player execution fails."""
    pass


@dataclass
class MetricStats:
    """Statistics for a single metric across episodes."""
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricStats":
        """Create MetricStats from a list of values."""
        if not values:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0)

        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a player across multiple episodes."""
    generic_metrics: Dict[str, MetricStats]
    env_metrics: Dict[str, MetricStats]
    code_errors: int
    episode_results: List[Dict[str, Any]]
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generic_metrics": {
                k: v.to_dict() for k, v in self.generic_metrics.items()
            },
            "env_metrics": {
                k: v.to_dict() for k, v in self.env_metrics.items()
            },
            "code_errors": self.code_errors,
            "episode_results": self.episode_results,
            "error_messages": self.error_messages,
        }

    def get_primary_score(self) -> float:
        """
        Compute a primary score for ranking players.

        The primary score combines multiple metrics into a single value
        for comparison and selection. Higher is better.

        Formula: env_score_mean + 0.1 * episode_length_mean - 100 * error_rate
        """
        # Get score mean (default to 0 if not available)
        score_mean = 0.0
        if "score" in self.env_metrics:
            score_mean = self.env_metrics["score"].mean

        # Get episode length mean
        length_mean = 0.0
        if "episode_length" in self.generic_metrics:
            length_mean = self.generic_metrics["episode_length"].mean

        # Calculate error penalty
        total_episodes = len(self.episode_results) + self.code_errors
        error_rate = self.code_errors / max(total_episodes, 1)

        # Combine into primary score
        return score_mean + 0.1 * length_mean - 100 * error_rate


def evaluate_player(
    player: Player,
    env_config: EnvironmentConfig,
    num_episodes: int = 10,
    max_steps_per_episode: int = 10000,
    parallel: bool = False,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """
    Evaluate a player by running it through multiple episodes.

    Args:
        player: Player object with select_action() and reset() methods
        env_config: Environment configuration
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        parallel: Whether to run episodes in parallel (not yet implemented)
        seed: Random seed for reproducibility

    Returns:
        EvaluationResult with aggregated metrics
    """
    if parallel:
        # For now, fall back to serial
        # TODO: Implement parallel evaluation
        pass

    return _evaluate_serial(
        player, env_config, num_episodes, max_steps_per_episode, seed
    )


def _evaluate_serial(
    player: Player,
    env_config: EnvironmentConfig,
    num_episodes: int,
    max_steps_per_episode: int,
    seed: Optional[int],
    max_consecutive_failures: int = 3,  # FIXED: Add early termination threshold
) -> EvaluationResult:
    """Run serial evaluation."""
    episode_results: List[Dict[str, Any]] = []
    code_errors = 0
    error_messages: List[str] = []

    # Collect raw values for aggregation
    total_rewards: List[float] = []
    episode_lengths: List[float] = []
    env_metric_values: Dict[str, List[float]] = {}

    # FIXED: Track consecutive failures for early termination
    consecutive_failures = 0

    # Create environment
    env = env_config.create_env()

    try:
        for episode_id in range(num_episodes):
            # FIXED: Early termination on consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                error_messages.append(
                    f"Stopping early after {max_consecutive_failures} consecutive failures"
                )
                break

            episode_failed = False
            try:
                # Reset player and environment
                if hasattr(player, 'reset'):
                    player.reset()

                episode_seed = None if seed is None else seed + episode_id
                obs, info = env.reset(seed=episode_seed)

                episode_reward = 0.0
                episode_length = 0
                terminated = False
                truncated = False
                last_info = info

                # Run episode
                while not (terminated or truncated) and episode_length < max_steps_per_episode:
                    try:
                        action = player.select_action(obs)
                    except Exception as e:
                        code_errors += 1
                        error_messages.append(f"Episode {episode_id}: {str(e)}")
                        episode_failed = True
                        break

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward)
                    episode_length += 1
                    last_info = info

                # Extract environment-specific metrics
                env_metrics = env_config.extract_episode_metrics(last_info)

                # Store episode result
                episode_result = {
                    "episode_id": episode_id,
                    "reward": episode_reward,
                    "length": episode_length,
                    "env_metrics": env_metrics,
                    "terminated": terminated,
                    "truncated": truncated,
                }
                episode_results.append(episode_result)

                # Collect for aggregation
                total_rewards.append(episode_reward)
                episode_lengths.append(float(episode_length))
                for k, v in env_metrics.items():
                    if k not in env_metric_values:
                        env_metric_values[k] = []
                    env_metric_values[k].append(v)

                # FIXED: Reset consecutive failure counter on success
                if not episode_failed:
                    consecutive_failures = 0

            except Exception as e:
                code_errors += 1
                error_messages.append(f"Episode {episode_id} failed: {str(e)}\n{traceback.format_exc()}")
                episode_failed = True

            # FIXED: Track consecutive failures
            if episode_failed:
                consecutive_failures += 1

    finally:
        env.close()

    # Aggregate metrics
    generic_metrics = {
        "total_reward": MetricStats.from_values(total_rewards),
        "episode_length": MetricStats.from_values(episode_lengths),
    }

    env_metrics_aggregated = {
        k: MetricStats.from_values(v) for k, v in env_metric_values.items()
    }

    return EvaluationResult(
        generic_metrics=generic_metrics,
        env_metrics=env_metrics_aggregated,
        code_errors=code_errors,
        episode_results=episode_results,
        error_messages=error_messages,
    )


def evaluate_player_from_code(
    code: str,
    env_config: EnvironmentConfig,
    num_episodes: int = 10,
    max_steps_per_episode: int = 10000,
    player_class_name: str = "TetrisPlayer",
    seed: Optional[int] = None,
) -> EvaluationResult:
    """
    Evaluate a player defined as a source code string.

    Args:
        code: Python source code defining a player class
        env_config: Environment configuration
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        player_class_name: Name of the player class in the code
        seed: Random seed for reproducibility

    Returns:
        EvaluationResult with metrics (code_errors > 0 if code is invalid)
    """
    # Try to compile and execute the code
    try:
        # Create a namespace for execution
        namespace: Dict[str, Any] = {"np": np, "numpy": np}

        # Execute the code
        exec(code, namespace)

        # Get the player class
        if player_class_name not in namespace:
            # Try to find any class with select_action method
            player_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and hasattr(obj, 'select_action'):
                    player_class = obj
                    break

            if player_class is None:
                return EvaluationResult(
                    generic_metrics={
                        "total_reward": MetricStats(0.0, 0.0, 0.0, 0.0),
                        "episode_length": MetricStats(0.0, 0.0, 0.0, 0.0),
                    },
                    env_metrics={},
                    code_errors=1,
                    episode_results=[],
                    error_messages=[f"No class named '{player_class_name}' or class with select_action method found"],
                )
        else:
            player_class = namespace[player_class_name]

        # Instantiate the player
        player = player_class()

        # Evaluate
        return evaluate_player(
            player, env_config, num_episodes, max_steps_per_episode, seed=seed
        )

    except SyntaxError as e:
        return EvaluationResult(
            generic_metrics={
                "total_reward": MetricStats(0.0, 0.0, 0.0, 0.0),
                "episode_length": MetricStats(0.0, 0.0, 0.0, 0.0),
            },
            env_metrics={},
            code_errors=1,
            episode_results=[],
            error_messages=[f"Syntax error in code: {str(e)}"],
        )
    except Exception as e:
        return EvaluationResult(
            generic_metrics={
                "total_reward": MetricStats(0.0, 0.0, 0.0, 0.0),
                "episode_length": MetricStats(0.0, 0.0, 0.0, 0.0),
            },
            env_metrics={},
            code_errors=1,
            episode_results=[],
            error_messages=[f"Error executing code: {str(e)}\n{traceback.format_exc()}"],
        )
