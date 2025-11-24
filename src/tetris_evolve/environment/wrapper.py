"""
Generic environment wrapper for PufferLib/Gymnasium environments.

This module provides a wrapper that works with any environment through the
EnvironmentConfig interface, enabling environment-agnostic evolution.
"""

import time
from typing import Callable, Dict, Any, Tuple, Optional
import gymnasium
import numpy as np

from tetris_evolve.environment.base import EnvironmentConfig


class GenericEnvironmentWrapper:
    """
    Generic wrapper for PufferLib/Gymnasium environments.

    This wrapper provides a unified interface for running episodes and collecting
    metrics from any environment that has an EnvironmentConfig implementation.

    Example:
        >>> config = TetrisConfig()
        >>> with GenericEnvironmentWrapper(config) as env:
        ...     def player(obs, info):
        ...         return env.action_space.sample()
        ...     metrics = env.run_episode(player, seed=42)
        ...     print(f"Score: {metrics['total_reward']}")
    """

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize wrapper with environment configuration.

        Args:
            config: EnvironmentConfig instance defining the environment
        """
        self.config = config
        self.env = gymnasium.make(config.get_env_id(), **config.get_env_kwargs())

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Tuple of (observation, info dict)
        """
        if seed is not None:
            return self.env.reset(seed=seed)
        return self.env.reset()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)

    def run_episode(
        self,
        player: Callable[[np.ndarray, Dict], Any],
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a complete episode with given player function.

        Args:
            player: Callable that takes (observation, info) and returns action
            seed: Optional random seed for reproducibility
            max_steps: Maximum steps before truncation (None for unlimited)

        Returns:
            Dict containing both generic and environment-specific metrics:
                - total_reward: Sum of rewards over episode
                - episode_length: Number of steps taken
                - episode_time: Wall-clock time in seconds
                - truncated: Whether episode was truncated
                - <env_specific>: Metrics from config.extract_episode_metrics()

        Example:
            >>> def simple_player(obs, info):
            ...     return 0  # Always move left
            >>> metrics = env.run_episode(simple_player, seed=42, max_steps=1000)
        """
        start_time = time.time()

        # Reset environment
        obs, info = self.reset(seed=seed)

        # Episode tracking
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        # Run episode
        while not (terminated or truncated):
            # Get action from player
            action = player(obs, info)

            # Take step
            obs, reward, terminated, truncated, info = self.step(action)

            total_reward += reward
            steps += 1

            # Check max steps
            if max_steps is not None and steps >= max_steps:
                truncated = True
                break

        # Calculate episode time
        episode_time = time.time() - start_time

        # Build generic metrics
        metrics = {
            "total_reward": float(total_reward),
            "episode_length": steps,
            "episode_time": episode_time,
            "truncated": truncated,
        }

        # Extract environment-specific metrics
        env_metrics = self.config.extract_episode_metrics(info)
        metrics.update(env_metrics)

        return metrics

    def close(self) -> None:
        """Close the environment and clean up resources."""
        self.env.close()

    @property
    def observation_space(self):
        """Get the observation space of the environment."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Get the action space of the environment."""
        return self.env.action_space

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures environment is closed."""
        self.close()
        return False
