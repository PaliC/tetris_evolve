"""
Generic environment wrapper for any PufferLib/Gymnasium environment.

This wrapper provides a unified interface for the evolution system to
interact with different environments through their EnvironmentConfig.
"""
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .base import EnvironmentConfig


class GenericEnvironmentWrapper:
    """
    Generic wrapper around any PufferLib/Gymnasium environment.

    Works with Tetris, Atari, custom environments, etc., through the
    EnvironmentConfig abstraction.

    Attributes:
        env_config: The environment configuration
        env: The underlying Gymnasium environment
        observation_space: The observation space of the environment
        action_space: The action space of the environment
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        num_envs: int = 1,
        **env_kwargs
    ):
        """
        Initialize the environment wrapper.

        Args:
            env_config: Environment configuration defining the environment
            num_envs: Number of parallel environments (for vectorized envs)
            **env_kwargs: Additional kwargs to pass to environment creation
        """
        self.env_config = env_config
        self.num_envs = num_envs

        # Create the environment
        if num_envs == 1:
            self.env = env_config.create_env(**env_kwargs)
            self._is_vectorized = False
        else:
            # For vectorized environments, try to use PufferLib or Gymnasium's vector API
            self.env = self._create_vectorized_env(env_config, num_envs, env_kwargs)
            self._is_vectorized = True

        # Expose spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Episode tracking
        self._episode_rewards: float = 0.0
        self._episode_length: int = 0

    def _create_vectorized_env(
        self,
        env_config: EnvironmentConfig,
        num_envs: int,
        env_kwargs: Dict
    ) -> gym.Env:
        """
        Create a vectorized environment.

        First tries PufferLib, falls back to Gymnasium's vector API.
        """
        try:
            import pufferlib
            import pufferlib.vector

            # Use PufferLib's vectorized environment
            return pufferlib.vector.make(
                env_config.get_env_id(),
                num_envs=num_envs,
                **env_config.get_env_kwargs(),
                **env_kwargs
            )
        except ImportError:
            # Fall back to Gymnasium's vector API
            def make_env():
                return env_config.create_env(**env_kwargs)

            return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            Tuple of (observation, info dict)
        """
        self._episode_rewards = 0.0
        self._episode_length = 0

        kwargs = {}
        if seed is not None:
            kwargs['seed'] = seed
        if options is not None:
            kwargs['options'] = options

        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(
        self,
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track episode stats
        self._episode_rewards += float(reward) if not self._is_vectorized else float(np.sum(reward))
        self._episode_length += 1

        return obs, reward, terminated, truncated, info

    def get_observation_description(self) -> str:
        """
        Get human-readable observation description for LLM prompts.

        Returns:
            Description string from the environment config
        """
        return self.env_config.get_observation_description()

    def get_action_description(self) -> str:
        """
        Get human-readable action description for LLM prompts.

        Returns:
            Description string from the environment config
        """
        return self.env_config.get_action_description()

    def get_reward_description(self) -> str:
        """
        Get reward structure description for LLM prompts.

        Returns:
            Description string from the environment config
        """
        return self.env_config.get_reward_description()

    def extract_metrics(self, info: Dict) -> Dict[str, float]:
        """
        Extract environment-specific metrics from info dict.

        Args:
            info: Info dictionary from step() or episode end

        Returns:
            Dictionary of metric names to values
        """
        return self.env_config.extract_episode_metrics(info)

    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get accumulated episode statistics.

        Returns:
            Dictionary with episode_reward and episode_length
        """
        return {
            "episode_reward": self._episode_rewards,
            "episode_length": self._episode_length,
        }

    def get_player_interface_template(self) -> str:
        """
        Get the player interface template for code generation.

        Returns:
            Python code template string
        """
        return self.env_config.get_player_interface_template()

    def get_context_for_llm(self) -> str:
        """
        Get comprehensive context for LLM prompts.

        Returns:
            Formatted context string
        """
        return self.env_config.get_context_for_llm()

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment and release resources."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    @property
    def unwrapped(self) -> gym.Env:
        """Access the underlying unwrapped environment."""
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
