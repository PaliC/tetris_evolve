"""
Base classes for environment abstraction.

This module provides the abstract interface that all environment configurations
must implement. The design allows the evolution system to work with any
PufferLib/Gymnasium-compatible environment.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import gymnasium as gym


class EnvironmentConfig(ABC):
    """
    Abstract base class for environment configuration.

    Each environment (Tetris, Atari, custom games, etc.) implements this interface
    to provide environment-specific configuration and metadata for the evolution system.

    The config provides:
    - Environment creation parameters
    - Human-readable descriptions for LLM prompts
    - Metric extraction from episode info
    - Player interface templates for code generation
    """

    @abstractmethod
    def get_env_id(self) -> str:
        """
        Return the environment identifier.

        This could be a Gymnasium/PufferLib registered ID or a custom identifier
        that the create_env method understands.

        Returns:
            Environment identifier string (e.g., "Tetris-v0", "ALE/Pong-v5")
        """
        pass

    @abstractmethod
    def get_env_kwargs(self) -> Dict[str, Any]:
        """
        Return environment-specific configuration parameters.

        These kwargs are passed to the environment constructor or registration
        function when creating the environment.

        Returns:
            Dictionary of environment configuration parameters
        """
        pass

    @abstractmethod
    def get_observation_description(self) -> str:
        """
        Return a human-readable description of the observation space.

        This description is included in LLM prompts to help the model
        understand what information is available in observations.

        Returns:
            Multi-line string describing observation format and contents
        """
        pass

    @abstractmethod
    def get_action_description(self) -> str:
        """
        Return a human-readable description of the action space.

        This description is included in LLM prompts to help the model
        understand what actions are available.

        Returns:
            Multi-line string describing available actions
        """
        pass

    @abstractmethod
    def get_reward_description(self) -> str:
        """
        Return a description of the reward structure.

        This helps the LLM understand the game's objectives and scoring.

        Returns:
            Multi-line string describing reward signals and goals
        """
        pass

    @abstractmethod
    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """
        Extract environment-specific metrics from episode info.

        This method processes the info dict returned by env.step() to
        extract meaningful metrics for evaluation and logging.

        Args:
            info: The info dictionary from env.step() or episode end

        Returns:
            Dictionary of metric names to numeric values
        """
        pass

    @abstractmethod
    def get_player_interface_template(self) -> str:
        """
        Return a code template for the player interface.

        This template defines the class/function signature that evolved
        programs must implement. It's used by the LLM to generate code
        with the correct interface.

        Returns:
            Python code string defining the player interface
        """
        pass

    def create_env(self, **override_kwargs) -> gym.Env:
        """
        Create and return an environment instance.

        Override kwargs can be provided to customize the environment
        beyond the default configuration.

        Args:
            **override_kwargs: Parameters to override from get_env_kwargs()

        Returns:
            A Gymnasium-compatible environment instance
        """
        kwargs = self.get_env_kwargs().copy()
        kwargs.update(override_kwargs)

        # Default implementation assumes a registered Gymnasium environment
        # Subclasses can override for custom environment creation
        return gym.make(self.get_env_id(), **kwargs)

    def get_metrics_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Return specification for metrics this environment produces.

        This is used for validation and documentation purposes.

        Returns:
            Dictionary mapping metric names to their specifications
            (type, description, aggregation method, etc.)
        """
        # Default implementation - subclasses can override
        return {
            "score": {
                "type": "float",
                "description": "Primary game score",
                "aggregation": "mean",
                "higher_is_better": True,
            }
        }

    def get_context_for_llm(self) -> str:
        """
        Generate a comprehensive context string for LLM prompts.

        Combines observation, action, and reward descriptions into a
        format suitable for inclusion in LLM prompts.

        Returns:
            Formatted context string
        """
        return f"""Environment: {self.get_env_id()}

OBSERVATIONS:
{self.get_observation_description()}

ACTIONS:
{self.get_action_description()}

REWARDS:
{self.get_reward_description()}

PLAYER INTERFACE:
{self.get_player_interface_template()}
"""
