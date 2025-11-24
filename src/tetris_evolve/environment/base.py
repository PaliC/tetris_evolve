"""
Base environment configuration interface.

This module provides the abstract base class for environment-agnostic configuration.
All environment implementations (Tetris, Atari, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class EnvironmentConfig(ABC):
    """
    Abstract base class for environment configuration.

    Each environment (Tetris, Atari, etc.) implements this interface to provide
    environment-specific configuration and metadata for the evolution system.

    This design allows the system to work with any PufferLib/Gymnasium environment
    while providing LLMs with necessary context about observations, actions, and rewards.
    """

    @abstractmethod
    def get_env_id(self) -> str:
        """
        Return PufferLib/Gymnasium environment ID.

        Returns:
            str: Environment identifier (e.g., "tetris-v0", "ALE/Pong-v5")

        Example:
            >>> config = TetrisConfig()
            >>> config.get_env_id()
            'tetris-v0'
        """
        pass

    @abstractmethod
    def get_env_kwargs(self) -> Dict[str, Any]:
        """
        Return environment-specific configuration kwargs.

        Returns:
            Dict[str, Any]: Keyword arguments to pass to environment constructor

        Example:
            >>> config = TetrisConfig()
            >>> config.get_env_kwargs()
            {'render_mode': None, 'max_episode_steps': 1000}
        """
        pass

    @abstractmethod
    def get_observation_description(self) -> str:
        """
        Return human-readable description of observation space for LLM.

        This description helps Child RLLMs understand what observations they'll
        receive when writing player code.

        Returns:
            str: Description of observation space, structure, and meaning

        Example:
            >>> config = TetrisConfig()
            >>> desc = config.get_observation_description()
            >>> print(desc)
            The observation is a numpy array of shape (20, 10) representing the
            Tetris board. Each cell contains:
            - 0: empty cell
            - 1-7: occupied by tetromino piece type
            ...
        """
        pass

    @abstractmethod
    def get_action_description(self) -> str:
        """
        Return human-readable description of action space for LLM.

        This description helps Child RLLMs understand what actions are available
        when writing player code.

        Returns:
            str: Description of action space and action meanings

        Example:
            >>> config = TetrisConfig()
            >>> desc = config.get_action_description()
            >>> print(desc)
            Actions are integers from 0 to 5:
            - 0: Move left
            - 1: Move right
            - 2: Rotate clockwise
            ...
        """
        pass

    @abstractmethod
    def get_reward_description(self) -> str:
        """
        Return description of reward structure for LLM.

        This description helps LLMs understand the objective and reward signal.

        Returns:
            str: Description of how rewards are calculated

        Example:
            >>> config = TetrisConfig()
            >>> desc = config.get_reward_description()
            >>> print(desc)
            Rewards are based on:
            - +100 for each line cleared
            - +1 for each piece placed successfully
            - 0 when game ends
        """
        pass

    @abstractmethod
    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """
        Extract environment-specific metrics from episode info dict.

        This allows tracking environment-specific statistics (e.g., lines cleared
        for Tetris, score for Atari) in addition to generic metrics.

        Args:
            info: Episode info dict from Gymnasium environment

        Returns:
            Dict[str, float]: Environment-specific metrics as key-value pairs

        Example:
            >>> config = TetrisConfig()
            >>> info = {"lines_cleared": 15, "pieces_placed": 50}
            >>> metrics = config.extract_episode_metrics(info)
            >>> print(metrics)
            {'lines_cleared': 15.0, 'pieces_placed': 50.0, 'efficiency': 0.3}
        """
        pass

    @abstractmethod
    def get_player_interface_template(self) -> str:
        """
        Return code template for player interface.

        This template shows Child RLLMs the expected function signature and
        structure for player code they need to generate.

        Returns:
            str: Python code template with required interface

        Example:
            >>> config = TetrisConfig()
            >>> template = config.get_player_interface_template()
            >>> print(template)
            '''
            def select_action(observation, info):
                \"\"\"
                Select action based on current observation.

                Args:
                    observation: Current game state
                    info: Additional game information

                Returns:
                    int: Action to take (0-5)
                \"\"\"
                # Your code here
                return action
            '''
        """
        pass
