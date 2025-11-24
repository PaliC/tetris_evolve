"""Tests for GenericEnvironmentWrapper."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
from tetris_evolve.environment.wrapper import GenericEnvironmentWrapper
from tetris_evolve.environment.base import EnvironmentConfig


class MockEnvironmentConfig(EnvironmentConfig):
    """Mock environment config for testing."""

    def get_env_id(self) -> str:
        return "MockEnv-v0"

    def get_env_kwargs(self) -> Dict[str, Any]:
        return {"render_mode": None}

    def get_observation_description(self) -> str:
        return "Mock observation"

    def get_action_description(self) -> str:
        return "Mock actions"

    def get_reward_description(self) -> str:
        return "Mock rewards"

    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        return {"mock_metric": float(info.get("mock_value", 0))}

    def get_player_interface_template(self) -> str:
        return "def select_action(obs): return 0"


@pytest.mark.unit
class TestGenericEnvironmentWrapper:
    """Test suite for GenericEnvironmentWrapper."""

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_initialization(self, mock_gym):
        """Test wrapper initialization."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)

        assert wrapper.config == config
        mock_gym.make.assert_called_once_with("MockEnv-v0", **{"render_mode": None})

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_reset(self, mock_gym):
        """Test environment reset."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_obs = np.array([1, 2, 3])
        mock_info = {"step": 0}
        mock_env.reset.return_value = (mock_obs, mock_info)
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)
        obs, info = wrapper.reset()

        assert np.array_equal(obs, mock_obs)
        assert info == mock_info
        mock_env.reset.assert_called_once()

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_reset_with_seed(self, mock_gym):
        """Test environment reset with seed."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.array([1, 2, 3]), {})
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)
        wrapper.reset(seed=42)

        mock_env.reset.assert_called_with(seed=42)

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_step(self, mock_gym):
        """Test environment step."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_obs = np.array([1, 2, 3])
        mock_env.step.return_value = (mock_obs, 1.0, False, False, {"info": "data"})
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)
        obs, reward, terminated, truncated, info = wrapper.step(0)

        assert np.array_equal(obs, mock_obs)
        assert reward == 1.0
        assert terminated == False
        assert truncated == False
        assert info == {"info": "data"}
        mock_env.step.assert_called_once_with(0)

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_run_episode_with_callable(self, mock_gym):
        """Test running a complete episode with callable player."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()

        # Mock reset
        mock_env.reset.return_value = (np.array([0]), {"start": True})

        # Mock three steps then done
        mock_env.step.side_effect = [
            (np.array([1]), 1.0, False, False, {"step": 1}),
            (np.array([2]), 2.0, False, False, {"step": 2}),
            (np.array([3]), 3.0, True, False, {"step": 3, "mock_value": 10}),
        ]

        mock_gym.make.return_value = mock_env

        # Simple player that always returns action 0
        def player(obs, info):
            return 0

        wrapper = GenericEnvironmentWrapper(config)
        metrics = wrapper.run_episode(player, seed=42)

        # Check generic metrics
        assert "total_reward" in metrics
        assert metrics["total_reward"] == 6.0  # 1 + 2 + 3
        assert metrics["episode_length"] == 3
        assert "episode_time" in metrics

        # Check environment-specific metrics
        assert "mock_metric" in metrics
        assert metrics["mock_metric"] == 10.0

        mock_env.reset.assert_called_once_with(seed=42)
        assert mock_env.step.call_count == 3

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_run_episode_truncated(self, mock_gym):
        """Test running episode that gets truncated."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()

        mock_env.reset.return_value = (np.array([0]), {})
        mock_env.step.return_value = (
            np.array([1]),
            1.0,
            False,
            True,  # truncated
            {"mock_value": 5},
        )

        mock_gym.make.return_value = mock_env

        def player(obs, info):
            return 0

        wrapper = GenericEnvironmentWrapper(config)
        metrics = wrapper.run_episode(player, max_steps=1)

        assert metrics["total_reward"] == 1.0
        assert metrics["episode_length"] == 1
        assert metrics["truncated"] == True

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_run_episode_with_max_steps(self, mock_gym):
        """Test episode terminates after max_steps."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()

        mock_env.reset.return_value = (np.array([0]), {})
        # Episode never terminates naturally
        mock_env.step.return_value = (np.array([1]), 1.0, False, False, {})

        mock_gym.make.return_value = mock_env

        def player(obs, info):
            return 0

        wrapper = GenericEnvironmentWrapper(config)
        metrics = wrapper.run_episode(player, max_steps=5)

        assert metrics["episode_length"] == 5
        assert mock_env.step.call_count == 5

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_close(self, mock_gym):
        """Test closing the environment."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)
        wrapper.close()

        mock_env.close.assert_called_once()

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_context_manager(self, mock_gym):
        """Test using wrapper as context manager."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        with GenericEnvironmentWrapper(config) as wrapper:
            assert wrapper is not None

        mock_env.close.assert_called_once()

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_observation_space(self, mock_gym):
        """Test accessing observation space."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_obs_space = MagicMock()
        mock_env.observation_space = mock_obs_space
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)

        assert wrapper.observation_space == mock_obs_space

    @patch("tetris_evolve.environment.wrapper.gymnasium")
    def test_action_space(self, mock_gym):
        """Test accessing action space."""
        config = MockEnvironmentConfig()
        mock_env = MagicMock()
        mock_action_space = MagicMock()
        mock_env.action_space = mock_action_space
        mock_gym.make.return_value = mock_env

        wrapper = GenericEnvironmentWrapper(config)

        assert wrapper.action_space == mock_action_space
