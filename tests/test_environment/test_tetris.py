"""
Tests for Tetris environment configuration.
"""
import pytest
import numpy as np
from typing import Dict

from tetris_evolve.environment.base import EnvironmentConfig
from tetris_evolve.environment.tetris import TetrisConfig


class TestTetrisConfig:
    """Tests for TetrisConfig implementation."""

    @pytest.fixture
    def config(self):
        """Create a TetrisConfig instance."""
        return TetrisConfig()

    @pytest.fixture
    def custom_config(self):
        """Create a TetrisConfig with custom parameters."""
        return TetrisConfig(width=12, height=24)

    def test_tetris_config_is_environment_config(self, config):
        """TetrisConfig should be an EnvironmentConfig."""
        assert isinstance(config, EnvironmentConfig)

    def test_get_env_id(self, config):
        """Should return a valid environment ID."""
        env_id = config.get_env_id()
        assert isinstance(env_id, str)
        assert len(env_id) > 0

    def test_get_env_kwargs_default(self, config):
        """Should return default environment kwargs."""
        kwargs = config.get_env_kwargs()
        assert isinstance(kwargs, dict)
        assert kwargs.get("width") == 10
        assert kwargs.get("height") == 20

    def test_get_env_kwargs_custom(self, custom_config):
        """Should return custom environment kwargs."""
        kwargs = custom_config.get_env_kwargs()
        assert kwargs.get("width") == 12
        assert kwargs.get("height") == 24

    def test_get_observation_description(self, config):
        """Should return a descriptive observation string for LLM."""
        desc = config.get_observation_description()

        assert isinstance(desc, str)
        assert len(desc) > 50  # Should be descriptive

        # Should mention key observation components
        assert "board" in desc.lower()
        assert "piece" in desc.lower()

    def test_get_action_description(self, config):
        """Should return a descriptive action string for LLM."""
        desc = config.get_action_description()

        assert isinstance(desc, str)
        assert len(desc) > 30  # Should be descriptive

        # Should mention key actions
        desc_lower = desc.lower()
        assert any(word in desc_lower for word in ["left", "right", "rotate", "drop"])

    def test_get_reward_description(self, config):
        """Should return a descriptive reward string for LLM."""
        desc = config.get_reward_description()

        assert isinstance(desc, str)
        assert len(desc) > 20  # Should be descriptive

        # Should mention line clearing or scoring
        desc_lower = desc.lower()
        assert any(word in desc_lower for word in ["line", "score", "clear", "reward"])

    def test_extract_episode_metrics_with_full_info(self, config):
        """Should extract Tetris-specific metrics from info dict."""
        info = {
            "score": 1200,
            "lines_cleared": 10,
            "pieces_placed": 50,
        }

        metrics = config.extract_episode_metrics(info)

        assert isinstance(metrics, dict)
        assert metrics["score"] == 1200
        assert metrics["lines_cleared"] == 10
        assert metrics["pieces_placed"] == 50

    def test_extract_episode_metrics_with_empty_info(self, config):
        """Should handle empty info dict gracefully."""
        info = {}

        metrics = config.extract_episode_metrics(info)

        assert isinstance(metrics, dict)
        # Should provide defaults
        assert "score" in metrics
        assert metrics["score"] == 0

    def test_extract_episode_metrics_with_partial_info(self, config):
        """Should handle partial info dict."""
        info = {"score": 500}

        metrics = config.extract_episode_metrics(info)

        assert metrics["score"] == 500
        assert metrics.get("lines_cleared", 0) == 0

    def test_get_player_interface_template(self, config):
        """Should return a valid player interface template."""
        template = config.get_player_interface_template()

        assert isinstance(template, str)
        assert len(template) > 50

        # Should be valid Python that defines a class
        assert "class" in template
        assert "def" in template
        assert "select_action" in template or "get_action" in template

    def test_create_env(self, config):
        """Should be able to create an environment from config."""
        env = config.create_env()

        assert env is not None
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Clean up
        env.close()

    def test_created_env_is_playable(self, config):
        """Created environment should be playable."""
        env = config.create_env()

        obs, info = env.reset()
        assert obs is not None

        # Take a few random steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                break

        env.close()

    def test_config_immutable_parameters(self, config):
        """Config parameters should be accessible."""
        assert config.width == 10
        assert config.height == 20

    def test_config_with_render_mode(self):
        """Should support render_mode parameter."""
        config = TetrisConfig(render_mode="rgb_array")
        kwargs = config.get_env_kwargs()
        assert kwargs.get("render_mode") == "rgb_array"


class TestTetrisConfigMetrics:
    """Tests for Tetris metrics extraction."""

    @pytest.fixture
    def config(self):
        return TetrisConfig()

    def test_metrics_types(self, config):
        """All extracted metrics should be numeric."""
        info = {
            "score": 1000,
            "lines_cleared": 5,
            "pieces_placed": 20,
        }

        metrics = config.extract_episode_metrics(info)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"

    def test_metrics_keys_are_consistent(self, config):
        """Metrics should always have consistent keys."""
        info1 = {"score": 100}
        info2 = {"score": 200, "lines_cleared": 5, "pieces_placed": 10}

        metrics1 = config.extract_episode_metrics(info1)
        metrics2 = config.extract_episode_metrics(info2)

        # Both should have the same keys
        assert set(metrics1.keys()) == set(metrics2.keys())
