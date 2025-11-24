"""Tests for EnvironmentConfig ABC."""

import pytest
from abc import ABC
from typing import Dict, Any
from tetris_evolve.environment.base import EnvironmentConfig


class ConcreteEnvironmentConfig(EnvironmentConfig):
    """Concrete implementation for testing."""

    def get_env_id(self) -> str:
        return "TestEnv-v0"

    def get_env_kwargs(self) -> Dict[str, Any]:
        return {"render_mode": None}

    def get_observation_description(self) -> str:
        return "Test observation space"

    def get_action_description(self) -> str:
        return "Test action space"

    def get_reward_description(self) -> str:
        return "Test reward structure"

    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        return {"test_metric": 1.0}

    def get_player_interface_template(self) -> str:
        return "def select_action(observation): pass"


class IncompleteEnvironmentConfig(EnvironmentConfig):
    """Incomplete implementation missing some methods."""

    def get_env_id(self) -> str:
        return "IncompleteEnv-v0"


@pytest.mark.unit
class TestEnvironmentConfig:
    """Test suite for EnvironmentConfig ABC."""

    def test_is_abstract_base_class(self):
        """Test that EnvironmentConfig is an ABC."""
        assert issubclass(EnvironmentConfig, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that EnvironmentConfig cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EnvironmentConfig()

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError):
            IncompleteEnvironmentConfig()

    def test_concrete_implementation_works(self):
        """Test that complete concrete implementations can be instantiated."""
        config = ConcreteEnvironmentConfig()
        assert config is not None

    def test_get_env_id(self):
        """Test get_env_id returns string."""
        config = ConcreteEnvironmentConfig()
        env_id = config.get_env_id()
        assert isinstance(env_id, str)
        assert env_id == "TestEnv-v0"

    def test_get_env_kwargs(self):
        """Test get_env_kwargs returns dict."""
        config = ConcreteEnvironmentConfig()
        kwargs = config.get_env_kwargs()
        assert isinstance(kwargs, dict)
        assert "render_mode" in kwargs

    def test_get_observation_description(self):
        """Test get_observation_description returns string."""
        config = ConcreteEnvironmentConfig()
        desc = config.get_observation_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_action_description(self):
        """Test get_action_description returns string."""
        config = ConcreteEnvironmentConfig()
        desc = config.get_action_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_reward_description(self):
        """Test get_reward_description returns string."""
        config = ConcreteEnvironmentConfig()
        desc = config.get_reward_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_extract_episode_metrics(self):
        """Test extract_episode_metrics returns dict of floats."""
        config = ConcreteEnvironmentConfig()
        info = {"episode": {"r": 100, "l": 50}}
        metrics = config.extract_episode_metrics(info)
        assert isinstance(metrics, dict)
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_get_player_interface_template(self):
        """Test get_player_interface_template returns code string."""
        config = ConcreteEnvironmentConfig()
        template = config.get_player_interface_template()
        assert isinstance(template, str)
        assert "def" in template  # Should contain function definition
