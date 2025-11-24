"""
Tests for environment abstraction layer.

Following TDD: These tests define the expected behavior of EnvironmentConfig
and GenericEnvironmentWrapper before implementation.
"""
import pytest
from abc import ABC
from typing import Dict, Any

# These imports will work once we implement the modules
from tetris_evolve.environment.base import EnvironmentConfig
from tetris_evolve.environment.wrapper import GenericEnvironmentWrapper


class TestEnvironmentConfigABC:
    """Tests for the EnvironmentConfig abstract base class."""

    def test_environment_config_is_abstract(self):
        """EnvironmentConfig should be an abstract base class."""
        assert issubclass(EnvironmentConfig, ABC)

        # Should not be instantiable directly
        with pytest.raises(TypeError):
            EnvironmentConfig()

    def test_environment_config_has_required_abstract_methods(self):
        """EnvironmentConfig should define required abstract methods."""
        required_methods = [
            'get_env_id',
            'get_env_kwargs',
            'get_observation_description',
            'get_action_description',
            'get_reward_description',
            'extract_episode_metrics',
            'get_player_interface_template',
        ]

        for method in required_methods:
            assert hasattr(EnvironmentConfig, method), f"Missing abstract method: {method}"

    def test_concrete_implementation_works(self):
        """A concrete implementation should be instantiable."""

        class MockConfig(EnvironmentConfig):
            def get_env_id(self) -> str:
                return "MockEnv-v0"

            def get_env_kwargs(self) -> Dict[str, Any]:
                return {"param": "value"}

            def get_observation_description(self) -> str:
                return "Mock observation"

            def get_action_description(self) -> str:
                return "Mock action"

            def get_reward_description(self) -> str:
                return "Mock reward"

            def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
                return {"score": info.get("score", 0.0)}

            def get_player_interface_template(self) -> str:
                return "class MockPlayer: pass"

        config = MockConfig()
        assert config.get_env_id() == "MockEnv-v0"
        assert config.get_env_kwargs() == {"param": "value"}


class TestGenericEnvironmentWrapper:
    """Tests for GenericEnvironmentWrapper."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock environment config for testing."""
        from tetris_evolve.environment.tetris import TetrisConfig
        return TetrisConfig()

    def test_wrapper_initialization(self, mock_config):
        """Wrapper should initialize with an environment config."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        assert wrapper.env_config is mock_config
        assert wrapper.env is not None

    def test_wrapper_has_observation_space(self, mock_config):
        """Wrapper should expose observation space."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        assert wrapper.observation_space is not None

    def test_wrapper_has_action_space(self, mock_config):
        """Wrapper should expose action space."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        assert wrapper.action_space is not None

    def test_wrapper_reset(self, mock_config):
        """Wrapper reset should return observation and info."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        obs, info = wrapper.reset()

        assert obs is not None
        assert isinstance(info, dict)

    def test_wrapper_step(self, mock_config):
        """Wrapper step should return standard Gymnasium tuple."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        wrapper.reset()

        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_wrapper_get_observation_description(self, mock_config):
        """Wrapper should provide observation description for LLM."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        desc = wrapper.get_observation_description()

        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_wrapper_get_action_description(self, mock_config):
        """Wrapper should provide action description for LLM."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        desc = wrapper.get_action_description()

        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_wrapper_extract_metrics(self, mock_config):
        """Wrapper should extract environment-specific metrics."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        wrapper.reset()

        # Run a few steps to generate info
        for _ in range(10):
            action = wrapper.action_space.sample()
            obs, reward, terminated, truncated, info = wrapper.step(action)
            if terminated:
                break

        metrics = wrapper.extract_metrics(info)

        assert isinstance(metrics, dict)
        # Tetris should have score metric
        assert "score" in metrics

    def test_wrapper_close(self, mock_config):
        """Wrapper should have a close method."""
        wrapper = GenericEnvironmentWrapper(mock_config)
        wrapper.reset()
        wrapper.close()  # Should not raise


class TestEnvironmentConfigRegistry:
    """Tests for environment config registry/factory."""

    def test_get_available_environments(self):
        """Should be able to list available environment configs."""
        from tetris_evolve.environment import get_available_environments

        envs = get_available_environments()
        assert isinstance(envs, list)
        assert "tetris" in envs

    def test_create_environment_config(self):
        """Should be able to create environment config by name."""
        from tetris_evolve.environment import create_environment_config

        config = create_environment_config("tetris")
        assert isinstance(config, EnvironmentConfig)
        assert "tetris" in config.get_env_id().lower() or "Tetris" in config.get_env_id()

    def test_create_unknown_environment_raises(self):
        """Creating unknown environment should raise ValueError."""
        from tetris_evolve.environment import create_environment_config

        with pytest.raises(ValueError):
            create_environment_config("unknown_game_xyz")
