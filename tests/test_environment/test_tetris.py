"""Tests for TetrisConfig."""

import pytest
from tetris_evolve.environment.tetris import TetrisConfig
from tetris_evolve.environment.base import EnvironmentConfig


@pytest.mark.unit
class TestTetrisConfig:
    """Test suite for TetrisConfig."""

    def test_is_environment_config(self):
        """Test that TetrisConfig is an EnvironmentConfig."""
        config = TetrisConfig()
        assert isinstance(config, EnvironmentConfig)

    def test_get_env_id(self):
        """Test get_env_id returns correct Tetris environment ID."""
        config = TetrisConfig()
        env_id = config.get_env_id()
        assert isinstance(env_id, str)
        # Should be a Tetris-related environment
        assert "tetris" in env_id.lower() or "Tetris" in env_id

    def test_get_env_kwargs(self):
        """Test get_env_kwargs returns valid kwargs dict."""
        config = TetrisConfig()
        kwargs = config.get_env_kwargs()
        assert isinstance(kwargs, dict)

    def test_get_env_kwargs_with_custom_params(self):
        """Test creating TetrisConfig with custom parameters."""
        config = TetrisConfig(render_mode="human", max_episode_steps=500)
        kwargs = config.get_env_kwargs()
        assert kwargs.get("render_mode") == "human"
        assert kwargs.get("max_episode_steps") == 500

    def test_get_observation_description(self):
        """Test get_observation_description returns helpful description."""
        config = TetrisConfig()
        desc = config.get_observation_description()
        assert isinstance(desc, str)
        assert len(desc) > 50  # Should be detailed
        # Should mention key concepts
        desc_lower = desc.lower()
        assert "board" in desc_lower or "grid" in desc_lower
        assert "observation" in desc_lower or "state" in desc_lower

    def test_get_action_description(self):
        """Test get_action_description returns helpful description."""
        config = TetrisConfig()
        desc = config.get_action_description()
        assert isinstance(desc, str)
        assert len(desc) > 50  # Should be detailed
        # Should mention Tetris actions
        desc_lower = desc.lower()
        assert "action" in desc_lower
        # Typical Tetris actions
        assert any(
            word in desc_lower
            for word in ["move", "rotate", "left", "right", "down", "drop"]
        )

    def test_get_reward_description(self):
        """Test get_reward_description returns helpful description."""
        config = TetrisConfig()
        desc = config.get_reward_description()
        assert isinstance(desc, str)
        assert len(desc) > 30  # Should be descriptive
        desc_lower = desc.lower()
        assert "reward" in desc_lower
        # Should mention Tetris scoring concepts
        assert any(word in desc_lower for word in ["line", "piece", "score", "clear"])

    def test_extract_episode_metrics_basic(self):
        """Test extract_episode_metrics with basic info dict."""
        config = TetrisConfig()
        info = {}
        metrics = config.extract_episode_metrics(info)
        assert isinstance(metrics, dict)
        # Should return dict of floats (or ints convertible to float)
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))

    def test_extract_episode_metrics_with_lines(self):
        """Test extract_episode_metrics extracts lines cleared."""
        config = TetrisConfig()
        info = {"lines_cleared": 10}
        metrics = config.extract_episode_metrics(info)
        # Should extract lines_cleared if present
        if "lines_cleared" in metrics:
            assert metrics["lines_cleared"] == 10.0

    def test_extract_episode_metrics_with_pieces(self):
        """Test extract_episode_metrics extracts pieces placed."""
        config = TetrisConfig()
        info = {"pieces_placed": 50}
        metrics = config.extract_episode_metrics(info)
        # Should extract pieces_placed if present
        if "pieces_placed" in metrics:
            assert metrics["pieces_placed"] == 50.0

    def test_extract_episode_metrics_multiple_fields(self):
        """Test extract_episode_metrics with multiple fields."""
        config = TetrisConfig()
        info = {
            "lines_cleared": 15,
            "pieces_placed": 60,
            "max_height": 18,
            "holes": 5,
        }
        metrics = config.extract_episode_metrics(info)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        # All values should be numeric
        for value in metrics.values():
            assert isinstance(value, (int, float))

    def test_get_player_interface_template(self):
        """Test get_player_interface_template returns valid Python code template."""
        config = TetrisConfig()
        template = config.get_player_interface_template()
        assert isinstance(template, str)
        assert len(template) > 100  # Should be substantial
        # Should contain function definition
        assert "def " in template
        # Should have select_action or similar
        assert "action" in template.lower()
        # Should mention observation
        assert "observation" in template.lower() or "obs" in template.lower()
        # Should have return statement
        assert "return" in template.lower()

    def test_get_player_interface_template_is_valid_python(self):
        """Test that player interface template is syntactically valid Python."""
        config = TetrisConfig()
        template = config.get_player_interface_template()
        # Try to compile it
        try:
            compile(template, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Player interface template has syntax error: {e}")

    def test_config_immutability(self):
        """Test that config parameters are stored correctly."""
        config1 = TetrisConfig()
        config2 = TetrisConfig(render_mode="rgb_array")

        # Should have different configs
        kwargs1 = config1.get_env_kwargs()
        kwargs2 = config2.get_env_kwargs()

        assert kwargs1.get("render_mode") != kwargs2.get("render_mode")
