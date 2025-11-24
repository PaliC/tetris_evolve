"""
Tests for the evaluation framework.

Following TDD: These tests define the expected behavior of the player
evaluation system.
"""
import pytest
import numpy as np
from typing import Dict, Any

from tetris_evolve.environment import TetrisConfig
from tetris_evolve.evaluation import (
    evaluate_player,
    EvaluationResult,
    MetricStats,
    PlayerExecutionError,
)


class MockPlayer:
    """A simple mock player for testing."""

    def __init__(self, action: int = 5):
        """Initialize with a fixed action to take."""
        self.action = action
        self.call_count = 0

    def select_action(self, observation: np.ndarray) -> int:
        """Always return the same action."""
        self.call_count += 1
        return self.action

    def reset(self):
        """Reset player state."""
        self.call_count = 0


class BrokenPlayer:
    """A player that raises errors."""

    def select_action(self, observation: np.ndarray) -> int:
        raise RuntimeError("Intentional error for testing")

    def reset(self):
        pass


class TestEvaluatePlayer:
    """Tests for the evaluate_player function."""

    @pytest.fixture
    def env_config(self):
        """Create a Tetris config for testing."""
        return TetrisConfig()

    @pytest.fixture
    def player(self):
        """Create a mock player."""
        return MockPlayer(action=5)  # Hard drop

    def test_evaluate_player_returns_result(self, env_config, player):
        """evaluate_player should return an EvaluationResult."""
        result = evaluate_player(player, env_config, num_episodes=2)

        assert isinstance(result, EvaluationResult)

    def test_result_has_generic_metrics(self, env_config, player):
        """Result should include generic metrics (reward, length)."""
        result = evaluate_player(player, env_config, num_episodes=2)

        assert "total_reward" in result.generic_metrics
        assert "episode_length" in result.generic_metrics

        # Each metric should have stats
        reward_stats = result.generic_metrics["total_reward"]
        assert isinstance(reward_stats, MetricStats)
        assert hasattr(reward_stats, "mean")
        assert hasattr(reward_stats, "std")
        assert hasattr(reward_stats, "min")
        assert hasattr(reward_stats, "max")

    def test_result_has_env_metrics(self, env_config, player):
        """Result should include environment-specific metrics."""
        result = evaluate_player(player, env_config, num_episodes=2)

        # Tetris should have score metric
        assert "score" in result.env_metrics
        assert isinstance(result.env_metrics["score"], MetricStats)

    def test_result_has_episode_details(self, env_config, player):
        """Result should include per-episode details."""
        result = evaluate_player(player, env_config, num_episodes=3)

        assert len(result.episode_results) == 3

        for episode in result.episode_results:
            assert "episode_id" in episode
            assert "reward" in episode
            assert "length" in episode
            assert "env_metrics" in episode

    def test_result_tracks_errors(self, env_config):
        """Result should track code errors."""
        broken_player = BrokenPlayer()
        result = evaluate_player(broken_player, env_config, num_episodes=2)

        assert result.code_errors > 0

    def test_num_episodes_respected(self, env_config, player):
        """Should run the specified number of episodes."""
        result = evaluate_player(player, env_config, num_episodes=5)

        # May have fewer if errors occurred
        assert len(result.episode_results) <= 5

    def test_max_steps_per_episode(self, env_config, player):
        """Should respect max_steps_per_episode limit."""
        result = evaluate_player(
            player, env_config, num_episodes=1, max_steps_per_episode=50
        )

        for episode in result.episode_results:
            assert episode["length"] <= 50

    def test_player_reset_called(self, env_config, player):
        """Player reset should be called between episodes."""
        result = evaluate_player(player, env_config, num_episodes=3)

        # Player was used, reset should have been called
        # (We can't easily verify this with mock, but function shouldn't error)
        assert result is not None


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Should be able to create EvaluationResult."""
        result = EvaluationResult(
            generic_metrics={
                "total_reward": MetricStats(mean=100.0, std=10.0, min=80.0, max=120.0)
            },
            env_metrics={
                "score": MetricStats(mean=500.0, std=100.0, min=300.0, max=700.0)
            },
            code_errors=0,
            episode_results=[],
        )

        assert result.generic_metrics["total_reward"].mean == 100.0
        assert result.env_metrics["score"].mean == 500.0
        assert result.code_errors == 0

    def test_evaluation_result_to_dict(self):
        """EvaluationResult should be convertible to dict."""
        result = EvaluationResult(
            generic_metrics={
                "total_reward": MetricStats(mean=100.0, std=10.0, min=80.0, max=120.0)
            },
            env_metrics={},
            code_errors=0,
            episode_results=[],
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert "generic_metrics" in d
        assert "env_metrics" in d
        assert "code_errors" in d

    def test_evaluation_result_primary_score(self):
        """EvaluationResult should compute a primary score."""
        result = EvaluationResult(
            generic_metrics={
                "total_reward": MetricStats(mean=100.0, std=10.0, min=80.0, max=120.0)
            },
            env_metrics={
                "score": MetricStats(mean=500.0, std=100.0, min=300.0, max=700.0)
            },
            code_errors=0,
            episode_results=[],
        )

        # Primary score should be computable
        score = result.get_primary_score()
        assert isinstance(score, float)


class TestMetricStats:
    """Tests for MetricStats dataclass."""

    def test_metric_stats_from_values(self):
        """Should create MetricStats from a list of values."""
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = MetricStats.from_values(values)

        assert stats.mean == 300.0
        assert stats.min == 100.0
        assert stats.max == 500.0
        assert stats.std > 0

    def test_metric_stats_from_empty_values(self):
        """Should handle empty values list."""
        stats = MetricStats.from_values([])

        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_metric_stats_to_dict(self):
        """Should convert to dictionary."""
        stats = MetricStats(mean=10.0, std=2.0, min=5.0, max=15.0)
        d = stats.to_dict()

        assert d == {"mean": 10.0, "std": 2.0, "min": 5.0, "max": 15.0}


class TestEvaluatePlayerFromCode:
    """Tests for evaluating player from source code string."""

    @pytest.fixture
    def env_config(self):
        return TetrisConfig()

    def test_evaluate_from_code_string(self, env_config):
        """Should evaluate a player defined as a code string."""
        from tetris_evolve.evaluation import evaluate_player_from_code

        code = '''
import numpy as np

class TetrisPlayer:
    def __init__(self):
        pass

    def select_action(self, observation):
        return 5  # Hard drop

    def reset(self):
        pass
'''
        result = evaluate_player_from_code(code, env_config, num_episodes=2)

        assert isinstance(result, EvaluationResult)
        assert result.code_errors == 0

    def test_evaluate_from_invalid_code(self, env_config):
        """Should handle invalid code gracefully."""
        from tetris_evolve.evaluation import evaluate_player_from_code

        code = "this is not valid python {"

        result = evaluate_player_from_code(code, env_config, num_episodes=2)

        assert result.code_errors > 0

    def test_evaluate_from_code_missing_class(self, env_config):
        """Should handle code without required class."""
        from tetris_evolve.evaluation import evaluate_player_from_code

        code = '''
def some_function():
    pass
'''
        result = evaluate_player_from_code(code, env_config, num_episodes=2)

        assert result.code_errors > 0


class TestParallelEvaluation:
    """Tests for parallel evaluation (if implemented)."""

    @pytest.fixture
    def env_config(self):
        return TetrisConfig()

    @pytest.fixture
    def player(self):
        return MockPlayer()

    @pytest.mark.slow
    def test_parallel_evaluation(self, env_config, player):
        """Parallel evaluation should produce same results as serial."""
        from tetris_evolve.evaluation import evaluate_player

        # Serial evaluation
        serial_result = evaluate_player(
            player, env_config, num_episodes=4, parallel=False
        )

        # Parallel evaluation (if supported)
        parallel_result = evaluate_player(
            player, env_config, num_episodes=4, parallel=True
        )

        # Results should be similar (not identical due to randomness)
        assert parallel_result.code_errors == serial_result.code_errors
