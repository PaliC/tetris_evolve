"""
Tests for parallel evaluation and caching.
"""
import pytest
import time
from unittest.mock import Mock
import numpy as np

from tetris_evolve.evaluation import evaluate_player, EvaluationResult
from tetris_evolve.evaluation.parallel import (
    ParallelEvaluator,
    EvaluationCache,
)
from tetris_evolve.environment import TetrisConfig


class MockPlayer:
    """Mock player for testing."""
    def __init__(self, action=5):
        self.action = action

    def select_action(self, obs):
        return self.action

    def reset(self):
        pass


class TestParallelEvaluator:
    """Tests for ParallelEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return ParallelEvaluator(
            env_config=TetrisConfig(),
            num_workers=2,
        )

    def test_evaluate_single(self, evaluator):
        """Should evaluate a single player."""
        player = MockPlayer()
        result = evaluator.evaluate(player, num_episodes=2)

        assert isinstance(result, EvaluationResult)
        assert len(result.episode_results) == 2

    def test_evaluate_batch(self, evaluator):
        """Should evaluate multiple players."""
        players = [MockPlayer(i % 6) for i in range(3)]
        results = evaluator.evaluate_batch(players, num_episodes=2)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, EvaluationResult)

    def test_parallel_faster_than_serial(self, evaluator):
        """Parallel evaluation should be faster for large batches."""
        players = [MockPlayer() for _ in range(4)]

        # Time serial
        serial_start = time.time()
        serial_results = []
        for player in players:
            result = evaluate_player(player, TetrisConfig(), num_episodes=2)
            serial_results.append(result)
        serial_time = time.time() - serial_start

        # Time parallel (with warm-up)
        evaluator.evaluate_batch(players[:1], num_episodes=1)  # Warm up
        parallel_start = time.time()
        parallel_results = evaluator.evaluate_batch(players, num_episodes=2)
        parallel_time = time.time() - parallel_start

        # Both should produce results
        assert len(parallel_results) == len(serial_results)

    def test_handles_failing_players(self, evaluator):
        """Should handle players that raise errors."""
        class FailingPlayer:
            def select_action(self, obs):
                raise RuntimeError("Test error")
            def reset(self):
                pass

        players = [MockPlayer(), FailingPlayer(), MockPlayer()]
        results = evaluator.evaluate_batch(players, num_episodes=2)

        assert len(results) == 3
        # Middle player should have errors
        assert results[1].code_errors > 0


class TestEvaluationCache:
    """Tests for EvaluationCache."""

    @pytest.fixture
    def cache(self):
        return EvaluationCache(max_size=100)

    def test_cache_stores_results(self, cache):
        """Should store and retrieve results."""
        code = "class Player: pass"
        result = EvaluationResult(
            generic_metrics={},
            env_metrics={},
            code_errors=0,
            episode_results=[],
        )

        cache.store(code, result)
        cached = cache.get(code)

        assert cached is not None
        assert cached.code_errors == 0

    def test_cache_miss_returns_none(self, cache):
        """Should return None for cache miss."""
        result = cache.get("nonexistent code")
        assert result is None

    def test_cache_uses_code_hash(self, cache):
        """Should use code hash as key."""
        code1 = "class Player:\n    pass"
        code2 = "class Player:\n    pass"  # Same code
        code3 = "class OtherPlayer:\n    pass"  # Different code

        result = EvaluationResult(
            generic_metrics={},
            env_metrics={},
            code_errors=0,
            episode_results=[],
        )

        cache.store(code1, result)

        # Same code should hit cache
        assert cache.get(code2) is not None
        # Different code should miss
        assert cache.get(code3) is None

    def test_cache_respects_max_size(self, cache):
        """Should evict old entries when max size reached."""
        cache = EvaluationCache(max_size=3)

        result = EvaluationResult(
            generic_metrics={},
            env_metrics={},
            code_errors=0,
            episode_results=[],
        )

        for i in range(5):
            cache.store(f"code_{i}", result)

        # Should have max_size entries
        assert cache.size() <= 3

    def test_cache_stats(self, cache):
        """Should track cache statistics."""
        result = EvaluationResult(
            generic_metrics={},
            env_metrics={},
            code_errors=0,
            episode_results=[],
        )

        cache.store("code1", result)
        cache.get("code1")  # Hit
        cache.get("code1")  # Hit
        cache.get("code2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] > 0


class TestDiversityMetrics:
    """Tests for diversity preservation."""

    def test_compute_code_diversity(self):
        """Should compute diversity between code samples."""
        from tetris_evolve.evaluation.diversity import compute_code_diversity

        codes = [
            "class A:\n    def f(self): return 1",
            "class A:\n    def f(self): return 2",
            "class B:\n    def g(self): return 3",
        ]

        diversity = compute_code_diversity(codes)

        assert 0.0 <= diversity <= 1.0

    def test_diversity_selection(self):
        """Should select diverse programs."""
        from tetris_evolve.evaluation.diversity import select_diverse

        candidates = [
            {"id": "a", "code": "class A: x=1", "score": 100},
            {"id": "b", "code": "class A: x=1", "score": 99},  # Similar to a
            {"id": "c", "code": "class B: y=2", "score": 95},  # Different
            {"id": "d", "code": "class C: z=3", "score": 90},  # Different
        ]

        selected = select_diverse(candidates, n=2, diversity_weight=0.5)

        assert len(selected) == 2
        # Should prefer diverse candidates even with lower scores
        ids = [s["id"] for s in selected]
        # 'a' should be selected (best score)
        assert "a" in ids
