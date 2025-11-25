"""
Parallel evaluation and caching for improved performance.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import copy
import hashlib
from collections import OrderedDict
import threading

from .evaluator import evaluate_player, EvaluationResult, Player
from ..environment.base import EnvironmentConfig


class EvaluationCache:
    """
    Cache for evaluation results to avoid re-evaluating identical code.

    Uses an LRU eviction policy when max_size is reached.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, EvaluationResult] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _hash_code(self, code: str) -> str:
        """Create a hash of the code for use as cache key."""
        # FIXED: Use longer hash (32 chars = 128 bits) to reduce collision risk
        return hashlib.sha256(code.encode()).hexdigest()[:32]

    def get(self, code: str) -> Optional[EvaluationResult]:
        """
        Get cached result for code.

        Args:
            code: Source code to look up

        Returns:
            A deep copy of the cached EvaluationResult or None if not found
        """
        key = self._hash_code(code)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                # FIXED: Return a deep copy to prevent mutation of cached data
                return copy.deepcopy(self._cache[key])
            else:
                self._misses += 1
                return None

    def store(self, code: str, result: EvaluationResult) -> None:
        """
        Store a result in the cache.

        Args:
            code: Source code
            result: Evaluation result
        """
        key = self._hash_code(code)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                # FIXED: Store a deep copy to prevent external mutation
                self._cache[key] = copy.deepcopy(result)
            else:
                if len(self._cache) >= self.max_size:
                    # Remove oldest (first) item
                    self._cache.popitem(last=False)
                # FIXED: Store a deep copy to prevent external mutation
                self._cache[key] = copy.deepcopy(result)

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class ParallelEvaluator:
    """
    Parallel evaluation of multiple players.

    Uses thread or process pools to evaluate players concurrently.
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        num_workers: int = 4,
        use_processes: bool = False,
        cache: Optional[EvaluationCache] = None,
    ):
        """
        Initialize parallel evaluator.

        Args:
            env_config: Environment configuration
            num_workers: Number of worker threads/processes
            use_processes: Whether to use processes instead of threads
            cache: Optional evaluation cache
        """
        self.env_config = env_config
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.cache = cache

        # Create executor
        if use_processes:
            self._executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def evaluate(
        self,
        player: Player,
        num_episodes: int = 10,
        max_steps_per_episode: int = 10000,
    ) -> EvaluationResult:
        """
        Evaluate a single player.

        Args:
            player: Player to evaluate
            num_episodes: Number of episodes
            max_steps_per_episode: Max steps per episode

        Returns:
            Evaluation result
        """
        return evaluate_player(
            player,
            self.env_config,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )

    def evaluate_batch(
        self,
        players: List[Player],
        num_episodes: int = 10,
        max_steps_per_episode: int = 10000,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple players in parallel.

        Args:
            players: List of players to evaluate
            num_episodes: Number of episodes per player
            max_steps_per_episode: Max steps per episode

        Returns:
            List of evaluation results (same order as input)
        """
        def eval_one(player):
            try:
                return evaluate_player(
                    player,
                    self.env_config,
                    num_episodes=num_episodes,
                    max_steps_per_episode=max_steps_per_episode,
                )
            except Exception as e:
                # Return error result
                from .evaluator import MetricStats
                return EvaluationResult(
                    generic_metrics={
                        "total_reward": MetricStats(0, 0, 0, 0),
                        "episode_length": MetricStats(0, 0, 0, 0),
                    },
                    env_metrics={},
                    code_errors=1,
                    episode_results=[],
                    error_messages=[str(e)],
                )

        # Submit all tasks
        futures = [self._executor.submit(eval_one, p) for p in players]

        # Collect results in order
        results = [f.result() for f in futures]
        return results

    def close(self):
        """Shut down the executor."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
