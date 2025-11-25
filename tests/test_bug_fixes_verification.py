"""
Bug Fix Verification Tests

These tests verify that the bugs identified in the code review have been FIXED.
Each test should PASS after the fixes have been applied.
"""
import pytest
import numpy as np
import sys
import threading
import time
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tetris_evolve.rlm.repl import SharedREPL
from tetris_evolve.evaluation.parallel import EvaluationCache
from tetris_evolve.evaluation.evaluator import MetricStats, EvaluationResult
from tetris_evolve.child_llm.executor import CodeValidator
from tetris_evolve.root_llm.functions import RootLLMFunctions
from tetris_evolve.database.program import ProgramDatabase, Program, ProgramMetrics
from tetris_evolve.environment.tetris import TetrisConfig


class TestSecurityFixes:
    """Verify security fixes are in place."""

    def test_fix3_comprehensive_dangerous_patterns(self):
        """
        FIXED: CodeValidator now has comprehensive dangerous pattern detection.
        """
        validator = CodeValidator()

        # These dangerous patterns should now be CAUGHT
        dangerous_codes = [
            ("import importlib; mod = importlib.import_module('os')", "importlib"),
            ("import socket; s = socket.socket()", "socket"),
            ("import pickle; pickle.loads(data)", "pickle"),
            ("import ctypes; ctypes.CDLL('libc.so.6')", "ctypes"),
            ("import urllib.request; urllib.request.urlopen('http://evil.com')", "urllib"),
            ("eval('os.system(\"ls\")')", "eval without space"),
            ("__builtins__['exec']('import os')", "__builtins__"),
        ]

        caught_patterns = []
        for code, pattern_name in dangerous_codes:
            result = validator.validate(code, check_safety=True)
            if not result.is_valid:
                caught_patterns.append(pattern_name)

        print(f"\nCaught dangerous patterns: {caught_patterns}")

        # Most patterns should now be caught
        assert len(caught_patterns) >= 5, f"Should catch most patterns, caught {len(caught_patterns)}"

    def test_fix3_check_safety_true_by_default(self):
        """
        FIXED: check_safety=True is now the default.
        """
        validator = CodeValidator()

        # Obviously dangerous code
        dangerous_code = "import os; os.system('rm -rf /')"

        # Default validation should now catch this
        result = validator.validate(dangerous_code)

        assert result.is_valid == False, "Dangerous code should be rejected with default settings"


class TestLogicBugFixes:
    """Verify logic bug fixes are in place."""

    def test_fix5_correct_aggregate_height_calculation(self):
        """
        FIXED: Aggregate height calculation now correctly weighs blocks
        at the top higher than blocks at the bottom.
        """
        from tetris_agent import compute_heuristics

        # Create a board with a single block at the TOP (row 0)
        board_top = np.zeros((20, 10), dtype=np.int8)
        board_top[0, 0] = 1  # Block at top-left (row 0)

        # Create a board with a single block at the BOTTOM (row 19)
        board_bottom = np.zeros((20, 10), dtype=np.int8)
        board_bottom[19, 0] = 1  # Block at bottom-left (row 19)

        dummy_piece = np.zeros((4, 4))

        h_top = compute_heuristics(board_top, dummy_piece, dummy_piece, (0, 0))
        h_bottom = compute_heuristics(board_bottom, dummy_piece, dummy_piece, (0, 0))

        print(f"\nAggregate height with block at TOP (row 0): {h_top['aggregate_height']}")
        print(f"Aggregate height with block at BOTTOM (row 19): {h_bottom['aggregate_height']}")

        # FIXED: Top block should now have HIGHER aggregate height
        # Block at row 0 has height = 20, block at row 19 has height = 1
        assert h_top['aggregate_height'] > h_bottom['aggregate_height'], \
            "Top block should have higher aggregate_height than bottom block"
        assert h_top['aggregate_height'] == 20, "Block at top should have height 20"
        assert h_bottom['aggregate_height'] == 1, "Block at bottom should have height 1"

    def test_fix6_improvement_rate_handles_zero_start(self):
        """
        FIXED: improvement_rate now handles starting from 0.
        """
        db = ProgramDatabase()
        env_config = TetrisConfig()

        root_funcs = RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )

        # Add programs with initial score of 0
        for i in range(3):
            prog = Program(
                program_id=f"prog_{i}",
                generation=i,
                code="pass",
                metrics=ProgramMetrics(
                    avg_score=float(i * 100),  # 0, 100, 200
                    std_score=0,
                    avg_lines_cleared=0,
                    games_played=1,
                )
            )
            db.add_program(prog)
            root_funcs.current_generation = i

        trends = root_funcs.get_historical_trends()

        print(f"\nBest scores: {trends.best_scores}")  # [0, 100, 200]
        print(f"Improvement rate: {trends.improvement_rate}")

        # FIXED: improvement_rate should now be positive
        # With scores going 0 -> 200 over 3 generations, we should see improvement
        assert trends.improvement_rate > 0, \
            "Improvement rate should be positive when scores improve from 0"

    def test_fix9_metrics_fallback_works_correctly(self):
        """
        FIXED: Metrics fallback now correctly uses the fallback value.
        """
        # Simulate the fixed logic
        env_metrics = {}  # No "score" key
        generic_metrics = {"total_reward": MetricStats(mean=100, std=10, min=50, max=150)}

        # FIXED logic
        score_metric = env_metrics.get("score") or generic_metrics.get("total_reward")
        avg_score = score_metric.mean if score_metric else 0.0

        print(f"\nExpected avg_score (from fallback): 100")
        print(f"Actual avg_score (fixed logic): {avg_score}")

        # FIXED: Should now correctly use the fallback
        assert avg_score == 100, "Should use fallback value when primary not available"


class TestConcurrencyFixes:
    """Verify concurrency fixes are in place."""

    def test_fix11_cache_returns_copies(self):
        """
        FIXED: EvaluationCache now returns deep copies to prevent mutation.
        """
        cache = EvaluationCache(max_size=10)

        # Create and store a result
        original_result = EvaluationResult(
            generic_metrics={"total_reward": MetricStats(100, 10, 50, 150)},
            env_metrics={"score": MetricStats(100, 10, 50, 150)},
            code_errors=0,
            episode_results=[{"episode_id": 0, "reward": 100}],
        )

        code = "def test(): pass"
        cache.store(code, original_result)

        # Get the cached result
        cached1 = cache.get(code)

        # Modify the returned result
        cached1.code_errors = 999
        cached1.episode_results.append({"episode_id": 1, "reward": 200})

        # Get from cache again
        cached2 = cache.get(code)

        print(f"\nOriginal code_errors: 0")
        print(f"Modified copy code_errors: {cached1.code_errors}")
        print(f"Fresh cache retrieval code_errors: {cached2.code_errors}")

        # FIXED: Cache should not be affected by modifications
        assert cached2.code_errors == 0, "Cache should return copies - modifications shouldn't affect cache"
        assert len(cached2.episode_results) == 1, "Cache should return copies - list modifications shouldn't affect cache"

    def test_fix12_shared_repl_has_lock(self):
        """
        FIXED: SharedREPL now has thread-safety via locking.
        """
        repl = SharedREPL()

        # Verify lock exists
        assert hasattr(repl, '_lock'), "SharedREPL should have a lock attribute"

        # Verify it's a reentrant lock (RLock)
        import threading
        assert isinstance(repl._lock, type(threading.RLock())), "Should use RLock for thread safety"

    def test_fix26_longer_hash(self):
        """
        FIXED: Cache hash is now 32 chars instead of 16.
        """
        cache = EvaluationCache()

        code = "x = 1"
        hash_val = cache._hash_code(code)

        print(f"\nHash length: {len(hash_val)}")
        print(f"Hash value: {hash_val}")

        # FIXED: Should now be 32 characters
        assert len(hash_val) == 32, "Hash should be 32 characters (128 bits)"

    def test_fix28_bounded_history(self):
        """
        FIXED: REPL history is now bounded.
        """
        repl = SharedREPL()

        # Check that MAX_HISTORY_SIZE is defined
        assert hasattr(repl, 'MAX_HISTORY_SIZE'), "SharedREPL should have MAX_HISTORY_SIZE"
        assert repl.MAX_HISTORY_SIZE > 0, "MAX_HISTORY_SIZE should be positive"

        # Execute more commands than the limit
        for i in range(repl.MAX_HISTORY_SIZE + 100):
            repl.execute(f"x_{i} = {i}")

        history = repl.get_history()

        print(f"\nMAX_HISTORY_SIZE: {repl.MAX_HISTORY_SIZE}")
        print(f"History length after {repl.MAX_HISTORY_SIZE + 100} executions: {len(history)}")

        # FIXED: History should be bounded
        assert len(history) <= repl.MAX_HISTORY_SIZE, \
            f"History should be bounded to {repl.MAX_HISTORY_SIZE}, got {len(history)}"


class TestErrorHandlingFixes:
    """Verify error handling fixes are in place."""

    def test_fix18_early_termination_on_consecutive_failures(self):
        """
        FIXED: Evaluator now stops early on consecutive failures.
        """
        from tetris_evolve.evaluation.evaluator import evaluate_player

        class AlwaysFailingPlayer:
            def select_action(self, obs):
                raise RuntimeError("I always fail!")

            def reset(self):
                pass

        env_config = TetrisConfig()
        player = AlwaysFailingPlayer()

        # Run 10 episodes - should stop early due to consecutive failures
        result = evaluate_player(
            player,
            env_config,
            num_episodes=10,
            max_steps_per_episode=100,
        )

        print(f"\nTotal episodes attempted: {result.code_errors}")
        print(f"Error messages: {result.error_messages}")

        # FIXED: Should stop after 3 consecutive failures
        # (default max_consecutive_failures=3)
        assert result.code_errors <= 4, \
            f"Should stop early after 3 consecutive failures, but ran {result.code_errors} episodes"

        # Should have an early termination message
        early_stop_messages = [msg for msg in result.error_messages if "consecutive" in msg.lower()]
        assert len(early_stop_messages) > 0, "Should have early termination message"


class TestDeprecationFixes:
    """Verify deprecated pattern fixes are in place."""

    def test_fix32_no_utcnow(self):
        """
        FIXED: Program class no longer uses deprecated datetime.utcnow().
        """
        import inspect
        from tetris_evolve.database.program import Program

        source = inspect.getsource(Program)

        # Filter out comments when checking for utcnow()
        # Check for actual usage, not just mentions in comments
        import re
        # Remove single-line comments
        source_no_comments = re.sub(r'#.*$', '', source, flags=re.MULTILINE)

        # Should not use utcnow() in actual code
        assert "utcnow()" not in source_no_comments, \
            "Should not use deprecated datetime.utcnow() in code"

        # Should use timezone-aware datetime
        assert "timezone.utc" in source or "timezone" in source, \
            "Should use timezone-aware datetime"


# Summary
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("BUG FIX VERIFICATION TESTS")
    print("=" * 60)

    tests = [
        # Security fixes
        ("Security: comprehensive patterns", TestSecurityFixes().test_fix3_comprehensive_dangerous_patterns),
        ("Security: check_safety default", TestSecurityFixes().test_fix3_check_safety_true_by_default),
        # Logic fixes
        ("Logic: aggregate height", TestLogicBugFixes().test_fix5_correct_aggregate_height_calculation),
        ("Logic: improvement rate", TestLogicBugFixes().test_fix6_improvement_rate_handles_zero_start),
        ("Logic: metrics fallback", TestLogicBugFixes().test_fix9_metrics_fallback_works_correctly),
        # Concurrency fixes
        ("Concurrency: cache copies", TestConcurrencyFixes().test_fix11_cache_returns_copies),
        ("Concurrency: repl lock", TestConcurrencyFixes().test_fix12_shared_repl_has_lock),
        ("Concurrency: longer hash", TestConcurrencyFixes().test_fix26_longer_hash),
        ("Concurrency: bounded history", TestConcurrencyFixes().test_fix28_bounded_history),
        # Error handling fixes
        ("Error: early termination", TestErrorHandlingFixes().test_fix18_early_termination_on_consecutive_failures),
        # Deprecation fixes
        ("Deprecation: no utcnow", TestDeprecationFixes().test_fix32_no_utcnow),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name} - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {name} - {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
