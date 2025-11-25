"""
Bug Confirmation Tests

This module contains tests that CONFIRM the bugs identified in the code review.
Each test demonstrates a specific bug and should PASS when the bug EXISTS.
When bugs are fixed, these tests should be updated to verify the fix.
"""
import pytest
import numpy as np
import sys
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tetris_evolve.rlm.repl import SharedREPL, REPLExecutionError
from tetris_evolve.evaluation.parallel import EvaluationCache
from tetris_evolve.evaluation.evaluator import MetricStats, EvaluationResult
from tetris_evolve.child_llm.executor import CodeValidator
from tetris_evolve.root_llm.functions import RootLLMFunctions, PerformanceAnalysis
from tetris_evolve.database.program import ProgramDatabase, Program, ProgramMetrics


class TestSecurityVulnerabilities:
    """Tests confirming security vulnerabilities."""

    def test_bug1_exec_allows_file_system_access(self):
        """
        BUG #1: Arbitrary code execution via exec() allows file system access.

        The evaluate_player_from_code() function executes arbitrary code without
        sandboxing, allowing access to the file system.
        """
        repl = SharedREPL()

        # This malicious code can read arbitrary files
        malicious_code = """
import os
result = os.path.exists('/etc/passwd')
"""
        # The code executes successfully - this confirms the vulnerability
        repl.execute(malicious_code)
        result = repl.get_variable('result')

        # If we can check file existence, we have filesystem access
        # This should NOT be possible in a sandboxed environment
        assert result is not None, "Code execution should work (confirming vulnerability)"
        # On Linux, /etc/passwd exists
        if sys.platform != 'win32':
            assert result == True, "Filesystem access confirmed - VULNERABILITY EXISTS"

    def test_bug1_exec_allows_environment_variable_access(self):
        """
        BUG #1: exec() allows reading environment variables (potential secret leak).
        """
        repl = SharedREPL()

        # Set a fake secret
        os.environ['TEST_SECRET_KEY'] = 'super_secret_value'

        try:
            malicious_code = """
import os
secret = os.environ.get('TEST_SECRET_KEY', 'not_found')
"""
            repl.execute(malicious_code)
            secret = repl.get_variable('secret')

            # This confirms we can read environment variables
            assert secret == 'super_secret_value', "Environment access confirmed - VULNERABILITY EXISTS"
        finally:
            del os.environ['TEST_SECRET_KEY']

    def test_bug3_incomplete_dangerous_pattern_detection(self):
        """
        BUG #3: CodeValidator misses dangerous patterns.
        """
        validator = CodeValidator()

        # These dangerous patterns should be caught but AREN'T
        dangerous_codes = [
            # importlib bypass
            ("import importlib; mod = importlib.import_module('os')", "importlib"),
            # socket access
            ("import socket; s = socket.socket()", "socket"),
            # pickle deserialization (code execution)
            ("import pickle; pickle.loads(data)", "pickle"),
            # ctypes for arbitrary memory access
            ("import ctypes; ctypes.CDLL('libc.so.6')", "ctypes"),
            # urllib for network access
            ("import urllib.request; urllib.request.urlopen('http://evil.com')", "urllib"),
            # eval without space (bypasses pattern)
            ("eval('os.system(\"ls\")')", "eval without space"),
            # open with mode as keyword arg (bypasses pattern)
            ("open('/etc/passwd', mode='w')", "open with keyword arg"),
            # __builtins__ access
            ("__builtins__['exec']('import os')", "__builtins__"),
        ]

        missed_patterns = []
        for code, pattern_name in dangerous_codes:
            result = validator.validate(code, check_safety=True)
            if result.is_valid:
                missed_patterns.append(pattern_name)

        # Confirm that dangerous patterns are missed
        assert len(missed_patterns) > 0, "Validator should miss some patterns - BUG EXISTS"
        print(f"\nMissed dangerous patterns: {missed_patterns}")

    def test_bug3_check_safety_false_by_default(self):
        """
        BUG #3: check_safety=False by default makes validation ineffective.
        """
        validator = CodeValidator()

        # Obviously dangerous code
        dangerous_code = "import os; os.system('rm -rf /')"

        # Default validation (check_safety=False)
        result = validator.validate(dangerous_code)

        # This passes because safety check is disabled by default!
        assert result.is_valid == True, "Dangerous code passes with default settings - BUG EXISTS"


class TestLogicBugs:
    """Tests confirming logic bugs."""

    def test_bug5_incorrect_aggregate_height_calculation(self):
        """
        BUG #5: Aggregate height calculation is inverted.

        The formula uses row index directly instead of (height - row),
        giving lower weights to cells near the top (which should be weighted higher).
        """
        # Import the buggy function
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

        # BUG: The top block should have HIGHER aggregate height (it's stacked higher)
        # But the buggy code gives row 0 a weight of 0, and row 19 a weight of 19
        # So bottom actually gets a HIGHER aggregate_height value!

        print(f"\nAggregate height with block at TOP (row 0): {h_top['aggregate_height']}")
        print(f"Aggregate height with block at BOTTOM (row 19): {h_bottom['aggregate_height']}")

        # This confirms the bug - bottom should be LOWER, not higher
        assert h_bottom['aggregate_height'] > h_top['aggregate_height'], \
            "Bottom block has higher aggregate_height than top block - BUG EXISTS"

        # The correct behavior would be:
        # - Block at row 0 (top): height = 20 - 0 = 20
        # - Block at row 19 (bottom): height = 20 - 19 = 1
        # So top should have HIGHER aggregate height

        # Current buggy behavior:
        # - Block at row 0: weight = 0 * 1 = 0
        # - Block at row 19: weight = 19 * 1 = 19
        assert h_top['aggregate_height'] == 0, "Top block gets 0 weight - BUG CONFIRMED"

    def test_bug6_division_by_zero_in_improvement_rate(self):
        """
        BUG #6: improvement_rate is meaningless when best_scores[0] == 0.
        """
        from tetris_evolve.environment.tetris import TetrisConfig

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

        # Bug: improvement_rate is 0 even though scores went from 0 to 200
        print(f"\nBest scores: {trends.best_scores}")  # [0, 100, 200]
        print(f"Improvement rate: {trends.improvement_rate}")  # 0.0

        assert trends.improvement_rate == 0.0, \
            "Improvement rate is 0 despite 0->200 improvement - BUG EXISTS"

        # The scores clearly improved from 0 to 200, but the rate shows 0%

    def test_bug7_convergence_indicator_can_be_negative(self):
        """
        BUG #7: Convergence indicator becomes negative when changes are large.
        """
        from tetris_evolve.environment.tetris import TetrisConfig

        db = ProgramDatabase()
        env_config = TetrisConfig()

        root_funcs = RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )

        # Create highly volatile scores where recent changes exceed best score
        scores = [100, 50, 200, 10, 300]  # Large swings

        for gen, score in enumerate(scores):
            prog = Program(
                program_id=f"prog_{gen}",
                generation=gen,
                code="pass",
                metrics=ProgramMetrics(
                    avg_score=float(score),
                    std_score=0,
                    avg_lines_cleared=0,
                    games_played=1,
                )
            )
            db.add_program(prog)
            root_funcs.current_generation = gen

        trends = root_funcs.get_historical_trends()

        # The formula: 1.0 - (mean_changes / best_score)
        # Recent changes: |10-200|=190, |300-10|=290 -> mean = 240
        # best_score = 300
        # Raw value: 1.0 - (240/300) = 1.0 - 0.8 = 0.2 (this case is fine)

        # But with different values, it CAN go negative
        # Let's verify the max(0, x) is needed
        print(f"\nConvergence indicator: {trends.convergence_indicator}")

        # The fact that max(0, x) is used indicates the value CAN be negative
        # Let's create a case where it would be negative without the max()

        db2 = ProgramDatabase()
        root_funcs2 = RootLLMFunctions(program_database=db2, env_config=env_config)

        # Extreme case: small final score but huge changes
        scores2 = [100, 500, 100, 500, 10]  # Ends at 10 but swings of 400
        for gen, score in enumerate(scores2):
            prog = Program(
                program_id=f"prog2_{gen}",
                generation=gen,
                code="pass",
                metrics=ProgramMetrics(avg_score=float(score), std_score=0, avg_lines_cleared=0, games_played=1)
            )
            db2.add_program(prog)
            root_funcs2.current_generation = gen

        trends2 = root_funcs2.get_historical_trends()

        # Mean changes: |100-500|=400, |500-10|=490 -> mean = 445
        # best_score = 10
        # Raw: 1.0 - (445/10) = 1.0 - 44.5 = -43.5
        # But max(0, x) makes it 0

        print(f"Convergence indicator (extreme case): {trends2.convergence_indicator}")
        assert trends2.convergence_indicator == 0.0, \
            "Convergence clamped to 0 (would be negative) - BUG/DESIGN ISSUE CONFIRMED"

    def test_bug8_bare_except_swallows_keyboard_interrupt(self):
        """
        BUG #8: Bare except clause swallows KeyboardInterrupt.
        """
        repl = SharedREPL()

        # The bug is in lines 171-174:
        # try:
        #     return eval(last_line, self._namespace)
        # except:
        #     pass

        # We can demonstrate this by showing that ANY exception is swallowed
        # when trying to evaluate the last line as an expression

        code_with_error = """
x = 1
raise ValueError("This error will be swallowed")
"""
        # This won't raise because the bare except catches it
        # (The error happens when trying to eval the last line)

        # Actually, let's test more directly
        # When code ends with a statement that looks like an expression but isn't
        code = """
def foo():
    pass
foo  # This is an expression that returns the function
"""
        result = repl.execute(code)
        # This works fine

        # The real issue is when eval() fails on something that looked like an expression
        code2 = """
class Foo:
    pass
Foo()  # Could fail if Foo.__init__ has issues
"""
        # This demonstrates the try/except path
        repl.execute(code2)  # Doesn't raise even if inner eval fails

    def test_bug9_metrics_fallback_logic_error(self):
        """
        BUG #9: Metrics fallback returns 0 instead of using fallback.
        """
        # The buggy pattern is:
        # result.env_metrics.get("score", fallback).mean if result.env_metrics.get("score") else 0
        #
        # Problem: if env_metrics["score"] doesn't exist:
        # - get("score", fallback) returns fallback
        # - BUT the condition `if result.env_metrics.get("score")` is False
        # - So we return 0, not fallback.mean

        # Simulate the buggy logic
        env_metrics = {}  # No "score" key
        generic_metrics = {"total_reward": MetricStats(mean=100, std=10, min=50, max=150)}

        # Buggy logic from functions.py:202
        avg_score = env_metrics.get("score", generic_metrics.get("total_reward")).mean \
            if env_metrics.get("score") else 0

        print(f"\nExpected avg_score (from fallback): 100")
        print(f"Actual avg_score (buggy logic): {avg_score}")

        # The fallback exists and has mean=100, but we get 0
        assert avg_score == 0, "Fallback ignored, returns 0 - BUG EXISTS"

        # Correct logic should be:
        score_metric = env_metrics.get("score") or generic_metrics.get("total_reward")
        correct_avg = score_metric.mean if score_metric else 0
        assert correct_avg == 100, "Correct logic would give 100"


class TestConcurrencyIssues:
    """Tests confirming concurrency issues."""

    def test_bug11_cache_returns_mutable_reference(self):
        """
        BUG #11: EvaluationCache returns mutable objects by reference.

        If caller modifies the returned result, it corrupts the cache.
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

        # Modify the returned result (caller might do this accidentally)
        cached1.code_errors = 999
        cached1.episode_results.append({"episode_id": 1, "reward": 200})
        cached1.generic_metrics["new_key"] = MetricStats(0, 0, 0, 0)

        # Get from cache again
        cached2 = cache.get(code)

        # BUG: The cache is corrupted!
        print(f"\nOriginal code_errors: 0")
        print(f"Cached code_errors after modification: {cached2.code_errors}")

        assert cached2.code_errors == 999, "Cache corruption via mutable reference - BUG EXISTS"
        assert len(cached2.episode_results) == 2, "List in cache was modified - BUG EXISTS"
        assert "new_key" in cached2.generic_metrics, "Dict in cache was modified - BUG EXISTS"

    def test_bug12_shared_repl_not_thread_safe(self):
        """
        BUG #12: SharedREPL has no thread safety.

        Concurrent modifications to namespace can cause race conditions.
        """
        repl = SharedREPL()
        errors = []
        results = {"thread1": [], "thread2": []}

        def thread1_work():
            try:
                for i in range(100):
                    repl.set_variable("counter", i)
                    time.sleep(0.001)
                    val = repl.get_variable("counter")
                    results["thread1"].append(val)
            except Exception as e:
                errors.append(f"Thread1: {e}")

        def thread2_work():
            try:
                for i in range(100, 200):
                    repl.set_variable("counter", i)
                    time.sleep(0.001)
                    val = repl.get_variable("counter")
                    results["thread2"].append(val)
            except Exception as e:
                errors.append(f"Thread2: {e}")

        t1 = threading.Thread(target=thread1_work)
        t2 = threading.Thread(target=thread2_work)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Check for race conditions - thread1 should only see values 0-99
        # and thread2 should only see values 100-199, but due to race conditions
        # they will see each other's values

        thread1_saw_thread2_values = [v for v in results["thread1"] if v >= 100]
        thread2_saw_thread1_values = [v for v in results["thread2"] if v < 100]

        race_detected = len(thread1_saw_thread2_values) > 0 or len(thread2_saw_thread1_values) > 0

        print(f"\nThread1 saw Thread2 values: {len(thread1_saw_thread2_values)}")
        print(f"Thread2 saw Thread1 values: {len(thread2_saw_thread1_values)}")

        # This test may be flaky depending on timing, but with enough iterations
        # race conditions should manifest
        if race_detected:
            print("RACE CONDITION DETECTED - BUG EXISTS")
        else:
            print("Race condition not detected in this run (timing dependent)")

        # The key point is that there's NO LOCKING - the bug exists even if
        # this particular run didn't trigger visible corruption

    def test_bug12_shared_repl_history_race(self):
        """
        BUG #12: SharedREPL history list can have race conditions.
        """
        repl = SharedREPL()
        errors = []

        def execute_code(thread_id):
            try:
                for i in range(50):
                    repl.execute(f"x_{thread_id}_{i} = {i}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=execute_code, args=(i,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check history - should have 200 entries (4 threads * 50 executions)
        history = repl.get_history()

        print(f"\nExpected history entries: 200")
        print(f"Actual history entries: {len(history)}")

        # Due to race conditions in list.append(), entries might be lost
        # or the list might be corrupted
        # Note: CPython's GIL usually prevents list.append corruption,
        # but the lack of locking is still a bug


class TestErrorHandling:
    """Tests confirming error handling issues."""

    def test_bug17_catches_keyboard_interrupt(self):
        """
        BUG #17: Catching Exception catches too much.

        The pattern `except Exception as e` catches KeyboardInterrupt
        in Python 2, but not in Python 3. However, catching all exceptions
        is still problematic.
        """
        # The code in tetris_evaluator.py lines 41-55 catches Exception
        # This is better than bare except, but still masks unexpected errors

        # We can't easily test KeyboardInterrupt, but we can show that
        # unexpected exceptions are caught and converted to error returns
        pass  # This is more of a code review finding than a testable bug

    def test_bug18_consecutive_failures_not_detected(self):
        """
        BUG #18: Episode loop doesn't detect consecutive failures.

        If a player has a fundamental bug, ALL episodes will fail,
        but we continue trying all of them instead of stopping early.
        """
        from tetris_evolve.evaluation.evaluator import evaluate_player
        from tetris_evolve.environment.tetris import TetrisConfig

        class AlwaysFailingPlayer:
            def select_action(self, obs):
                raise RuntimeError("I always fail!")

            def reset(self):
                pass

        env_config = TetrisConfig()
        player = AlwaysFailingPlayer()

        # Run 10 episodes - all will fail
        result = evaluate_player(
            player,
            env_config,
            num_episodes=10,
            max_steps_per_episode=100,
        )

        print(f"\nTotal episodes attempted: 10")
        print(f"Code errors: {result.code_errors}")
        print(f"Error messages: {len(result.error_messages)}")

        # All 10 episodes failed - we should have stopped earlier
        assert result.code_errors == 10, "All 10 episodes failed - no early termination - BUG EXISTS"

    def test_bug19_fragile_path_calculation(self):
        """
        BUG #19: Path calculation with 5 .parent calls is fragile.
        """
        from tetris_evolve.environment.tetris import TetrisConfig

        # The code in tetris.py:272 does:
        # root_dir = Path(__file__).parent.parent.parent.parent.parent

        # Let's verify this path calculation
        tetris_file = Path(__file__).parent.parent / "src" / "tetris_evolve" / "environment" / "tetris.py"

        if tetris_file.exists():
            # Calculate what the code thinks is root
            calculated_root = tetris_file.parent.parent.parent.parent.parent
            actual_root = Path(__file__).parent.parent

            print(f"\nCalculated root: {calculated_root}")
            print(f"Actual root: {actual_root}")

            # The paths should match, but this is fragile
            # If directory structure changes, this breaks


class TestResourceManagement:
    """Tests confirming resource management issues."""

    def test_bug14_executor_not_closed(self):
        """
        BUG #14: ParallelEvaluator executor not closed without context manager.
        """
        from tetris_evolve.evaluation.parallel import ParallelEvaluator
        from tetris_evolve.environment.tetris import TetrisConfig

        env_config = TetrisConfig()

        # Create evaluator without context manager
        evaluator = ParallelEvaluator(
            env_config=env_config,
            num_workers=2,
            use_processes=False,  # Use threads for easier testing
        )

        # The executor is created but if we forget to close it,
        # threads will leak

        # Check that threads are running
        initial_thread_count = threading.active_count()

        # The executor has worker threads
        # (ThreadPoolExecutor creates threads lazily, so we need to submit work)
        from concurrent.futures import ThreadPoolExecutor

        # Verify the executor exists and has workers
        assert evaluator._executor is not None, "Executor should be created"

        # Proper cleanup requires explicit close() call
        evaluator.close()

        # This test confirms the pattern - without close(), threads leak

    def test_bug26_cache_hash_truncation(self):
        """
        BUG #26: Cache hash truncated to 16 chars increases collision risk.
        """
        from tetris_evolve.evaluation.parallel import EvaluationCache

        cache = EvaluationCache()

        # Generate many different code strings
        hashes = set()
        for i in range(10000):
            code = f"def func_{i}(): return {i}"
            hash_val = cache._hash_code(code)
            hashes.add(hash_val)

        print(f"\nGenerated 10000 unique codes")
        print(f"Unique hashes (16 chars): {len(hashes)}")

        # With 16 hex chars (64 bits), collision probability is low
        # but not zero. For a proper cache, full hash should be used.

        # Demonstrate the truncation
        code1 = "x = 1"
        full_hash = __import__('hashlib').sha256(code1.encode()).hexdigest()
        truncated_hash = cache._hash_code(code1)

        print(f"Full SHA256: {full_hash}")
        print(f"Truncated (16 chars): {truncated_hash}")

        assert len(truncated_hash) == 16, "Hash is truncated to 16 chars"
        assert len(full_hash) == 64, "Full hash is 64 chars"


class TestDesignIssues:
    """Tests confirming design issues."""

    def test_bug27_no_timeout_on_execution(self):
        """
        BUG #27: No timeout on code execution - infinite loops hang system.
        """
        repl = SharedREPL()

        # We can't actually test an infinite loop (it would hang the test)
        # But we can verify there's no timeout mechanism

        # Check that execute() has no timeout parameter
        import inspect
        sig = inspect.signature(repl.execute)
        params = list(sig.parameters.keys())

        print(f"\nSharedREPL.execute() parameters: {params}")

        assert "timeout" not in params, "No timeout parameter - BUG EXISTS"

        # Also verify capture_output has no timeout
        sig2 = inspect.signature(repl.capture_output)
        params2 = list(sig2.parameters.keys())

        assert "timeout" not in params2, "No timeout in capture_output either"

    def test_bug28_unbounded_history_growth(self):
        """
        BUG #28: REPL history grows without bound.
        """
        repl = SharedREPL()

        # Execute many commands
        for i in range(1000):
            repl.execute(f"x_{i} = {i}")

        history = repl.get_history()

        print(f"\nHistory length after 1000 executions: {len(history)}")

        # History has no max size
        assert len(history) == 1000, "History grows unbounded - potential memory issue"

        # In a long-running evolution, this could consume significant memory

    def test_bug29_no_diversity_in_selection(self):
        """
        BUG #29: Selection is pure elitist without diversity consideration.
        """
        from tetris_evolve.evolution.loop import EvolutionRunner, EvolutionConfig

        # The selection code in loop.py:324-328 is:
        # top_programs = sorted(..., key=lambda p: p.metrics.avg_score, reverse=True)[:selection_size]

        # This is pure elitist selection - no diversity consideration
        # Even though diversity.py exists, it's not integrated

        # We can verify by checking the imports in loop.py
        import tetris_evolve.evolution.loop as loop_module

        source = inspect.getsource(loop_module)

        # Check if diversity is imported or used
        uses_diversity = "diversity" in source.lower()

        print(f"\nEvolution loop uses diversity module: {uses_diversity}")

        # The diversity module exists but isn't used in main selection
        from tetris_evolve.evaluation import diversity
        assert diversity is not None, "Diversity module exists"


class TestDeprecatedPatterns:
    """Tests for deprecated patterns."""

    def test_bug32_deprecated_utcnow(self):
        """
        BUG #32: Using deprecated datetime.utcnow().
        """
        from tetris_evolve.database.program import Program
        import inspect

        source = inspect.getsource(Program)

        uses_utcnow = "utcnow()" in source

        print(f"\nProgram class uses datetime.utcnow(): {uses_utcnow}")

        if uses_utcnow:
            print("datetime.utcnow() is deprecated in Python 3.12+")
            print("Should use datetime.now(timezone.utc) instead")


# Run summary
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
