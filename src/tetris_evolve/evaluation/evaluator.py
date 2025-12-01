"""Program evaluator for safely executing and scoring evolved Tetris players."""

import ast
import signal
import traceback
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class GameResult:
    """Result from a single Tetris game."""

    score: int
    lines_cleared: int
    steps: int
    ep_return: float
    terminated_reason: str  # "game_over", "max_steps", "error"


@dataclass
class EvaluationResult:
    """Result from evaluating a program across multiple games."""

    trial_id: str
    success: bool
    error: Optional[str]
    games_played: int
    avg_score: Optional[float]
    avg_lines_cleared: Optional[float]
    avg_survival_steps: Optional[float]
    max_score: Optional[int]
    min_score: Optional[int]
    std_score: Optional[float]
    game_results: list[GameResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "trial_id": self.trial_id,
            "success": self.success,
            "error": self.error,
            "games_played": self.games_played,
            "avg_score": self.avg_score,
            "avg_lines_cleared": self.avg_lines_cleared,
            "avg_survival_steps": self.avg_survival_steps,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "std_score": self.std_score,
            "game_results": [
                {
                    "score": g.score,
                    "lines_cleared": g.lines_cleared,
                    "steps": g.steps,
                    "ep_return": g.ep_return,
                    "terminated_reason": g.terminated_reason,
                }
                for g in self.game_results
            ],
        }


# Blocked imports for safety
BLOCKED_IMPORTS = {
    "os",
    "subprocess",
    "sys",
    "socket",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "smtplib",
    "telnetlib",
    "pathlib",
    "shutil",
    "tempfile",
    "multiprocessing",
    "threading",
    "asyncio",
    "ctypes",
    "importlib",
}

# Blocked builtins for safety
BLOCKED_BUILTINS = {
    "eval",
    "exec",
    "__import__",
    "open",
    "compile",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
}


class TimeoutError(Exception):
    """Raised when code execution times out."""

    pass


class SafetyError(Exception):
    """Raised when code fails safety checks."""

    pass


class InterfaceError(Exception):
    """Raised when code doesn't implement required interface."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Execution timed out")


class ProgramEvaluator:
    """Safely executes and evaluates evolved Tetris player code."""

    def __init__(
        self,
        num_games: int = 10,
        max_steps: int = 10000,
        timeout_seconds: int = 30,
    ) -> None:
        """Initialize evaluator.

        Args:
            num_games: Number of games to play for evaluation.
            max_steps: Maximum steps per game.
            timeout_seconds: Timeout for entire evaluation.
        """
        self.num_games = num_games
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def evaluate(self, code: str, trial_id: str) -> EvaluationResult:
        """Evaluate code by running multiple Tetris games.

        Args:
            code: Python code containing choose_action function.
            trial_id: Unique identifier for this evaluation.

        Returns:
            EvaluationResult with success status and metrics.
        """
        # Validate syntax
        valid, error = self._validate_syntax(code)
        if not valid:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error=f"Syntax error: {error}",
                games_played=0,
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
            )

        # Check safety
        safe, error = self._check_safety(code)
        if not safe:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error=f"Safety violation: {error}",
                games_played=0,
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
            )

        # Validate interface
        valid, error = self._validate_interface(code)
        if not valid:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error=f"Interface error: {error}",
                games_played=0,
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
            )

        # Execute code to get player function
        try:
            player_fn = self._execute_code(code)
        except Exception as e:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error=f"Execution error: {e}",
                games_played=0,
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
            )

        # Run games
        game_results = []
        try:
            # Set up timeout for entire evaluation
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                for _ in range(self.num_games):
                    result = self._run_single_game(player_fn)
                    game_results.append(result)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error="Evaluation timed out",
                games_played=len(game_results),
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
                game_results=game_results,
            )
        except Exception as e:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error=f"Game execution error: {e}",
                games_played=len(game_results),
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
                game_results=game_results,
            )

        # Aggregate results
        return self._aggregate_results(trial_id, game_results)

    def _validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax.

        Args:
            code: Python code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _check_safety(self, code: str) -> tuple[bool, Optional[str]]:
        """Check code for potentially dangerous operations.

        Args:
            code: Python code to check.

        Returns:
            Tuple of (is_safe, error_message).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors handled elsewhere
            return True, None

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in BLOCKED_IMPORTS:
                        return False, f"Blocked import: {module}"

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in BLOCKED_IMPORTS:
                        return False, f"Blocked import: {module}"

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        return False, f"Blocked builtin: {node.func.id}"

            # Check attribute access for blocked builtins
            elif isinstance(node, ast.Attribute):
                if node.attr in {"__class__", "__bases__", "__subclasses__", "__mro__"}:
                    return False, f"Blocked attribute access: {node.attr}"

        return True, None

    def _validate_interface(self, code: str) -> tuple[bool, Optional[str]]:
        """Check that code implements required choose_action function.

        Args:
            code: Python code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, "Could not parse code"

        # Find function definitions
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node

        if "choose_action" not in functions:
            return False, "Missing required function: choose_action"

        func = functions["choose_action"]

        # Check it has at least one argument (obs)
        args = func.args
        total_args = len(args.args) + len(args.posonlyargs)
        if total_args < 1:
            return False, "choose_action must accept at least one argument (obs)"

        return True, None

    def _execute_code(self, code: str) -> Callable:
        """Execute code in restricted namespace and extract choose_action.

        Args:
            code: Python code to execute.

        Returns:
            The choose_action function.

        Raises:
            Exception: If execution fails.
        """
        # Create restricted namespace with safe builtins
        import random
        import math

        # Safe import function that only allows approved modules
        allowed_modules = {
            "numpy": np,
            "np": np,
            "random": random,
            "math": math,
        }

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in allowed_modules:
                return allowed_modules[name]
            # Check for submodules
            base_name = name.split('.')[0]
            if base_name in allowed_modules:
                return allowed_modules[base_name]
            raise ImportError(f"Import of '{name}' is not allowed")

        namespace = {
            "__builtins__": {
                # Allow safe builtins
                "__import__": safe_import,
                "__build_class__": __builtins__["__build_class__"] if isinstance(__builtins__, dict) else getattr(__builtins__, "__build_class__"),
                "__name__": "__main__",
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "pow": pow,
                "print": print,
                "range": range,
                "reversed": reversed,
                "round": round,
                "set": set,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "True": True,
                "False": False,
                "None": None,
                "isinstance": isinstance,
                "type": type,
                "object": object,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "super": super,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "IndexError": IndexError,
                "KeyError": KeyError,
            },
            # Pre-import allowed modules
            "np": np,
            "numpy": np,
            "random": random,
            "math": math,
        }

        exec(code, namespace)

        if "choose_action" not in namespace:
            raise InterfaceError("choose_action function not found after execution")

        return namespace["choose_action"]

    def _run_single_game(self, player_fn: Callable) -> GameResult:
        """Run a single Tetris game with the player function.

        Args:
            player_fn: Function that takes observation and returns action.

        Returns:
            GameResult with game statistics.
        """
        try:
            import pufferlib.environments.ocean.tetris as tetris_env
        except ImportError:
            # Fallback for testing without PufferLib
            return self._run_mock_game(player_fn)

        env = tetris_env.Tetris()
        obs, _ = env.reset()

        total_reward = 0.0
        steps = 0
        lines_cleared = 0
        terminated_reason = "max_steps"

        for step in range(self.max_steps):
            try:
                action = player_fn(obs)
                # Ensure action is valid integer
                action = int(action) % 7
            except Exception as e:
                return GameResult(
                    score=int(total_reward),
                    lines_cleared=lines_cleared,
                    steps=steps,
                    ep_return=total_reward,
                    terminated_reason=f"error: {e}",
                )

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Track lines cleared if available in info
            if "lines_cleared" in info:
                lines_cleared = info["lines_cleared"]

            if terminated:
                terminated_reason = "game_over"
                break

            if truncated:
                terminated_reason = "truncated"
                break

        env.close()

        return GameResult(
            score=int(total_reward),
            lines_cleared=lines_cleared,
            steps=steps,
            ep_return=total_reward,
            terminated_reason=terminated_reason,
        )

    def _run_mock_game(self, player_fn: Callable) -> GameResult:
        """Run a mock game for testing without PufferLib.

        Args:
            player_fn: Function that takes observation and returns action.

        Returns:
            GameResult with simulated statistics.
        """
        # Create fake observation (244-dim array like PufferLib Tetris)
        obs = np.zeros(244, dtype=np.float32)

        steps = 0
        total_reward = 0.0

        for step in range(min(100, self.max_steps)):
            try:
                action = player_fn(obs)
                action = int(action) % 7
            except Exception as e:
                return GameResult(
                    score=0,
                    lines_cleared=0,
                    steps=steps,
                    ep_return=0.0,
                    terminated_reason=f"error: {e}",
                )

            # Simulate some reward
            total_reward += np.random.uniform(0, 1)
            steps += 1

            # Random termination
            if np.random.random() < 0.01:
                break

        return GameResult(
            score=int(total_reward * 10),
            lines_cleared=int(total_reward / 10),
            steps=steps,
            ep_return=total_reward,
            terminated_reason="game_over",
        )

    def _aggregate_results(
        self, trial_id: str, game_results: list[GameResult]
    ) -> EvaluationResult:
        """Aggregate results from multiple games.

        Args:
            trial_id: Trial identifier.
            game_results: List of game results.

        Returns:
            Aggregated EvaluationResult.
        """
        if not game_results:
            return EvaluationResult(
                trial_id=trial_id,
                success=False,
                error="No games completed",
                games_played=0,
                avg_score=None,
                avg_lines_cleared=None,
                avg_survival_steps=None,
                max_score=None,
                min_score=None,
                std_score=None,
            )

        scores = [g.score for g in game_results]
        lines = [g.lines_cleared for g in game_results]
        steps = [g.steps for g in game_results]

        return EvaluationResult(
            trial_id=trial_id,
            success=True,
            error=None,
            games_played=len(game_results),
            avg_score=float(np.mean(scores)),
            avg_lines_cleared=float(np.mean(lines)),
            avg_survival_steps=float(np.mean(steps)),
            max_score=int(np.max(scores)),
            min_score=int(np.min(scores)),
            std_score=float(np.std(scores)),
            game_results=game_results,
        )
