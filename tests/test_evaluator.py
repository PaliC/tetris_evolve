"""Tests for Program Evaluator (Component 1)."""

import pytest
import numpy as np

from tetris_evolve.evaluation.evaluator import (
    ProgramEvaluator,
    EvaluationResult,
    GameResult,
    BLOCKED_IMPORTS,
    BLOCKED_BUILTINS,
)


class TestDataClasses:
    """Tests for data classes."""

    def test_game_result_dataclass(self):
        """GameResult creates correctly."""
        result = GameResult(
            score=1000,
            lines_cleared=10,
            steps=500,
            ep_return=150.5,
            terminated_reason="game_over",
        )

        assert result.score == 1000
        assert result.lines_cleared == 10
        assert result.steps == 500
        assert result.ep_return == 150.5

    def test_evaluation_result_dataclass(self):
        """EvaluationResult creates correctly."""
        result = EvaluationResult(
            trial_id="trial_001",
            success=True,
            error=None,
            games_played=10,
            avg_score=500.0,
            avg_lines_cleared=5.0,
            avg_survival_steps=200.0,
            max_score=800,
            min_score=200,
            std_score=150.0,
        )

        assert result.trial_id == "trial_001"
        assert result.success is True
        assert result.avg_score == 500.0

    def test_evaluation_result_to_dict(self):
        """EvaluationResult converts to dict correctly."""
        game = GameResult(
            score=100,
            lines_cleared=1,
            steps=50,
            ep_return=10.0,
            terminated_reason="game_over",
        )
        result = EvaluationResult(
            trial_id="trial_001",
            success=True,
            error=None,
            games_played=1,
            avg_score=100.0,
            avg_lines_cleared=1.0,
            avg_survival_steps=50.0,
            max_score=100,
            min_score=100,
            std_score=0.0,
            game_results=[game],
        )

        data = result.to_dict()

        assert data["trial_id"] == "trial_001"
        assert len(data["game_results"]) == 1


class TestSyntaxValidation:
    """Tests for syntax validation."""

    def test_syntax_valid(self):
        """Valid code passes syntax check."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    return 5
"""
        valid, error = evaluator._validate_syntax(code)

        assert valid is True
        assert error is None

    def test_syntax_invalid(self):
        """Invalid code fails syntax check."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs)
    return 5
"""
        valid, error = evaluator._validate_syntax(code)

        assert valid is False
        assert error is not None

    def test_syntax_with_imports(self):
        """Code with imports passes syntax check."""
        evaluator = ProgramEvaluator()

        code = """
import numpy as np

def choose_action(obs):
    return np.argmax(obs[:7])
"""
        valid, error = evaluator._validate_syntax(code)

        assert valid is True


class TestSafetyChecks:
    """Tests for safety validation."""

    def test_safety_blocks_os(self):
        """import os is blocked."""
        evaluator = ProgramEvaluator()

        code = """
import os

def choose_action(obs):
    os.system("ls")
    return 0
"""
        safe, error = evaluator._check_safety(code)

        assert safe is False
        assert "os" in error.lower()

    def test_safety_blocks_subprocess(self):
        """subprocess is blocked."""
        evaluator = ProgramEvaluator()

        code = """
import subprocess

def choose_action(obs):
    return 0
"""
        safe, error = evaluator._check_safety(code)

        assert safe is False

    def test_safety_blocks_eval(self):
        """eval() is blocked."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    return eval("5")
"""
        safe, error = evaluator._check_safety(code)

        assert safe is False
        assert "eval" in error.lower()

    def test_safety_blocks_exec(self):
        """exec() is blocked."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    exec("x = 5")
    return 0
"""
        safe, error = evaluator._check_safety(code)

        assert safe is False

    def test_safety_blocks_open(self):
        """open() is blocked."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    open("/etc/passwd", "r")
    return 0
"""
        safe, error = evaluator._check_safety(code)

        assert safe is False

    def test_safety_allows_numpy(self):
        """numpy is allowed."""
        evaluator = ProgramEvaluator()

        code = """
import numpy as np

def choose_action(obs):
    return int(np.argmax(obs[:7]))
"""
        safe, error = evaluator._check_safety(code)

        assert safe is True

    def test_safety_allows_random(self):
        """random is allowed."""
        evaluator = ProgramEvaluator()

        code = """
import random

def choose_action(obs):
    return random.randint(0, 6)
"""
        safe, error = evaluator._check_safety(code)

        assert safe is True


class TestInterfaceValidation:
    """Tests for interface validation."""

    def test_interface_valid(self):
        """Valid choose_action passes interface check."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    return 5
"""
        valid, error = evaluator._validate_interface(code)

        assert valid is True

    def test_interface_missing_function(self):
        """Missing choose_action fails."""
        evaluator = ProgramEvaluator()

        code = """
def some_other_function(x):
    return x
"""
        valid, error = evaluator._validate_interface(code)

        assert valid is False
        assert "choose_action" in error

    def test_interface_with_helper_functions(self):
        """Allows helper functions alongside choose_action."""
        evaluator = ProgramEvaluator()

        code = """
def helper(x):
    return x * 2

def choose_action(obs):
    return helper(obs[0])
"""
        valid, error = evaluator._validate_interface(code)

        assert valid is True

    def test_interface_no_args_fails(self):
        """choose_action with no args fails."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action():
    return 5
"""
        valid, error = evaluator._validate_interface(code)

        assert valid is False


class TestCodeExecution:
    """Tests for code execution."""

    def test_execute_extracts_function(self):
        """Execution returns callable function."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    return 5
"""
        player_fn = evaluator._execute_code(code)

        assert callable(player_fn)
        assert player_fn(np.zeros(244)) == 5

    def test_execute_with_numpy(self):
        """Execution works with numpy."""
        evaluator = ProgramEvaluator()

        code = """
import numpy as np

def choose_action(obs):
    return int(np.argmax(obs[:7]))
"""
        player_fn = evaluator._execute_code(code)

        obs = np.zeros(244)
        obs[3] = 1.0  # Set action 3 as highest

        assert player_fn(obs) == 3

    def test_execute_runtime_error(self):
        """Runtime errors in execution are caught."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs):
    return undefined_variable
"""
        with pytest.raises(Exception):
            player_fn = evaluator._execute_code(code)
            player_fn(np.zeros(244))


class TestSingleGame:
    """Tests for single game execution."""

    def test_single_game_completes(self):
        """Game runs to completion."""
        evaluator = ProgramEvaluator(num_games=1, max_steps=100)

        code = """
def choose_action(obs):
    return 5  # Hard drop
"""
        player_fn = evaluator._execute_code(code)
        result = evaluator._run_single_game(player_fn)

        assert isinstance(result, GameResult)
        assert result.steps > 0

    def test_single_game_with_random(self):
        """Game works with random actions."""
        evaluator = ProgramEvaluator(num_games=1, max_steps=100)

        code = """
import random

def choose_action(obs):
    return random.randint(0, 6)
"""
        player_fn = evaluator._execute_code(code)
        result = evaluator._run_single_game(player_fn)

        assert isinstance(result, GameResult)


class TestFullEvaluation:
    """Tests for full evaluation pipeline."""

    def test_evaluate_valid_code(self):
        """Full evaluation returns metrics."""
        evaluator = ProgramEvaluator(num_games=2, max_steps=100)

        code = """
def choose_action(obs):
    return 5
"""
        result = evaluator.evaluate(code, "trial_001")

        assert result.success is True
        assert result.games_played == 2
        assert result.avg_score is not None

    def test_evaluator_syntax_error(self):
        """Evaluator handles syntax error."""
        evaluator = ProgramEvaluator()

        code = """
def choose_action(obs)
    return 5
"""
        result = evaluator.evaluate(code, "trial_001")

        assert result.success is False
        assert "Syntax error" in result.error

    def test_evaluator_runtime_error(self):
        """Evaluator handles runtime error."""
        evaluator = ProgramEvaluator(num_games=2, max_steps=100)

        code = """
def choose_action(obs):
    return undefined_variable
"""
        result = evaluator.evaluate(code, "trial_001")

        # Runtime errors during game execution should be captured
        assert result.success is False or (result.game_results and "error" in result.game_results[0].terminated_reason)

    def test_evaluator_unsafe_code(self):
        """Evaluator blocks unsafe code."""
        evaluator = ProgramEvaluator()

        code = """
import os

def choose_action(obs):
    os.system("rm -rf /")
    return 0
"""
        result = evaluator.evaluate(code, "trial_001")

        assert result.success is False
        assert "Safety violation" in result.error

    def test_evaluate_aggregation(self):
        """Aggregation calculates correctly."""
        evaluator = ProgramEvaluator(num_games=5, max_steps=100)

        code = """
def choose_action(obs):
    return 5
"""
        result = evaluator.evaluate(code, "trial_001")

        if result.success:
            assert result.games_played == 5
            assert result.max_score >= result.min_score
            assert result.avg_score >= result.min_score
            assert result.avg_score <= result.max_score


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_code(self):
        """Empty code fails gracefully."""
        evaluator = ProgramEvaluator()

        result = evaluator.evaluate("", "trial_001")

        assert result.success is False

    def test_whitespace_only_code(self):
        """Whitespace-only code fails gracefully."""
        evaluator = ProgramEvaluator()

        result = evaluator.evaluate("   \n\n  ", "trial_001")

        assert result.success is False

    def test_code_with_class(self):
        """Code with class is allowed."""
        evaluator = ProgramEvaluator(num_games=2, max_steps=100)

        code = """
class Strategy:
    def __init__(self):
        self.count = 0

    def get_action(self, obs):
        self.count += 1
        return self.count % 7

strategy = Strategy()

def choose_action(obs):
    return strategy.get_action(obs)
"""
        result = evaluator.evaluate(code, "trial_001")

        assert result.success is True
