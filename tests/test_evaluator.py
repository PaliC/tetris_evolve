"""
Tests for mango_evolve.evaluation.circle_packing module.
"""

import numpy as np
import pytest

from mango_evolve.evaluation import (
    CirclePackingEvaluator,
    validate_packing,
)


class TestValidatePacking:
    """Tests for validate_packing function."""

    def test_valid_packing(self):
        """Valid packing passes validation."""
        # Simple non-overlapping circles
        centers = np.array(
            [
                [0.2, 0.2],
                [0.8, 0.8],
            ]
        )
        radii = np.array([0.1, 0.1])

        valid, error = validate_packing(centers, radii, n_circles=2)

        assert valid is True
        assert error is None

    def test_validate_packing_overlap(self):
        """Detects overlapping circles."""
        # Two circles at same position
        centers = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
        radii = np.array([0.1, 0.1])

        valid, error = validate_packing(centers, radii, n_circles=2)

        assert valid is False
        assert "overlap" in error.lower()

    def test_validate_packing_bounds(self):
        """Detects circles outside unit square."""
        centers = np.array(
            [
                [0.05, 0.5],  # Circle extends past x=0
            ]
        )
        radii = np.array([0.1])  # Extends to x=-0.05

        valid, error = validate_packing(centers, radii, n_circles=1)

        assert valid is False
        assert "outside" in error.lower() or "bounds" in error.lower()

    def test_validate_packing_negative_radii(self):
        """Detects negative radii."""
        centers = np.array([[0.5, 0.5]])
        radii = np.array([-0.1])

        valid, error = validate_packing(centers, radii, n_circles=1)

        assert valid is False
        assert "negative" in error.lower()

    def test_validate_wrong_shape(self):
        """Detects wrong array shapes."""
        centers = np.array([[0.5, 0.5]])
        radii = np.array([0.1, 0.2])  # Wrong number of radii

        valid, error = validate_packing(centers, radii, n_circles=1)

        assert valid is False
        assert "shape" in error.lower()


class TestCirclePackingEvaluator:
    """Tests for CirclePackingEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with short timeout."""
        return CirclePackingEvaluator(timeout_seconds=10)

    def test_evaluate_valid_code(self, evaluator, sample_valid_packing_code):
        """Valid packing code runs and returns metrics."""
        result = evaluator.evaluate(sample_valid_packing_code)

        assert result["valid"] is True
        assert result["score"] > 0
        assert result["error"] is None

    def test_evaluate_no_function(self, evaluator, sample_no_function_code):
        """Detects missing construct_packing/run_packing."""
        result = evaluator.evaluate(sample_no_function_code)

        assert result["valid"] is False
        assert result["error"] is not None
        assert "run_packing" in result["error"] or "construct_packing" in result["error"]

    def test_evaluate_solve_function_gives_helpful_error(self, evaluator):
        """Gives helpful error when code uses solve() instead of construct_packing()."""
        code = """
def solve():
    return [(0.5, 0.5, 0.1)]
"""
        result = evaluator.evaluate(code)

        assert result["valid"] is False
        assert "solve()" in result["error"]
        assert "construct_packing" in result["error"]

    def test_evaluate_syntax_error(self, evaluator):
        """Handles syntax errors."""
        code = """
def run_packing(
    # Missing closing paren
"""
        result = evaluator.evaluate(code)

        assert result["valid"] is False
        assert result["error"] is not None

    def test_evaluate_runtime_error(self, evaluator, sample_broken_code):
        """Handles runtime errors."""
        result = evaluator.evaluate(sample_broken_code)

        assert result["valid"] is False
        assert result["error"] is not None
        assert "undefined" in result["error"].lower() or "NameError" in result["error"]

    def test_evaluate_invalid_packing(self, evaluator, sample_invalid_packing_code):
        """Detects invalid packing (overlaps)."""
        result = evaluator.evaluate(sample_invalid_packing_code)

        assert result["valid"] is False
        assert "overlap" in result["error"].lower()

    def test_metrics_computed(self, evaluator, sample_valid_packing_code):
        """All expected metrics returned."""
        result = evaluator.evaluate(sample_valid_packing_code)

        assert "valid" in result
        assert "score" in result
        assert "eval_time" in result
        assert "error" in result

    def test_timeout_handling(self, evaluator):
        """Code that takes too long is killed."""
        # Create evaluator with very short timeout
        fast_evaluator = CirclePackingEvaluator(timeout_seconds=1)

        code = """
import time

def run_packing():
    time.sleep(10)  # Will timeout
    return None, None, 0
"""
        result = fast_evaluator.evaluate(code)

        assert result["valid"] is False
        assert "timeout" in result["error"].lower()

    def test_invalid_score_is_zero(self, evaluator, sample_invalid_packing_code):
        """Invalid packings have zero scores."""
        result = evaluator.evaluate(sample_invalid_packing_code)

        assert result["score"] == 0.0

