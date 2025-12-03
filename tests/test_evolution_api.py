"""
Tests for the Evolution API module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tetris_evolve import (
    CostTracker,
    ExperimentLogger,
    EvolutionAPI,
    TrialResult,
    BudgetExceededError,
)
from tetris_evolve.llm import MockLLMClient
from tetris_evolve.evaluation.circle_packing import CirclePackingEvaluator


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "valid": True,
        "sum_radii": 2.0,
        "target_ratio": 0.76,
        "combined_score": 0.76,
        "eval_time": 0.1,
        "error": None,
    }
    return evaluator


@pytest.fixture
def evolution_api(sample_config, temp_dir, mock_evaluator):
    """Create an EvolutionAPI instance for testing."""
    # Override output directory
    sample_config.experiment.output_dir = str(temp_dir)

    cost_tracker = CostTracker(sample_config)
    logger = ExperimentLogger(sample_config)
    logger.create_experiment_directory()

    child_llm = MockLLMClient(
        model=sample_config.child_llm.model,
        cost_tracker=cost_tracker,
        llm_type="child",
        responses=[
            '```python\nimport numpy as np\n\ndef construct_packing():\n    return np.zeros((26,2)), np.zeros(26), 0.0\n\ndef run_packing():\n    return construct_packing()\n```',
        ],
    )

    api = EvolutionAPI(
        evaluator=mock_evaluator,
        child_llm=child_llm,
        cost_tracker=cost_tracker,
        logger=logger,
        max_generations=5,
        max_children_per_generation=3,
    )

    return api


class TestEvolutionAPIBasic:
    """Basic tests for EvolutionAPI."""

    def test_initialization(self, evolution_api):
        """Test API initializes correctly."""
        assert evolution_api.current_generation == 0
        assert len(evolution_api.generations) == 1
        assert len(evolution_api.all_trials) == 0
        assert not evolution_api.is_terminated

    def test_get_current_generation(self, evolution_api):
        """Test getting current generation."""
        assert evolution_api._get_current_generation() == 0

    def test_get_cost_remaining(self, evolution_api):
        """Test getting remaining cost."""
        remaining = evolution_api._get_cost_remaining()
        assert remaining > 0


class TestSpawnChildLLM:
    """Tests for spawn_child_llm."""

    def test_spawn_returns_result(self, evolution_api):
        """Test that spawn returns a trial result."""
        result = evolution_api.spawn_child_llm(
            prompt="Generate a circle packing algorithm",
        )

        assert "trial_id" in result
        assert "code" in result
        assert "metrics" in result
        assert "success" in result
        assert result["trial_id"].startswith("trial_0_")

    def test_spawn_with_parent(self, evolution_api):
        """Test spawning with parent ID."""
        result = evolution_api.spawn_child_llm(
            prompt="Improve this algorithm",
            parent_id="trial_0_0",
        )

        assert result["parent_id"] == "trial_0_0"

    def test_spawn_records_trial(self, evolution_api):
        """Test that spawn records the trial."""
        result = evolution_api.spawn_child_llm(
            prompt="Generate algorithm",
        )

        assert result["trial_id"] in evolution_api.all_trials
        assert len(evolution_api.generations[0].trials) == 1

    def test_spawn_evaluates_code(self, evolution_api, mock_evaluator):
        """Test that spawn evaluates the code."""
        evolution_api.spawn_child_llm(prompt="Generate algorithm")

        mock_evaluator.evaluate.assert_called_once()

    def test_spawn_handles_no_code(self, sample_config, temp_dir, mock_evaluator):
        """Test handling when LLM response has no code."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Here's my explanation without any code."],
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        result = api.spawn_child_llm(prompt="Test")

        assert result["success"] is False
        assert "No Python code block" in result["error"]

    def test_spawn_budget_exceeded(self, evolution_api):
        """Test that spawn raises on budget exceeded."""
        evolution_api.cost_tracker.total_cost = 100.0  # Exceed budget

        with pytest.raises(BudgetExceededError):
            evolution_api.spawn_child_llm(prompt="Test")


class TestEvaluateProgram:
    """Tests for evaluate_program."""

    def test_evaluate_calls_evaluator(self, evolution_api, mock_evaluator):
        """Test that evaluate_program calls the evaluator."""
        code = "def run_packing(): pass"
        evolution_api.evaluate_program(code)

        mock_evaluator.evaluate.assert_called_with(code)

    def test_evaluate_returns_metrics(self, evolution_api):
        """Test that evaluate_program returns metrics."""
        result = evolution_api.evaluate_program("test code")

        assert "valid" in result
        assert "sum_radii" in result

    def test_evaluate_handles_error(self, sample_config, temp_dir):
        """Test that evaluate_program handles errors gracefully."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # Evaluator that raises
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = Exception("Evaluation failed")

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        result = api.evaluate_program("broken code")

        assert result["valid"] is False
        assert "Evaluation error" in result["error"]


class TestAdvanceGeneration:
    """Tests for advance_generation."""

    def test_advance_increments_generation(self, evolution_api):
        """Test that advance_generation increments the counter."""
        evolution_api.spawn_child_llm(prompt="Test")
        new_gen = evolution_api.advance_generation(
            selected_trial_ids=["trial_0_0"],
            reasoning="Selected best trial",
        )

        assert new_gen == 1
        assert evolution_api.current_generation == 1

    def test_advance_creates_new_generation(self, evolution_api):
        """Test that advance creates a new generation entry."""
        evolution_api.spawn_child_llm(prompt="Test")
        evolution_api.advance_generation(
            selected_trial_ids=["trial_0_0"],
            reasoning="Test",
        )

        assert len(evolution_api.generations) == 2
        assert evolution_api.generations[1].generation_num == 1

    def test_advance_records_selection(self, evolution_api):
        """Test that selection is recorded."""
        evolution_api.spawn_child_llm(prompt="Test")
        evolution_api.advance_generation(
            selected_trial_ids=["trial_0_0"],
            reasoning="Selected best trial",
        )

        gen0 = evolution_api.generations[0]
        assert gen0.selected_trial_ids == ["trial_0_0"]
        assert gen0.selection_reasoning == "Selected best trial"


class TestTerminateEvolution:
    """Tests for terminate_evolution."""

    def test_terminate_sets_flag(self, evolution_api):
        """Test that terminate sets the terminated flag."""
        evolution_api.spawn_child_llm(prompt="Test")
        evolution_api.terminate_evolution("Test complete")

        assert evolution_api.is_terminated

    def test_terminate_returns_results(self, evolution_api):
        """Test that terminate returns final results."""
        evolution_api.spawn_child_llm(prompt="Test")
        result = evolution_api.terminate_evolution("Test complete")

        assert result["terminated"] is True
        assert result["reason"] == "Test complete"
        assert "best_program" in result
        assert "cost_summary" in result

    def test_terminate_with_best_program(self, evolution_api):
        """Test that terminate accepts best_program argument."""
        evolution_api.spawn_child_llm(prompt="Test")
        best_code = "def run_packing(): return best_solution()"
        result = evolution_api.terminate_evolution("Done", best_program=best_code)

        assert result["best_program"] == best_code


class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_get_best_trials_returns_sorted(self, sample_config, temp_dir):
        """Test that _get_best_trials returns sorted trials."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # Create evaluator that returns different scores
        evaluator = MagicMock()
        scores = [1.5, 2.0, 1.8]
        call_count = [0]

        def mock_evaluate(code):
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return {
                "valid": True,
                "sum_radii": score,
                "target_ratio": score / 2.635,
                "combined_score": score / 2.635,
                "eval_time": 0.1,
                "error": None,
            }

        evaluator.evaluate.side_effect = mock_evaluate

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=[
                '```python\ndef run_packing(): pass\n```',
                '```python\ndef run_packing(): pass\n```',
                '```python\ndef run_packing(): pass\n```',
            ],
        )

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn three trials
        api.spawn_child_llm(prompt="Test 1")
        api.spawn_child_llm(prompt="Test 2")
        api.spawn_child_llm(prompt="Test 3")

        best = api._get_best_trials(n=3)

        assert len(best) == 3
        assert best[0]["metrics"]["sum_radii"] == 2.0
        assert best[1]["metrics"]["sum_radii"] == 1.8
        assert best[2]["metrics"]["sum_radii"] == 1.5

    def test_get_best_filters_invalid(self, evolution_api, mock_evaluator):
        """Test that _get_best_trials filters invalid trials."""
        # Make evaluator return invalid
        mock_evaluator.evaluate.return_value = {
            "valid": False,
            "sum_radii": 0,
            "error": "Invalid packing",
        }

        evolution_api.spawn_child_llm(prompt="Test")
        best = evolution_api._get_best_trials(n=5)

        assert len(best) == 0

    def test_get_generation_history(self, evolution_api):
        """Test that generation history is returned."""
        evolution_api.spawn_child_llm(prompt="Test")
        evolution_api.advance_generation(["trial_0_0"], "Test")

        history = evolution_api._get_generation_history()

        assert len(history) == 2
        assert history[0]["generation_num"] == 0
        assert history[1]["generation_num"] == 1


class TestGetAPIFunctions:
    """Tests for get_api_functions."""

    def test_returns_only_4_functions(self, evolution_api):
        """Test that only 4 core API functions are returned."""
        funcs = evolution_api.get_api_functions()

        assert len(funcs) == 4
        assert "spawn_child_llm" in funcs
        assert "evaluate_program" in funcs
        assert "advance_generation" in funcs
        assert "terminate_evolution" in funcs

    def test_internal_functions_not_exposed(self, evolution_api):
        """Test that internal helper functions are not exposed."""
        funcs = evolution_api.get_api_functions()

        # These should NOT be in the API
        assert "get_best_trials" not in funcs
        assert "get_generation_history" not in funcs
        assert "get_cost_remaining" not in funcs
        assert "get_trial" not in funcs
        assert "get_current_generation" not in funcs

    def test_functions_are_callable(self, evolution_api):
        """Test that returned functions are callable."""
        funcs = evolution_api.get_api_functions()

        for name, func in funcs.items():
            assert callable(func), f"{name} is not callable"


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_to_dict(self):
        """Test converting TrialResult to dictionary."""
        trial = TrialResult(
            trial_id="test_0_0",
            code="def run_packing(): pass",
            metrics={"sum_radii": 1.5},
            prompt="Test prompt",
            response="Test response",
            reasoning="Test reasoning",
            success=True,
            parent_id="parent_0_0",
            error=None,
            generation=0,
        )

        d = trial.to_dict()

        assert d["trial_id"] == "test_0_0"
        assert d["code"] == "def run_packing(): pass"
        assert d["metrics"] == {"sum_radii": 1.5}
        assert d["success"] is True
        assert d["parent_id"] == "parent_0_0"


class TestIntegrationWithRealEvaluator:
    """Integration tests using the real CirclePackingEvaluator."""

    def test_spawn_with_real_evaluator(self, sample_config, temp_dir, sample_valid_packing_code):
        """Test spawning with real evaluator and valid code."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        evaluator = CirclePackingEvaluator(
            target=2.635,
            n_circles=26,
            timeout_seconds=30,
        )

        # Wrap the valid code in a python block
        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=[f"```python\n{sample_valid_packing_code}\n```"],
        )

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        result = api.spawn_child_llm(prompt="Generate circle packing")

        assert result["success"] is True
        assert result["metrics"]["valid"] is True
        assert result["metrics"]["sum_radii"] > 0
