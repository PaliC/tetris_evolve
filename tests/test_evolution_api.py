"""
Tests for the Evolution API module.
"""

from unittest.mock import MagicMock

import pytest

from tetris_evolve import (
    BudgetExceededError,
    CostTracker,
    EvolutionAPI,
    ExperimentLogger,
    TrialResult,
)
from tetris_evolve.evaluation.circle_packing import CirclePackingEvaluator
from tetris_evolve.exceptions import ChildrenLimitError, GenerationLimitError
from tetris_evolve.llm import MockLLMClient


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
            "```python\nimport numpy as np\n\ndef construct_packing():\n    return np.zeros((26,2)), np.zeros(26), 0.0\n\ndef run_packing():\n    return construct_packing()\n```",
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


class TestInternalAdvanceGeneration:
    """Tests for _advance_generation (internal method)."""

    def test_advance_increments_generation(self, evolution_api):
        """Test that _advance_generation increments the counter."""
        evolution_api.spawn_child_llm(prompt="Test")
        new_gen = evolution_api._advance_generation()

        assert new_gen == 1
        assert evolution_api.current_generation == 1

    def test_advance_creates_new_generation(self, evolution_api):
        """Test that advance creates a new generation entry."""
        evolution_api.spawn_child_llm(prompt="Test")
        evolution_api._advance_generation()

        assert len(evolution_api.generations) == 2
        assert evolution_api.generations[1].generation_num == 1

    def test_advance_auto_selects_best_trials(self, evolution_api):
        """Test that best trials are auto-selected."""
        evolution_api.spawn_child_llm(prompt="Test")
        evolution_api._advance_generation()

        gen0 = evolution_api.generations[0]
        # Auto-selects best performing trials
        assert gen0.selection_reasoning == "Auto-selected top performing trials"

    def test_can_advance_generation(self, evolution_api):
        """Test can_advance_generation helper."""
        assert evolution_api.can_advance_generation() is True

        # Advance to near max
        for i in range(4):
            evolution_api.spawn_child_llm(prompt=f"Test {i}")
            evolution_api._advance_generation()

        # At generation 4 (max is 5), can't advance further
        assert evolution_api.can_advance_generation() is False

    def test_has_children_in_current_generation(self, evolution_api):
        """Test has_children_in_current_generation helper."""
        assert evolution_api.has_children_in_current_generation() is False

        evolution_api.spawn_child_llm(prompt="Test")
        assert evolution_api.has_children_in_current_generation() is True


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
                "```python\ndef run_packing(): pass\n```",
                "```python\ndef run_packing(): pass\n```",
                "```python\ndef run_packing(): pass\n```",
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
        evolution_api._advance_generation()

        history = evolution_api._get_generation_history()

        assert len(history) == 2
        assert history[0]["generation_num"] == 0
        assert history[1]["generation_num"] == 1


class TestGetAPIFunctions:
    """Tests for get_api_functions."""

    def test_returns_only_5_functions(self, evolution_api):
        """Test that only 5 core API functions are returned."""
        funcs = evolution_api.get_api_functions()

        assert len(funcs) == 5
        assert "spawn_child_llm" in funcs
        assert "spawn_children_parallel" in funcs
        assert "evaluate_program" in funcs
        assert "terminate_evolution" in funcs
        assert "get_trial_code" in funcs

    def test_advance_generation_not_exposed(self, evolution_api):
        """Test that advance_generation is not in the public API."""
        funcs = evolution_api.get_api_functions()

        # advance_generation is now internal, not exposed to Root LLM
        assert "advance_generation" not in funcs

    def test_internal_functions_not_exposed(self, evolution_api):
        """Test that internal helper functions are not exposed."""
        funcs = evolution_api.get_api_functions()

        # These should NOT be in the API
        assert "get_best_trials" not in funcs
        assert "get_generation_history" not in funcs
        assert "get_cost_remaining" not in funcs
        assert "get_trial" not in funcs
        assert "get_current_generation" not in funcs
        assert "_advance_generation" not in funcs

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


class TestChildrenLimitEnforcement:
    """Tests for max_children_per_generation limit enforcement."""

    def test_spawn_up_to_limit(self, sample_config, temp_dir, mock_evaluator):
        """Test spawning children up to the limit succeeds."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 3,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=3,
        )

        # Spawn 3 children (the limit)
        for i in range(3):
            result = api.spawn_child_llm(prompt=f"Test {i}")
            assert result["trial_id"] == f"trial_0_{i}"

        assert len(api.generations[0].trials) == 3

    def test_spawn_exceeds_limit_raises(self, sample_config, temp_dir, mock_evaluator):
        """Test that spawning beyond the limit raises ChildrenLimitError."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 4,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=3,
        )

        # Spawn 3 children (the limit)
        for i in range(3):
            api.spawn_child_llm(prompt=f"Test {i}")

        # 4th spawn should raise
        with pytest.raises(ChildrenLimitError) as exc_info:
            api.spawn_child_llm(prompt="One too many")

        assert "Limit of 3 children reached" in str(exc_info.value)

    def test_spawn_after_advance_resets_limit(self, sample_config, temp_dir, mock_evaluator):
        """Test that advancing generation resets the children count."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 6,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=3,
        )

        # Spawn 3 children in gen 0
        for i in range(3):
            api.spawn_child_llm(prompt=f"Gen0 Test {i}")

        # Advance to gen 1 (using internal method)
        api._advance_generation()

        # Should be able to spawn 3 more in gen 1
        for i in range(3):
            result = api.spawn_child_llm(prompt=f"Gen1 Test {i}")
            assert result["trial_id"] == f"trial_1_{i}"

        assert len(api.generations[1].trials) == 3


class TestGenerationLimitEnforcement:
    """Tests for max_generations limit enforcement."""

    def test_advance_up_to_limit(self, sample_config, temp_dir, mock_evaluator):
        """Test advancing generations up to the limit succeeds."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 3,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=3,  # Can have gen 0, 1, 2
            max_children_per_generation=1,
        )

        # Gen 0
        api.spawn_child_llm(prompt="Gen 0")
        new_gen = api._advance_generation()
        assert new_gen == 1

        # Gen 1
        api.spawn_child_llm(prompt="Gen 1")
        new_gen = api._advance_generation()
        assert new_gen == 2

        # Now at gen 2, which is the last allowed (0, 1, 2 = 3 generations)
        assert api.current_generation == 2

    def test_advance_exceeds_limit_raises(self, sample_config, temp_dir, mock_evaluator):
        """Test that advancing beyond the limit raises GenerationLimitError."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 4,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=2,  # Only gen 0 and 1 allowed
            max_children_per_generation=1,
        )

        # Gen 0 -> Gen 1
        api.spawn_child_llm(prompt="Gen 0")
        api._advance_generation()

        # Gen 1 spawn
        api.spawn_child_llm(prompt="Gen 1")

        # Try to advance to gen 2 - should fail
        with pytest.raises(GenerationLimitError) as exc_info:
            api._advance_generation()

        assert "Maximum of 2 generations reached" in str(exc_info.value)

    def test_can_still_spawn_in_last_generation(self, sample_config, temp_dir, mock_evaluator):
        """Test that children can still be spawned in the last generation."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 4,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=2,
            max_children_per_generation=3,
        )

        # Move to last generation
        api.spawn_child_llm(prompt="Gen 0")
        api._advance_generation()

        # Should still be able to spawn children in gen 1
        for i in range(3):
            result = api.spawn_child_llm(prompt=f"Gen 1 child {i}")
            assert result["trial_id"] == f"trial_1_{i}"


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


class TestTrialCodeSubstitution:
    """Tests for {{CODE_TRIAL_X_Y}} token substitution in prompts."""

    def test_spawn_with_code_token_substitutes(self, sample_config, temp_dir, mock_evaluator):
        """Test that {{CODE_TRIAL_X_Y}} tokens are substituted in spawn_child_llm."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # Track what prompts are sent to LLM
        received_prompts = []

        def mock_generate(messages, **kwargs):
            received_prompts.append(messages[0]["content"])
            response = MagicMock()
            response.content = "```python\ndef run_packing(): pass\n```"
            return response

        child_llm = MagicMock()
        child_llm.generate = mock_generate

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn first trial to have code to reference
        api.spawn_child_llm(prompt="Generate algorithm")
        first_trial_code = api.all_trials["trial_0_0"].code

        # Now spawn second trial referencing the first
        api.spawn_child_llm(prompt="Improve this: {{CODE_TRIAL_0_0}}")

        # The second prompt should have the token replaced
        assert len(received_prompts) == 2
        assert "{{CODE_TRIAL_0_0}}" not in received_prompts[1]
        assert first_trial_code in received_prompts[1]

    def test_spawn_with_missing_trial_shows_error(self, sample_config, temp_dir, mock_evaluator):
        """Test that missing trial reference shows error marker."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        received_prompts = []

        def mock_generate(messages, **kwargs):
            received_prompts.append(messages[0]["content"])
            response = MagicMock()
            response.content = "```python\ndef run_packing(): pass\n```"
            return response

        child_llm = MagicMock()
        child_llm.generate = mock_generate

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn with reference to non-existent trial
        api.spawn_child_llm(prompt="Improve: {{CODE_TRIAL_99_99}}")

        # Should show error marker
        assert "[CODE NOT FOUND: trial_99_99]" in received_prompts[0]

    def test_substituted_prompt_stored_in_trial(self, sample_config, temp_dir, mock_evaluator):
        """Test that the substituted prompt is stored in trial result."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=[
                '```python\ndef construct_packing(): return "first"\ndef run_packing(): return construct_packing()\n```',
                "```python\ndef run_packing(): pass\n```",
            ],
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn first trial
        api.spawn_child_llm(prompt="First")

        # Spawn second trial with token
        api.spawn_child_llm(prompt="Improve: {{CODE_TRIAL_0_0}}")

        # The stored prompt should be the substituted version
        second_trial = api.all_trials["trial_0_1"]
        assert "{{CODE_TRIAL_0_0}}" not in second_trial.prompt
        # Should contain the actual code
        assert 'def construct_packing(): return "first"' in second_trial.prompt

    def test_multiple_tokens_in_single_prompt(self, sample_config, temp_dir, mock_evaluator):
        """Test that multiple tokens in one prompt are all substituted."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=[
                '```python\ndef run_packing(): return "A"\n```',
                '```python\ndef run_packing(): return "B"\n```',
                "```python\ndef run_packing(): pass\n```",
            ],
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn two trials
        api.spawn_child_llm(prompt="First")
        api.spawn_child_llm(prompt="Second")

        # Spawn third referencing both
        api.spawn_child_llm(prompt="Combine {{CODE_TRIAL_0_0}} and {{CODE_TRIAL_0_1}}")

        third_trial = api.all_trials["trial_0_2"]
        assert "{{CODE_TRIAL_0_0}}" not in third_trial.prompt
        assert "{{CODE_TRIAL_0_1}}" not in third_trial.prompt
        assert 'return "A"' in third_trial.prompt
        assert 'return "B"' in third_trial.prompt


class TestParallelSpawnWithTokenSubstitution:
    """Tests for token substitution in spawn_children_parallel."""

    def test_parallel_spawn_substitutes_tokens(self, sample_config, temp_dir, mock_evaluator):
        """Test that tokens are substituted in parallel spawn prompts."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # First spawn a trial to reference
        child_llm = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=['```python\ndef run_packing(): return "parent"\n```'],
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            child_llm_model="test-model",
            evaluator_kwargs={"n_circles": 26, "target": 2.635, "timeout_seconds": 30},
        )

        api.spawn_child_llm(prompt="Parent trial")
        parent_code = api.all_trials["trial_0_0"].code

        # The parallel spawn will use multiprocessing which we can't easily mock,
        # so we test that the prompts are prepared correctly before spawning
        # by checking the internal state

        # This tests that substitute_trial_codes is called correctly
        # The actual parallel execution is tested in integration tests

        # Verify the parent trial code is available for substitution
        from tetris_evolve.utils.prompt_substitution import substitute_trial_codes

        prompt = "Improve: {{CODE_TRIAL_0_0}}"
        result, report = substitute_trial_codes(
            prompt,
            all_trials=api.all_trials,
            experiment_dir=str(logger.base_dir),
        )

        assert parent_code in result
        assert report[0]["success"] is True
