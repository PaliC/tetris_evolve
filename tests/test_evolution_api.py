"""
Tests for the Evolution API module.
"""

from unittest.mock import MagicMock

import pytest

from mango_evolve import (
    BudgetExceededError,
    CostTracker,
    EvolutionAPI,
    ExperimentLogger,
    TrialResult,
)
from mango_evolve.evolution_api import ScratchpadProxy
from mango_evolve.llm import MockLLMClient


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "valid": True,
        "score": 2.0,
        "eval_time": 0.1,
        "error": None,
    }
    return evaluator


@pytest.fixture
def child_llm_configs(sample_config):
    """Create child LLM configs dict from sample config."""
    return {cfg.effective_alias: cfg for cfg in sample_config.child_llms}


@pytest.fixture
def evolution_api(sample_config, temp_dir, mock_evaluator, child_llm_configs):
    """Create an EvolutionAPI instance for testing."""
    # Override output directory
    sample_config.experiment.output_dir = str(temp_dir)

    cost_tracker = CostTracker(sample_config)
    logger = ExperimentLogger(sample_config)
    logger.create_experiment_directory()

    api = EvolutionAPI(
        evaluator=mock_evaluator,
        child_llm_configs=child_llm_configs,
        cost_tracker=cost_tracker,
        logger=logger,
        max_generations=5,
        max_children_per_generation=3,
        default_child_llm_alias=sample_config.default_child_llm_alias,
    )

    # Inject mock LLM clients to avoid real API calls
    for alias in child_llm_configs:
        mock_client = MockLLMClient(
            model=child_llm_configs[alias].model,
            cost_tracker=cost_tracker,
            llm_type=f"child:{alias}",
            responses=["```python\ndef solve(): return 1.0\n```"],
        )
        api.child_llm_clients[alias] = mock_client

    # End calibration phase for testing
    api.end_calibration_phase()

    return api


class TestEvolutionAPIBasic:
    """Basic tests for EvolutionAPI."""

    def test_initialization(self, evolution_api):
        """Test API initializes correctly."""
        assert evolution_api.current_generation == 0
        assert len(evolution_api.generations) == 1
        assert len(evolution_api.all_trials) == 0
        assert not evolution_api.is_terminated


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

    def test_spawn_handles_no_code(
        self, sample_config, temp_dir, mock_evaluator, child_llm_configs
    ):
        """Test handling when LLM response has no code."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm_configs=child_llm_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            default_child_llm_alias=sample_config.default_child_llm_alias,
        )
        api.end_calibration_phase()

        # Inject a mock client that returns no code
        mock_client = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child:default",
            responses=["Here's my explanation without any code."],
        )
        api.child_llm_clients["default"] = mock_client

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
        assert "score" in result

    def test_evaluate_handles_error(self, sample_config, temp_dir, child_llm_configs):
        """Test that evaluate_program handles errors gracefully."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # Evaluator that raises
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = Exception("Evaluation failed")

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm_configs=child_llm_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            default_child_llm_alias=sample_config.default_child_llm_alias,
        )
        api.end_calibration_phase()

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
        # Auto-selects best performing trials (across all generations)
        assert "Auto-selected top performing trials" in gen0.selection_reasoning

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

    def test_get_best_trials_returns_sorted(self, sample_config, temp_dir, child_llm_configs):
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
                "score": score,
                "eval_time": 0.1,
                "error": None,
            }

        evaluator.evaluate.side_effect = mock_evaluate

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm_configs=child_llm_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            default_child_llm_alias=sample_config.default_child_llm_alias,
        )
        api.end_calibration_phase()

        # Inject mock client
        mock_client = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child:default",
            responses=[
                "```python\ndef run_packing(): pass\n```",
                "```python\ndef run_packing(): pass\n```",
                "```python\ndef run_packing(): pass\n```",
            ],
        )
        api.child_llm_clients["default"] = mock_client

        # Spawn three trials
        api.spawn_child_llm(prompt="Test 1")
        api.spawn_child_llm(prompt="Test 2")
        api.spawn_child_llm(prompt="Test 3")

        best = api._get_best_trials(n=3)

        assert len(best) == 3
        assert best[0]["metrics"]["score"] == 2.0
        assert best[1]["metrics"]["score"] == 1.8
        assert best[2]["metrics"]["score"] == 1.5

    def test_get_best_filters_invalid(self, evolution_api, mock_evaluator):
        """Test that _get_best_trials filters invalid trials."""
        # Make evaluator return invalid
        mock_evaluator.evaluate.return_value = {
            "valid": False,
            "score": 0,
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

    def test_returns_10_items(self, evolution_api):
        """Test that 10 items (9 functions + 1 scratchpad proxy) are returned."""
        funcs = evolution_api.get_api_functions()

        assert len(funcs) == 10
        assert "spawn_child_llm" in funcs
        assert "spawn_children_parallel" in funcs
        assert "evaluate_program" in funcs
        assert "terminate_evolution" in funcs
        assert "get_top_trials" in funcs
        assert "get_trial_code" in funcs
        assert "update_scratchpad" in funcs
        assert "end_calibration_phase" in funcs
        assert "get_calibration_status" in funcs
        assert "scratchpad" in funcs
        assert isinstance(funcs["scratchpad"], ScratchpadProxy)

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
        assert "get_trial" not in funcs
        assert "get_current_generation" not in funcs
        assert "_advance_generation" not in funcs


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_to_dict(self):
        """Test converting TrialResult to dictionary."""
        trial = TrialResult(
            trial_id="test_0_0",
            code="def run_packing(): pass",
            metrics={"score": 1.5},
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
        assert d["metrics"] == {"score": 1.5}
        assert d["success"] is True
        assert d["parent_id"] == "parent_0_0"


class TestPromptSubstitutionUtility:
    """Tests for prompt substitution utility (not requiring EvolutionAPI)."""

    def test_substitute_trial_codes_works(self, evolution_api):
        """Test that substitute_trial_codes utility works."""
        from mango_evolve.utils.prompt_substitution import substitute_trial_codes

        # Just test the utility function directly
        prompt = "Test: {{CODE_TRIAL_0_0}}"
        result, report = substitute_trial_codes(
            prompt,
            all_trials={},  # Empty - should show error marker
            experiment_dir="/tmp",
        )

        assert "[CODE NOT FOUND: trial_0_0]" in result


class TestScratchpad:
    """Tests for the scratchpad functionality."""

    def test_scratchpad_initially_empty(self, evolution_api):
        """Test that scratchpad is initially empty."""
        assert evolution_api.scratchpad == ""

    def test_update_scratchpad(self, evolution_api):
        """Test updating the scratchpad."""
        result = evolution_api.update_scratchpad("Test notes")

        assert result["success"] is True
        assert result["length"] == 10
        assert evolution_api.scratchpad == "Test notes"

    def test_update_scratchpad_replaces_content(self, evolution_api):
        """Test that update_scratchpad replaces existing content."""
        evolution_api.update_scratchpad("First notes")
        evolution_api.update_scratchpad("Second notes")

        assert evolution_api.scratchpad == "Second notes"

    def test_update_scratchpad_truncates_long_content(self, evolution_api):
        """Test that very long content is truncated."""
        long_content = "x" * 10000  # Exceeds 8000 limit
        result = evolution_api.update_scratchpad(long_content)

        assert result["length"] == 8000
        assert len(evolution_api.scratchpad) == 8000


class TestLineageMap:
    """Tests for the lineage map builder."""

    def test_empty_lineage_map(self, evolution_api):
        """Test lineage map with no trials."""
        result = evolution_api._build_lineage_map()
        assert result == "(No trials yet)"

    def test_lineage_map_includes_all_time_top_5(self, evolution_api):
        """Test lineage map includes All-Time Top 5 summary."""
        # Spawn some trials
        evolution_api.spawn_child_llm(prompt="Test 1")
        evolution_api.spawn_child_llm(prompt="Test 2")

        result = evolution_api._build_lineage_map()

        assert "All-Time Top 5" in result
        assert "cross-generation selection candidates" in result


class TestSelectionBehavior:
    """Tests for selection functionality (historical selection allowed)."""

    def test_selection_allows_historical_trials(self, sample_config, temp_dir, child_llm_configs):
        """Test that _advance_generation accepts trials from any generation."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # Create evaluator that returns different scores
        evaluator = MagicMock()
        call_count = [0]
        scores = [2.5, 2.3, 2.1, 2.6, 2.4]

        def mock_evaluate(code):
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return {"valid": True, "score": score, "eval_time": 0.1, "error": None}

        evaluator.evaluate.side_effect = mock_evaluate

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm_configs=child_llm_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=3,
            default_child_llm_alias=sample_config.default_child_llm_alias,
        )
        api.end_calibration_phase()

        mock_client = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child:default",
            responses=["```python\ndef run_packing(): pass\n```"] * 10,
        )
        api.child_llm_clients["default"] = mock_client

        # Spawn trials in Gen 0
        api.spawn_child_llm(prompt="Gen 0 Trial 1")  # trial_0_0
        api.spawn_child_llm(prompt="Gen 0 Trial 2")  # trial_0_1
        api._advance_generation()

        # Spawn trials in Gen 1
        api.spawn_child_llm(prompt="Gen 1 Trial 1")  # trial_1_0
        api.spawn_child_llm(prompt="Gen 1 Trial 2")  # trial_1_1

        # Try to select with a mix of historical (Gen 0) and current (Gen 1) trials
        selections = [
            {"trial_id": "trial_1_0", "reasoning": "Current gen", "category": "performance"},
            {"trial_id": "trial_0_0", "reasoning": "Historical (allowed)", "category": "diversity"},
        ]

        api._advance_generation(selections=selections, selection_summary="Test")

        # Both current and historical selections should be preserved
        gen1 = api.generations[1]
        assert len(gen1.trial_selections) == 2
        assert gen1.selected_trial_ids == ["trial_1_0", "trial_0_0"]
        assert gen1.trial_selections[0].source_generation == 1
        assert gen1.trial_selections[1].source_generation == 0

    def test_auto_select_from_current_generation_only(
        self, sample_config, temp_dir, child_llm_configs
    ):
        """Test that auto-selection only selects from current generation."""
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        # Gen 0 gets high score, Gen 1 gets lower scores
        evaluator = MagicMock()
        call_count = [0]
        scores = [2.8, 2.1, 2.0]  # Gen 0 trial is best, but shouldn't be selected

        def mock_evaluate(code):
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return {"valid": True, "score": score, "eval_time": 0.1, "error": None}

        evaluator.evaluate.side_effect = mock_evaluate

        api = EvolutionAPI(
            evaluator=evaluator,
            child_llm_configs=child_llm_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=3,
            default_child_llm_alias=sample_config.default_child_llm_alias,
        )
        api.end_calibration_phase()

        mock_client = MockLLMClient(
            model="test",
            cost_tracker=cost_tracker,
            llm_type="child:default",
            responses=["```python\ndef run_packing(): pass\n```"] * 10,
        )
        api.child_llm_clients["default"] = mock_client

        # Gen 0: one trial with high score
        api.spawn_child_llm(prompt="Gen 0 high score")  # trial_0_0, score 2.8
        api._advance_generation()

        # Gen 1: two trials with lower scores
        api.spawn_child_llm(prompt="Gen 1 low 1")  # trial_1_0, score 2.1
        api.spawn_child_llm(prompt="Gen 1 low 2")  # trial_1_1, score 2.0

        # Let auto-select happen (no selections provided)
        api._advance_generation()

        # Auto-select should only include Gen 1 trials (not Gen 0)
        gen1 = api.generations[1]
        assert "trial_0_0" not in gen1.selected_trial_ids
        assert "trial_1_0" in gen1.selected_trial_ids  # Best of Gen 1
        assert "Auto-selected top performing trials" in gen1.selection_reasoning

    def test_trial_selection_source_generation_field(self):
        """Test TrialSelection dataclass has source_generation field."""
        from mango_evolve.evolution_api import TrialSelection

        sel = TrialSelection(
            trial_id="trial_0_5",
            reasoning="Test",
            category="performance",
            source_generation=0,
        )

        assert sel.source_generation == 0

        # Test to_dict includes it
        d = sel.to_dict()
        assert d["source_generation"] == 0

        # Test from_dict with source_generation
        sel2 = TrialSelection.from_dict(
            {"trial_id": "trial_1_2", "reasoning": "Test2", "category": "diversity"},
            source_generation=1,
        )
        assert sel2.source_generation == 1


class TestScratchpadProxy:
    """Tests for ScratchpadProxy."""

    def test_content_getter(self, evolution_api):
        """Test reading scratchpad content via proxy."""
        proxy = ScratchpadProxy(evolution_api)

        assert proxy.content == ""

        evolution_api.scratchpad = "Test content"
        assert proxy.content == "Test content"

    def test_content_setter(self, evolution_api):
        """Test setting scratchpad content via proxy triggers persistence."""
        proxy = ScratchpadProxy(evolution_api)

        proxy.content = "New content via proxy"

        assert evolution_api.scratchpad == "New content via proxy"

    def test_append(self, evolution_api):
        """Test appending to scratchpad."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "Initial"
        proxy.append(" - appended")

        assert evolution_api.scratchpad == "Initial - appended"

    def test_clear(self, evolution_api):
        """Test clearing the scratchpad."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "Some content"
        proxy.clear()

        assert evolution_api.scratchpad == ""

    def test_str(self, evolution_api):
        """Test string conversion."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "String content"
        assert str(proxy) == "String content"

    def test_repr_empty(self, evolution_api):
        """Test repr for empty scratchpad."""
        proxy = ScratchpadProxy(evolution_api)

        assert repr(proxy) == "<scratchpad: empty>"

    def test_repr_short_content(self, evolution_api):
        """Test repr for short content."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "Short"
        result = repr(proxy)
        assert "<scratchpad: 5 chars>" in result
        assert "Short" in result

    def test_repr_long_content(self, evolution_api):
        """Test repr for long content (>100 chars) shows preview."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "x" * 150
        result = repr(proxy)
        assert "<scratchpad: 150 chars>" in result
        assert "..." in result

    def test_len(self, evolution_api):
        """Test len() on scratchpad."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "12345"
        assert len(proxy) == 5

    def test_contains(self, evolution_api):
        """Test 'in' operator on scratchpad."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "Hello world"
        assert "world" in proxy
        assert "xyz" not in proxy

    def test_add(self, evolution_api):
        """Test addition operator (returns new string, doesn't persist)."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "Hello"
        result = proxy + " world"

        # Result is a new string
        assert result == "Hello world"
        # Original scratchpad unchanged
        assert evolution_api.scratchpad == "Hello"

    def test_radd(self, evolution_api):
        """Test reverse addition operator (returns new string, doesn't persist)."""
        proxy = ScratchpadProxy(evolution_api)

        evolution_api.scratchpad = "world"
        result = "Hello " + proxy

        # Result is a new string
        assert result == "Hello world"
        # Original scratchpad unchanged
        assert evolution_api.scratchpad == "world"

    def test_persistence_on_content_set(self, evolution_api):
        """Test that setting content triggers logger.save_scratchpad."""
        proxy = ScratchpadProxy(evolution_api)

        # The evolution_api.update_scratchpad calls logger.save_scratchpad
        proxy.content = "Persisted content"

        # Verify content was set
        assert evolution_api.scratchpad == "Persisted content"

    def test_integration_with_repl_namespace(self, evolution_api):
        """Test that scratchpad proxy is accessible via get_api_functions."""
        funcs = evolution_api.get_api_functions()

        scratchpad = funcs["scratchpad"]
        assert isinstance(scratchpad, ScratchpadProxy)

        # Test operations through the namespace
        scratchpad.content = "Via namespace"
        assert evolution_api.scratchpad == "Via namespace"

        scratchpad.append(" - more")
        assert evolution_api.scratchpad == "Via namespace - more"
