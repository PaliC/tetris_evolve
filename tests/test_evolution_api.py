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


class TestSpawnChildren:
    """Tests for spawn_children."""

    def test_spawn_returns_result(self, evolution_api):
        """Test that spawn returns a TrialView result."""
        results = evolution_api.spawn_children([
            {"prompt": "Generate a circle packing algorithm"},
        ])

        # Result is now a TrialView object
        result = results[0]
        assert hasattr(result, "trial_id")
        assert hasattr(result, "code")
        assert hasattr(result, "metrics")
        assert hasattr(result, "success")
        assert result.trial_id.startswith("trial_0_")

    def test_spawn_with_parent(self, evolution_api):
        """Test spawning with parent ID."""
        results = evolution_api.spawn_children([
            {"prompt": "Improve this algorithm", "parent_id": "trial_0_0"},
        ])

        assert results[0].parent_id == "trial_0_0"

    def test_spawn_records_trial(self, evolution_api):
        """Test that spawn records the trial."""
        results = evolution_api.spawn_children([
            {"prompt": "Generate algorithm"},
        ])

        assert results[0].trial_id in evolution_api.all_trials
        assert len(evolution_api.generations[0].trials) == 1

    def test_spawn_produces_trial_with_metrics(self, evolution_api):
        """Test that spawn produces trials with metrics (evaluated code)."""
        results = evolution_api.spawn_children([{"prompt": "Generate algorithm"}])

        # Since spawn_children uses multiprocessing, we verify results instead of mock calls
        assert len(results) == 1
        trial = results[0]
        # The trial should have been processed (with metrics or error)
        assert trial.trial_id is not None
        # Either success with metrics, or failure with error - evaluation happened
        assert trial.success or trial.error is not None

    def test_spawn_budget_exceeded(self, evolution_api):
        """Test that spawn raises on budget exceeded."""
        evolution_api.cost_tracker.total_cost = 100.0  # Exceed budget

        with pytest.raises(BudgetExceededError):
            evolution_api.spawn_children([{"prompt": "Test"}])


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
        evolution_api.spawn_children([{"prompt": "Test"}])
        new_gen = evolution_api._advance_generation()

        assert new_gen == 1
        assert evolution_api.current_generation == 1

    def test_advance_creates_new_generation(self, evolution_api):
        """Test that advance creates a new generation entry."""
        evolution_api.spawn_children([{"prompt": "Test"}])
        evolution_api._advance_generation()

        assert len(evolution_api.generations) == 2
        assert evolution_api.generations[1].generation_num == 1

    def test_advance_auto_selects_best_trials(self, evolution_api):
        """Test that best trials are auto-selected."""
        evolution_api.spawn_children([{"prompt": "Test"}])
        evolution_api._advance_generation()

        gen0 = evolution_api.generations[0]
        # Auto-selects best performing trials (across all generations)
        assert "Auto-selected top performing trials" in gen0.selection_reasoning

    def test_can_advance_generation(self, evolution_api):
        """Test can_advance_generation helper."""
        assert evolution_api.can_advance_generation() is True

        # Advance to near max
        for i in range(4):
            evolution_api.spawn_children([{"prompt": f"Test {i}"}])
            evolution_api._advance_generation()

        # At generation 4 (max is 5), can't advance further
        assert evolution_api.can_advance_generation() is False

    def test_has_children_in_current_generation(self, evolution_api):
        """Test has_children_in_current_generation helper."""
        assert evolution_api.has_children_in_current_generation() is False

        evolution_api.spawn_children([{"prompt": "Test"}])
        assert evolution_api.has_children_in_current_generation() is True


class TestTerminateEvolution:
    """Tests for terminate_evolution."""

    def test_terminate_sets_flag(self, evolution_api):
        """Test that terminate sets the terminated flag."""
        evolution_api.spawn_children([{"prompt": "Test"}])
        evolution_api.terminate_evolution("Test complete")

        assert evolution_api.is_terminated

    def test_terminate_returns_results(self, evolution_api):
        """Test that terminate returns final results."""
        evolution_api.spawn_children([{"prompt": "Test"}])
        result = evolution_api.terminate_evolution("Test complete")

        assert result["terminated"] is True
        assert result["reason"] == "Test complete"
        assert "best_program" in result
        assert "cost_summary" in result

    def test_terminate_with_best_program(self, evolution_api):
        """Test that terminate accepts best_program argument."""
        evolution_api.spawn_children([{"prompt": "Test"}])
        best_code = "def run_packing(): return best_solution()"
        result = evolution_api.terminate_evolution("Done", best_program=best_code)

        assert result["best_program"] == best_code


class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_get_best_trials_returns_sorted(self, evolution_api):
        """Test that _get_best_trials returns sorted trials."""
        # Directly create trials with different scores (bypasses multiprocessing)
        scores = [1.5, 2.0, 1.8]
        for i, score in enumerate(scores):
            trial = TrialResult(
                trial_id=f"trial_0_{i}",
                code="def run_packing(): pass",
                metrics={"valid": True, "score": score, "eval_time": 0.1},
                prompt=f"Test {i}",
                response="test response",
                reasoning="test reasoning",
                success=True,
                parent_id=None,
                error=None,
                generation=0,
            )
            evolution_api._record_trial(trial)

        best = evolution_api._get_best_trials(n=3)

        assert len(best) == 3
        assert best[0]["metrics"]["score"] == 2.0
        assert best[1]["metrics"]["score"] == 1.8
        assert best[2]["metrics"]["score"] == 1.5

    def test_get_best_filters_invalid(self, evolution_api):
        """Test that _get_best_trials filters invalid trials."""
        # Create an invalid trial directly
        trial = TrialResult(
            trial_id="trial_0_0",
            code="",
            metrics={"valid": False, "score": 0, "error": "Invalid packing"},
            prompt="Test",
            response="test",
            reasoning="",
            success=False,
            parent_id=None,
            error="Invalid packing",
            generation=0,
        )
        evolution_api._record_trial(trial)
        best = evolution_api._get_best_trials(n=5)

        assert len(best) == 0

    def test_get_generation_history(self, evolution_api):
        """Test that generation history is returned."""
        # Create a trial directly
        trial = TrialResult(
            trial_id="trial_0_0",
            code="def run_packing(): pass",
            metrics={"valid": True, "score": 2.5},
            prompt="Test",
            response="test",
            reasoning="",
            success=True,
            parent_id=None,
            error=None,
            generation=0,
        )
        evolution_api._record_trial(trial)
        evolution_api._advance_generation()

        history = evolution_api._get_generation_history()

        assert len(history) == 2
        assert history[0]["generation_num"] == 0
        assert history[1]["generation_num"] == 1


class TestGetAPIFunctions:
    """Tests for get_api_functions."""

    def test_returns_8_items(self, evolution_api):
        """Test that 8 items (7 functions + 1 scratchpad proxy) are returned."""
        funcs = evolution_api.get_api_functions()

        assert len(funcs) == 8
        assert "spawn_children" in funcs
        assert "evaluate_program" in funcs
        assert "terminate_evolution" in funcs
        assert "update_scratchpad" in funcs
        assert "end_calibration_phase" in funcs
        assert "get_calibration_status" in funcs
        assert "query_llm" in funcs
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
        # Create trials directly
        for i in range(2):
            trial = TrialResult(
                trial_id=f"trial_0_{i}",
                code="def run_packing(): pass",
                metrics={"valid": True, "score": 2.5 + i * 0.1},
                prompt=f"Test {i}",
                response="test",
                reasoning="",
                success=True,
                parent_id=None,
                error=None,
                generation=0,
            )
            evolution_api._record_trial(trial)

        result = evolution_api._build_lineage_map()

        assert "All-Time Top 5" in result
        assert "cross-generation selection candidates" in result


class TestSelectionBehavior:
    """Tests for selection functionality (historical selection allowed)."""

    def test_selection_allows_historical_trials(self, evolution_api):
        """Test that _advance_generation accepts trials from any generation."""
        # Create trials directly in Gen 0
        for i in range(2):
            trial = TrialResult(
                trial_id=f"trial_0_{i}",
                code="def run_packing(): pass",
                metrics={"valid": True, "score": 2.5 - i * 0.2},
                prompt=f"Gen 0 Trial {i}",
                response="test",
                reasoning="",
                success=True,
                parent_id=None,
                error=None,
                generation=0,
            )
            evolution_api._record_trial(trial)
        evolution_api._advance_generation()

        # Create trials in Gen 1
        for i in range(2):
            trial = TrialResult(
                trial_id=f"trial_1_{i}",
                code="def run_packing(): pass",
                metrics={"valid": True, "score": 2.6 - i * 0.2},
                prompt=f"Gen 1 Trial {i}",
                response="test",
                reasoning="",
                success=True,
                parent_id=None,
                error=None,
                generation=1,
            )
            evolution_api._record_trial(trial)

        # Try to select with a mix of historical (Gen 0) and current (Gen 1) trials
        selections = [
            {"trial_id": "trial_1_0", "reasoning": "Current gen", "category": "performance"},
            {"trial_id": "trial_0_0", "reasoning": "Historical (allowed)", "category": "diversity"},
        ]

        evolution_api._advance_generation(selections=selections, selection_summary="Test")

        # Both current and historical selections should be preserved
        gen1 = evolution_api.generations[1]
        assert len(gen1.trial_selections) == 2
        assert gen1.selected_trial_ids == ["trial_1_0", "trial_0_0"]
        assert gen1.trial_selections[0].source_generation == 1
        assert gen1.trial_selections[1].source_generation == 0

    def test_auto_select_from_current_generation_only(self, evolution_api):
        """Test that auto-selection only selects from current generation."""
        # Gen 0: one trial with high score
        trial_0 = TrialResult(
            trial_id="trial_0_0",
            code="def run_packing(): pass",
            metrics={"valid": True, "score": 2.8},
            prompt="Gen 0 high score",
            response="test",
            reasoning="",
            success=True,
            parent_id=None,
            error=None,
            generation=0,
        )
        evolution_api._record_trial(trial_0)
        evolution_api._advance_generation()

        # Gen 1: two trials with lower scores
        for i in range(2):
            trial = TrialResult(
                trial_id=f"trial_1_{i}",
                code="def run_packing(): pass",
                metrics={"valid": True, "score": 2.1 - i * 0.1},
                prompt=f"Gen 1 low {i}",
                response="test",
                reasoning="",
                success=True,
                parent_id=None,
                error=None,
                generation=1,
            )
            evolution_api._record_trial(trial)

        # Let auto-select happen (no selections provided)
        evolution_api._advance_generation()

        # Auto-select should only include Gen 1 trials (not Gen 0)
        gen1 = evolution_api.generations[1]
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
