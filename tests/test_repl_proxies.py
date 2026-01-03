"""
Tests for TrialView, TrialsProxy, and REPL namespace injection.
"""

from unittest.mock import MagicMock

import pytest

from mango_evolve import (
    CostTracker,
    EvolutionAPI,
    ExperimentLogger,
    TrialResult,
)
from mango_evolve.evolution_api import TrialsProxy, TrialView
from mango_evolve.llm import MockLLMClient
from mango_evolve.repl import REPLEnvironment


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "valid": True,
        "score": 2.5,
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
        max_children_per_generation=10,
        default_child_llm_alias=sample_config.default_child_llm_alias,
    )

    # Inject mock LLM clients with plenty of responses
    for alias in child_llm_configs:
        mock_client = MockLLMClient(
            model=child_llm_configs[alias].model,
            cost_tracker=cost_tracker,
            llm_type=f"child:{alias}",
            responses=["```python\ndef solve(): return 1.0\n```"] * 100,  # Many responses
        )
        api.child_llm_clients[alias] = mock_client

    api.end_calibration_phase()
    return api


@pytest.fixture
def sample_trial_result():
    """Create a sample TrialResult for testing."""
    return TrialResult(
        trial_id="trial_0_1",
        code="def solve(): return 1.0",
        metrics={"valid": True, "score": 2.5},
        prompt="Generate a solution",
        response="Here is the solution...",
        reasoning="Using optimization",
        success=True,
        parent_id=None,
        error=None,
        generation=0,
        model_alias="default",
        model_config={"model": "test-model"},
    )


class TestTrialView:
    """Tests for TrialView dataclass."""

    def test_from_trial_result(self, sample_trial_result):
        """Test creating TrialView from TrialResult."""
        view = TrialView.from_trial_result(sample_trial_result)

        assert view.trial_id == "trial_0_1"
        assert view.code == "def solve(): return 1.0"
        assert view.score == 2.5
        assert view.success is True
        assert view.generation == 0
        assert view.parent_id is None
        assert view.reasoning == "Using optimization"
        assert view.error is None
        assert view.model_alias == "default"
        assert view.metrics == {"valid": True, "score": 2.5}

    def test_score_is_zero_for_failed_trial(self):
        """Test that score is 0 for failed trials."""
        trial = TrialResult(
            trial_id="trial_0_2",
            code="",
            metrics={"score": 1.0},  # Has a score but failed
            prompt="test",
            response="test",
            reasoning="",
            success=False,
            error="Syntax error",
            generation=0,
        )
        view = TrialView.from_trial_result(trial)
        assert view.score == 0

    def test_repr(self, sample_trial_result):
        """Test string representation."""
        view = TrialView.from_trial_result(sample_trial_result)
        repr_str = repr(view)
        assert "trial_0_1" in repr_str
        assert "✓" in repr_str
        assert "2.5" in repr_str

    def test_repr_failed(self):
        """Test repr for failed trial."""
        trial = TrialResult(
            trial_id="trial_0_2",
            code="",
            metrics={},
            prompt="test",
            response="test",
            reasoning="",
            success=False,
            error="Error",
            generation=0,
        )
        view = TrialView.from_trial_result(trial)
        assert "✗" in repr(view)

    def test_to_dict(self, sample_trial_result):
        """Test converting TrialView to dict."""
        view = TrialView.from_trial_result(sample_trial_result)
        d = view.to_dict()

        assert isinstance(d, dict)
        assert d["trial_id"] == "trial_0_1"
        assert d["score"] == 2.5
        assert d["success"] is True


class TestTrialsProxy:
    """Tests for TrialsProxy class."""

    def test_len_empty(self, evolution_api):
        """Test length when no trials."""
        proxy = TrialsProxy(evolution_api)
        assert len(proxy) == 0

    def test_len_with_trials(self, evolution_api):
        """Test length with trials."""
        # Spawn some trials
        evolution_api.spawn_child_llm(prompt="test 1")
        evolution_api.spawn_child_llm(prompt="test 2")

        proxy = TrialsProxy(evolution_api)
        assert len(proxy) == 2

    def test_iter(self, evolution_api):
        """Test iteration over trials."""
        evolution_api.spawn_child_llm(prompt="test 1")
        evolution_api.spawn_child_llm(prompt="test 2")

        proxy = TrialsProxy(evolution_api)
        trials = list(proxy)

        assert len(trials) == 2
        assert all(isinstance(t, TrialView) for t in trials)

    def test_getitem(self, evolution_api):
        """Test accessing trial by ID."""
        result = evolution_api.spawn_child_llm(prompt="test")
        trial_id = result.trial_id

        proxy = TrialsProxy(evolution_api)
        trial = proxy[trial_id]

        assert isinstance(trial, TrialView)
        assert trial.trial_id == trial_id

    def test_getitem_not_found(self, evolution_api):
        """Test accessing non-existent trial raises KeyError."""
        proxy = TrialsProxy(evolution_api)
        with pytest.raises(KeyError):
            _ = proxy["nonexistent_trial"]

    def test_contains(self, evolution_api):
        """Test 'in' operator."""
        result = evolution_api.spawn_child_llm(prompt="test")
        trial_id = result.trial_id

        proxy = TrialsProxy(evolution_api)
        assert trial_id in proxy
        assert "nonexistent_trial" not in proxy

    def test_repr(self, evolution_api):
        """Test string representation."""
        proxy = TrialsProxy(evolution_api)
        repr_str = repr(proxy)
        assert "Trials:" in repr_str
        assert "total" in repr_str
        assert "successful" in repr_str


class TestTrialsProxyFilter:
    """Tests for TrialsProxy.filter() method."""

    @pytest.fixture
    def api_with_trials(self, evolution_api, mock_evaluator):
        """Create API with multiple trials for filtering tests."""
        # Create some successful trials
        mock_evaluator.evaluate.return_value = {
            "valid": True,
            "score": 2.5,
            "eval_time": 0.1,
        }
        evolution_api.spawn_child_llm(prompt="test 1")  # trial_0_0
        evolution_api.spawn_child_llm(prompt="test 2")  # trial_0_1

        # Create a failed trial
        mock_evaluator.evaluate.return_value = {
            "valid": False,
            "score": 0,
            "error": "Failed",
        }
        evolution_api.spawn_child_llm(prompt="test 3")  # trial_0_2

        # Reset to success
        mock_evaluator.evaluate.return_value = {
            "valid": True,
            "score": 2.8,
            "eval_time": 0.1,
        }

        return evolution_api

    def test_filter_by_success(self, api_with_trials):
        """Test filtering by success status."""
        proxy = TrialsProxy(api_with_trials)

        successful = proxy.filter(success=True)
        assert len(successful) == 2
        assert all(t.success for t in successful)

        failed = proxy.filter(success=False)
        assert len(failed) == 1
        assert not failed[0].success

    def test_filter_by_generation(self, api_with_trials):
        """Test filtering by generation number."""
        proxy = TrialsProxy(api_with_trials)

        gen0 = proxy.filter(generation=0)
        assert len(gen0) == 3

        gen1 = proxy.filter(generation=1)
        assert len(gen1) == 0

    def test_filter_by_sort_ascending(self, api_with_trials, mock_evaluator):
        """Test sorting by score ascending."""
        # Create a trial with higher score
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 3.0}
        api_with_trials.spawn_child_llm(prompt="high score")

        proxy = TrialsProxy(api_with_trials)
        sorted_trials = proxy.filter(success=True, sort_by="score")

        assert len(sorted_trials) == 3
        scores = [t.score for t in sorted_trials]
        assert scores == sorted(scores)  # Ascending

    def test_filter_by_sort_descending(self, api_with_trials, mock_evaluator):
        """Test sorting by score descending."""
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 3.0}
        api_with_trials.spawn_child_llm(prompt="high score")

        proxy = TrialsProxy(api_with_trials)
        sorted_trials = proxy.filter(success=True, sort_by="-score")

        assert len(sorted_trials) == 3
        scores = [t.score for t in sorted_trials]
        assert scores == sorted(scores, reverse=True)  # Descending

    def test_filter_with_limit(self, api_with_trials, mock_evaluator):
        """Test limiting results."""
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 3.0}
        api_with_trials.spawn_child_llm(prompt="high score")

        proxy = TrialsProxy(api_with_trials)
        limited = proxy.filter(success=True, sort_by="-score", limit=2)

        assert len(limited) == 2

    def test_filter_by_predicate(self, api_with_trials):
        """Test filtering with custom predicate."""
        proxy = TrialsProxy(api_with_trials)

        result = proxy.filter(predicate=lambda t: t.score >= 2.5)
        assert len(result) == 2
        assert all(t.score >= 2.5 for t in result)

    def test_filter_by_parent_id(self, api_with_trials, mock_evaluator):
        """Test filtering by parent_id."""
        # Get first trial as parent
        first_trial = list(api_with_trials.all_trials.values())[0]
        parent_id = first_trial.trial_id

        # Spawn a child with parent
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.6}
        api_with_trials.spawn_child_llm(prompt="child", parent_id=parent_id)

        proxy = TrialsProxy(api_with_trials)
        children = proxy.filter(parent_id=parent_id)

        assert len(children) == 1
        assert children[0].parent_id == parent_id

    def test_filter_combined(self, api_with_trials):
        """Test combining multiple filters."""
        proxy = TrialsProxy(api_with_trials)

        result = proxy.filter(
            success=True,
            generation=0,
            sort_by="-score",
            limit=1,
        )

        assert len(result) == 1
        assert result[0].success
        assert result[0].generation == 0

    def test_filter_descendant_of(self, api_with_trials, mock_evaluator):
        """Test filtering by descendant_of."""
        # Get first trial as root
        root_id = list(api_with_trials.all_trials.keys())[0]

        # Spawn a child
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.6}
        child_result = api_with_trials.spawn_child_llm(prompt="child", parent_id=root_id)
        child_id = child_result.trial_id

        # Spawn a grandchild
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.7}
        api_with_trials.spawn_child_llm(prompt="grandchild", parent_id=child_id)

        proxy = TrialsProxy(api_with_trials)
        descendants = proxy.filter(descendant_of=root_id)

        assert len(descendants) == 2  # child and grandchild

    def test_filter_ancestor_of(self, api_with_trials, mock_evaluator):
        """Test filtering by ancestor_of."""
        # Get first trial as root
        root_id = list(api_with_trials.all_trials.keys())[0]

        # Spawn a child
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.6}
        child_result = api_with_trials.spawn_child_llm(prompt="child", parent_id=root_id)
        child_id = child_result.trial_id

        # Spawn a grandchild
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.7}
        grandchild_result = api_with_trials.spawn_child_llm(
            prompt="grandchild", parent_id=child_id
        )
        grandchild_id = grandchild_result.trial_id

        proxy = TrialsProxy(api_with_trials)
        ancestors = proxy.filter(ancestor_of=grandchild_id)

        assert len(ancestors) == 2  # root and child
        ancestor_ids = {a.trial_id for a in ancestors}
        assert root_id in ancestor_ids
        assert child_id in ancestor_ids


class TestREPLNamespaceInjection:
    """Tests for injecting trials into REPL namespace."""

    def test_get_repl_namespace_includes_trials(self, evolution_api):
        """Test that get_repl_namespace() includes trials variable."""
        namespace = evolution_api.get_repl_namespace()

        assert "trials" in namespace
        assert isinstance(namespace["trials"], TrialsProxy)

    def test_get_repl_namespace_includes_functions(self, evolution_api):
        """Test that get_repl_namespace() includes all API functions."""
        namespace = evolution_api.get_repl_namespace()

        # Check that all API functions are present
        assert "spawn_child_llm" in namespace
        assert "spawn_children_parallel" in namespace
        assert "evaluate_program" in namespace
        assert "terminate_evolution" in namespace
        assert "get_top_trials" in namespace
        assert "get_trial_code" in namespace
        assert "update_scratchpad" in namespace

    def test_repl_can_access_trials(self, evolution_api):
        """Test that REPL can access trials variable."""
        # Spawn a trial
        evolution_api.spawn_child_llm(prompt="test")

        # Create REPL with namespace
        repl = REPLEnvironment(namespace=evolution_api.get_repl_namespace())

        # Execute code that accesses trials
        result = repl.execute("len(trials)")
        assert result.success
        assert result.return_value == 1

    def test_repl_can_filter_trials(self, evolution_api, mock_evaluator):
        """Test that REPL can use trials.filter()."""
        # Spawn some trials
        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.5}
        evolution_api.spawn_child_llm(prompt="test 1")

        mock_evaluator.evaluate.return_value = {"valid": True, "score": 2.8}
        evolution_api.spawn_child_llm(prompt="test 2")

        repl = REPLEnvironment(namespace=evolution_api.get_repl_namespace())

        # Filter trials - first execute assignment, then get value
        repl.execute("top = trials.filter(success=True, sort_by='-score', limit=1)")
        result = repl.execute("top[0].score")
        assert result.success
        assert result.return_value == 2.8

    def test_repl_can_access_trial_by_id(self, evolution_api):
        """Test that REPL can access trial by ID."""
        trial_result = evolution_api.spawn_child_llm(prompt="test")
        trial_id = trial_result.trial_id

        repl = REPLEnvironment(namespace=evolution_api.get_repl_namespace())

        result = repl.execute(f'trials["{trial_id}"].trial_id')
        assert result.success
        assert result.return_value == trial_id


class TestSpawnReturnsTrialView:
    """Tests for spawn functions returning TrialView."""

    def test_spawn_child_llm_returns_trial_view(self, evolution_api):
        """Test that spawn_child_llm returns TrialView."""
        result = evolution_api.spawn_child_llm(prompt="test")

        assert isinstance(result, TrialView)
        assert hasattr(result, "trial_id")
        assert hasattr(result, "score")
        assert hasattr(result, "code")
        assert hasattr(result, "success")

    def test_spawn_children_parallel_returns_trial_views(self, evolution_api):
        """Test that spawn_children_parallel returns list of TrialView."""
        children = [
            {"prompt": "test 1"},
            {"prompt": "test 2"},
        ]
        results = evolution_api.spawn_children_parallel(children)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, TrialView) for r in results)

    def test_trial_view_to_dict_backward_compat(self, evolution_api):
        """Test that TrialView.to_dict() provides backward compatibility."""
        result = evolution_api.spawn_child_llm(prompt="test")

        # Convert to dict for backward compatibility
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "trial_id" in d
        assert "score" in d
        assert "code" in d
        assert "success" in d
