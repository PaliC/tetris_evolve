"""Tests for Root LLM Interface (Component 6)."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from tetris_evolve.llm.root_llm import (
    RootLLMInterface,
    PopulationMember,
    GenerationSummary,
    ResourceLimitError,
)
from tetris_evolve.llm.child_llm import ChildResult
from tetris_evolve.tracking.experiment_tracker import TrialData, GenerationStats


class TestDataClasses:
    """Tests for data classes."""

    def test_population_member_creation(self):
        """PopulationMember dataclass works."""
        member = PopulationMember(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            score=150.5,
            lines_cleared=10.2,
            survival_steps=500.0,
            code_preview="def choose_action(obs): return 5",
        )

        assert member.trial_id == "trial_001"
        assert member.score == 150.5
        assert member.parent_id is None

    def test_generation_summary_creation(self):
        """GenerationSummary dataclass works."""
        summary = GenerationSummary(
            generation=1,
            best_score=200.0,
            avg_score=100.0,
            num_trials=5,
            best_trial_id="trial_003",
        )

        assert summary.generation == 1
        assert summary.best_score == 200.0


class TestRootLLMInterfaceInit:
    """Tests for RootLLMInterface initialization."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        child_executor = Mock()
        evaluator = Mock()
        exp_tracker = Mock()
        cost_tracker = Mock()
        config = {
            "max_generations": 10,
            "max_children_per_generation": 5,
        }
        return child_executor, evaluator, exp_tracker, cost_tracker, config

    def test_init(self, mock_components):
        """Interface initializes correctly."""
        child_exec, evaluator, exp_tracker, cost_tracker, config = mock_components

        interface = RootLLMInterface(
            child_executor=child_exec,
            evaluator=evaluator,
            experiment_tracker=exp_tracker,
            cost_tracker=cost_tracker,
            config=config,
        )

        assert interface._max_generations == 10
        assert interface._max_children_per_gen == 5
        assert interface._current_generation == 1
        assert interface._children_this_gen == 0

    def test_init_default_limits(self, mock_components):
        """Uses default limits when not specified."""
        child_exec, evaluator, exp_tracker, cost_tracker, _ = mock_components

        interface = RootLLMInterface(
            child_executor=child_exec,
            evaluator=evaluator,
            experiment_tracker=exp_tracker,
            cost_tracker=cost_tracker,
            config={},
        )

        assert interface._max_generations == 50
        assert interface._max_children_per_gen == 20


class TestSpawnChild:
    """Tests for spawn_child_llm method."""

    @pytest.fixture
    def interface(self):
        """Create interface with mocked components."""
        child_executor = Mock()
        child_executor.llm_client = Mock(model="claude-haiku")
        child_executor.generate_and_validate.return_value = ChildResult(
            code="def choose_action(obs): return 5",
            reasoning="Simple strategy",
            raw_response="<code>...</code>",
            input_tokens=100,
            output_tokens=50,
            success=True,
            error=None,
        )

        evaluator = Mock()
        evaluator.evaluate.return_value = Mock(
            success=True,
            error=None,
            to_dict=lambda: {"avg_score": 100.0, "games_played": 10},
        )
        evaluator.max_steps = 10000
        evaluator.timeout_seconds = 30

        exp_tracker = Mock()
        exp_tracker._generate_trial_id.return_value = "trial_001"
        exp_tracker.get_trial.return_value = TrialData(
            trial_id="trial_parent",
            generation=1,
            parent_id=None,
            code="def choose_action(obs): return 0",
            prompt="p",
            reasoning="r",
            llm_response="l",
            metrics=None,
            timestamp=datetime.now(),
        )

        cost_tracker = Mock()

        return RootLLMInterface(
            child_executor=child_executor,
            evaluator=evaluator,
            experiment_tracker=exp_tracker,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 5},
        )

    def test_spawn_child_success(self, interface):
        """Spawns and evaluates code successfully."""
        result = interface.spawn_child_llm("Create a simple player")

        assert result["trial_id"] == "trial_001"
        assert "choose_action" in result["code"]
        assert result["metrics"] is not None
        assert result["success"] is True
        assert interface._children_this_gen == 1

    def test_spawn_child_with_parent(self, interface):
        """Uses parent code when specified."""
        result = interface.spawn_child_llm(
            "Improve this player",
            parent_id="trial_parent",
        )

        # Verify get_trial was called to get parent
        interface.exp_tracker.get_trial.assert_called_with("trial_parent")
        assert result["success"] is True

    def test_spawn_child_limit_exceeded(self, interface):
        """Raises error at children limit."""
        interface._children_this_gen = 5

        with pytest.raises(ResourceLimitError):
            interface.spawn_child_llm("Create player")

    def test_spawn_child_records_cost(self, interface):
        """Records LLM cost."""
        interface.spawn_child_llm("Create player")

        interface.cost_tracker.record_call.assert_called_once()
        call_args = interface.cost_tracker.record_call.call_args
        assert call_args.kwargs["role"] == "child"


class TestEvaluateProgram:
    """Tests for evaluate_program method."""

    @pytest.fixture
    def interface(self):
        """Create interface with mocked components."""
        evaluator = Mock()
        evaluator.max_steps = 10000
        evaluator.timeout_seconds = 30

        return RootLLMInterface(
            child_executor=Mock(),
            evaluator=evaluator,
            experiment_tracker=Mock(),
            cost_tracker=Mock(),
            config={},
        )

    def test_evaluate_program(self, interface):
        """Evaluates code without saving trial."""
        code = "def choose_action(obs): return 5"

        with patch("tetris_evolve.llm.root_llm.ProgramEvaluator") as mock_eval_cls:
            mock_eval = Mock()
            mock_eval.evaluate.return_value = Mock(
                to_dict=lambda: {"avg_score": 150.0, "success": True}
            )
            mock_eval_cls.return_value = mock_eval

            result = interface.evaluate_program(code, num_games=5)

        assert "avg_score" in result
        mock_eval_cls.assert_called_once()


class TestPopulationQueries:
    """Tests for population query methods."""

    @pytest.fixture
    def interface_with_trials(self):
        """Create interface with mock trials."""
        exp_tracker = Mock()
        exp_tracker.get_generation_trials.return_value = [
            TrialData(
                trial_id="trial_001",
                generation=1,
                parent_id=None,
                code="code1",
                prompt="p",
                reasoning="r",
                llm_response="l",
                metrics={"avg_score": 100.0, "avg_lines_cleared": 5.0, "avg_survival_steps": 200.0},
                timestamp=datetime.now(),
            ),
            TrialData(
                trial_id="trial_002",
                generation=1,
                parent_id="trial_001",
                code="code2",
                prompt="p",
                reasoning="r",
                llm_response="l",
                metrics={"avg_score": 150.0, "avg_lines_cleared": 8.0, "avg_survival_steps": 300.0},
                timestamp=datetime.now(),
            ),
        ]
        exp_tracker.get_trial.return_value = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code="def choose_action(obs): return 5",
            prompt="p",
            reasoning="r",
            llm_response="l",
            metrics=None,
            timestamp=datetime.now(),
        )

        return RootLLMInterface(
            child_executor=Mock(),
            evaluator=Mock(),
            experiment_tracker=exp_tracker,
            cost_tracker=Mock(),
            config={},
        )

    def test_get_population(self, interface_with_trials):
        """Returns current generation's trials."""
        population = interface_with_trials.get_population()

        assert len(population) == 2
        assert population[0].trial_id == "trial_001"
        assert population[1].score == 150.0

    def test_get_trial_code(self, interface_with_trials):
        """Returns full code for a trial."""
        code = interface_with_trials.get_trial_code("trial_001")

        assert "choose_action" in code


class TestGenerationHistory:
    """Tests for generation history methods."""

    @pytest.fixture
    def interface_with_history(self):
        """Create interface with mock history."""
        exp_tracker = Mock()

        def mock_get_trials(gen):
            if gen == 1:
                return [
                    TrialData(
                        trial_id="trial_001",
                        generation=1,
                        parent_id=None,
                        code="c",
                        prompt="p",
                        reasoning="r",
                        llm_response="l",
                        metrics={"avg_score": 100.0},
                        timestamp=datetime.now(),
                    ),
                ]
            elif gen == 2:
                return [
                    TrialData(
                        trial_id="trial_002",
                        generation=2,
                        parent_id="trial_001",
                        code="c",
                        prompt="p",
                        reasoning="r",
                        llm_response="l",
                        metrics={"avg_score": 150.0},
                        timestamp=datetime.now(),
                    ),
                ]
            return []

        exp_tracker.get_generation_trials.side_effect = mock_get_trials

        interface = RootLLMInterface(
            child_executor=Mock(),
            evaluator=Mock(),
            experiment_tracker=exp_tracker,
            cost_tracker=Mock(),
            config={},
        )
        interface._current_generation = 3

        return interface

    def test_get_generation_history(self, interface_with_history):
        """Returns all generation stats."""
        history = interface_with_history.get_generation_history()

        assert len(history) == 2
        assert history[0].generation == 1
        assert history[0].best_score == 100.0
        assert history[1].generation == 2
        assert history[1].best_score == 150.0

    def test_get_improvement_rate(self, interface_with_history):
        """Calculates improvement rate correctly."""
        rate = interface_with_history.get_improvement_rate()

        # 50% improvement from 100 to 150
        assert rate == 0.5


class TestAdvanceGeneration:
    """Tests for advance_generation method."""

    @pytest.fixture
    def interface(self):
        """Create interface."""
        exp_tracker = Mock()
        exp_tracker.get_generation_trials.return_value = [
            TrialData(
                trial_id="trial_001",
                generation=1,
                parent_id=None,
                code="c",
                prompt="p",
                reasoning="r",
                llm_response="l",
                metrics={"avg_score": 100.0},
                timestamp=datetime.now(),
            ),
        ]

        return RootLLMInterface(
            child_executor=Mock(),
            evaluator=Mock(),
            experiment_tracker=exp_tracker,
            cost_tracker=Mock(),
            config={"max_generations": 5},
        )

    def test_advance_generation(self, interface):
        """Advances to next generation."""
        interface._children_this_gen = 3

        new_gen = interface.advance_generation(
            selected_trial_ids=["trial_001"],
            reasoning="Selected best performer",
        )

        assert new_gen == 2
        assert interface._current_generation == 2
        assert interface._children_this_gen == 0
        interface.exp_tracker.complete_generation.assert_called_once()
        interface.exp_tracker.start_generation.assert_called_with(2)

    def test_advance_generation_limit(self, interface):
        """Raises at max generation."""
        interface._current_generation = 5

        with pytest.raises(ResourceLimitError):
            interface.advance_generation(["trial_001"], "reason")


class TestTermination:
    """Tests for termination methods."""

    @pytest.fixture
    def interface(self):
        """Create interface."""
        exp_tracker = Mock()
        exp_tracker.get_best_trial.return_value = TrialData(
            trial_id="trial_005",
            generation=2,
            parent_id=None,
            code="def choose_action(obs): return 5",
            prompt="p",
            reasoning="r",
            llm_response="l",
            metrics={"avg_score": 250.0},
            timestamp=datetime.now(),
        )

        cost_tracker = Mock()
        cost_tracker.get_total_cost.return_value = 5.50

        return RootLLMInterface(
            child_executor=Mock(),
            evaluator=Mock(),
            experiment_tracker=exp_tracker,
            cost_tracker=cost_tracker,
            config={},
        )

    def test_terminate_evolution(self, interface):
        """Returns summary on termination."""
        interface._current_generation = 3

        result = interface.terminate_evolution("Converged")

        assert result["terminated"] is True
        assert result["reason"] == "Converged"
        assert result["generations_completed"] == 3
        assert result["total_cost_usd"] == 5.50
        assert result["best_trial_id"] == "trial_005"
        assert interface.is_terminated is True

    def test_is_terminated(self, interface):
        """Tracks termination state."""
        assert interface.is_terminated is False

        interface.terminate_evolution("Done")

        assert interface.is_terminated is True


class TestResourceQueries:
    """Tests for resource query methods."""

    @pytest.fixture
    def interface(self):
        """Create interface."""
        cost_tracker = Mock()
        cost_tracker.get_remaining_budget.return_value = 45.50
        cost_tracker.get_total_cost.return_value = 4.50
        cost_tracker.max_cost_usd = 50.0

        return RootLLMInterface(
            child_executor=Mock(),
            evaluator=Mock(),
            experiment_tracker=Mock(),
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 8},
        )

    def test_get_cost_remaining(self, interface):
        """Returns correct remaining budget."""
        remaining = interface.get_cost_remaining()

        assert remaining == 45.50

    def test_get_limits(self, interface):
        """Returns all limits and current state."""
        interface._children_this_gen = 3

        limits = interface.get_limits()

        assert limits["max_generations"] == 10
        assert limits["max_children_per_gen"] == 8
        assert limits["current_gen"] == 1
        assert limits["children_this_gen"] == 3
        assert limits["max_cost_usd"] == 50.0
        assert limits["cost_remaining_usd"] == 45.50
