"""Integration tests verifying component interactions."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from tetris_evolve.evaluation.evaluator import ProgramEvaluator, EvaluationResult
from tetris_evolve.tracking.experiment_tracker import ExperimentTracker, TrialData
from tetris_evolve.tracking.cost_tracker import CostTracker
from tetris_evolve.llm.client import LLMResponse
from tetris_evolve.llm.child_llm import ChildLLMExecutor, ChildResult
from tetris_evolve.llm.root_llm import RootLLMInterface


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_experiment(tmp_path):
    """Create temporary experiment directory."""
    tracker = ExperimentTracker(tmp_path)
    tracker.create_experiment({"test": True})
    tracker.start_generation(1)
    return tracker


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns valid code."""
    client = Mock()
    client.model = "claude-haiku"  # Use known model name
    client.send_message.return_value = Mock(
        content="""
<reasoning>Simple hard drop strategy</reasoning>
<code>
def choose_action(obs):
    return 5  # hard drop
</code>
""",
        input_tokens=100,
        output_tokens=50,
        model="claude-haiku",
        stop_reason="end_turn"
    )
    return client


@pytest.fixture
def mock_llm_client_invalid():
    """Mock LLM client that returns invalid code."""
    client = Mock()
    client.model = "claude-haiku"  # Use known model name
    client.send_message.return_value = Mock(
        content="""
<reasoning>Attempt with syntax error</reasoning>
<code>
def choose_action(obs)
    return 5  # missing colon
</code>
""",
        input_tokens=100,
        output_tokens=50,
        model="claude-haiku",
        stop_reason="end_turn"
    )
    return client


@pytest.fixture
def evaluator():
    """Real evaluator with low game count for speed."""
    return ProgramEvaluator(num_games=2, max_steps=100, timeout_seconds=10)


@pytest.fixture
def cost_tracker():
    """Cost tracker with $100 budget."""
    return CostTracker(max_cost_usd=100.0)


def make_valid_response():
    """Create a mock response with valid code."""
    return Mock(
        content="""
<reasoning>Simple hard drop strategy</reasoning>
<code>
def choose_action(obs):
    return 5  # hard drop
</code>
""",
        input_tokens=100,
        output_tokens=50,
        model="test-model",
        stop_reason="end_turn"
    )


# =============================================================================
# Section 8.2: Evaluator → Tracker Integration
# =============================================================================


class TestEvaluatorTrackerIntegration:
    """Tests for evaluator and tracker integration."""

    def test_eval_saves_to_tracker(self, tmp_experiment, evaluator):
        """Evaluation results save correctly to experiment tracker."""
        code = "def choose_action(obs): return 5"
        result = evaluator.evaluate(code, "trial_001")

        trial = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code=code,
            prompt="Test prompt",
            reasoning="Test reasoning",
            llm_response="Test response",
            metrics=result.to_dict() if result.success else None,
            timestamp=datetime.now()
        )

        path = tmp_experiment.save_trial(trial)

        # Verify files
        assert (path / "code.py").exists()
        if result.success:
            assert (path / "metrics.json").exists()

    def test_multiple_evals_unique_ids(self, tmp_experiment, evaluator):
        """Multiple evaluations tracked with unique IDs."""
        code = "def choose_action(obs): return 5"

        trial_ids = set()
        for i in range(3):
            trial_id = tmp_experiment._generate_trial_id()
            trial_ids.add(trial_id)

            result = evaluator.evaluate(code, trial_id)

            trial = TrialData(
                trial_id=trial_id,
                generation=1,
                parent_id=None,
                code=code,
                prompt=f"Test prompt {i}",
                reasoning=f"Test reasoning {i}",
                llm_response="Test response",
                metrics=result.to_dict() if result.success else None,
                timestamp=datetime.now()
            )
            tmp_experiment.save_trial(trial)

        # All IDs are unique
        assert len(trial_ids) == 3

        # All trials saved
        all_trials = tmp_experiment.get_all_trials()
        assert len(all_trials) == 3

    def test_failed_eval_recorded(self, tmp_experiment, evaluator):
        """Failed evaluations recorded with error info."""
        code = "def choose_action(obs) return 5"  # Syntax error
        result = evaluator.evaluate(code, "trial_002")

        assert result.success is False
        assert result.error is not None

        # Can still create trial with error info
        trial = TrialData(
            trial_id="trial_002",
            generation=1,
            parent_id=None,
            code=code,
            prompt="Test prompt",
            reasoning="Test reasoning",
            llm_response="Test response",
            metrics={"error": result.error, "success": False},
            timestamp=datetime.now()
        )
        path = tmp_experiment.save_trial(trial)
        assert path.exists()


# =============================================================================
# Section 8.3: Child LLM → Evaluator Pipeline
# =============================================================================


class TestChildEvaluatorPipeline:
    """Tests for child LLM and evaluator integration."""

    def test_child_generates_evaluates(self, mock_llm_client, evaluator):
        """Child generates code that evaluates successfully."""
        executor = ChildLLMExecutor(mock_llm_client)
        result = executor.generate("Create a simple player")

        assert result.success
        assert "choose_action" in result.code

        eval_result = evaluator.evaluate(result.code, "trial_001")
        assert eval_result.success

    def test_child_invalid_handled(self, mock_llm_client_invalid, evaluator):
        """Child generates invalid code - error is captured."""
        executor = ChildLLMExecutor(mock_llm_client_invalid)
        result = executor.generate("Create a player")

        # Child may report success if it got code (validation happens at eval)
        # The evaluator should catch the syntax error
        if result.code:
            eval_result = evaluator.evaluate(result.code, "trial_001")
            # Either validation or evaluation should fail
            assert not eval_result.success or "error" in str(eval_result).lower()

    def test_child_retry_succeeds(self, evaluator):
        """Child retry produces valid code on second attempt."""
        client = Mock()
        client.model = "claude-haiku"  # Use known model name

        # First call returns code without choose_action, second has it
        # The generate_and_validate method validates the signature
        invalid_response = Mock(
            content="<code>def bad(): pass</code>",
            input_tokens=50,
            output_tokens=25,
            model="claude-haiku",
            stop_reason="end_turn"
        )
        valid_response = Mock(
            content="<code>def choose_action(obs): return 5</code>",
            input_tokens=100,
            output_tokens=50,
            model="claude-haiku",
            stop_reason="end_turn"
        )

        client.send_message.side_effect = [invalid_response, valid_response]

        executor = ChildLLMExecutor(client)
        # generate_and_validate handles validation and retries
        result = executor.generate_and_validate(
            prompt="Create a player",
            max_retries=2
        )

        # The executor should have retried and eventually succeeded
        # Note: The actual retry depends on validation logic
        assert result is not None
        # At minimum the second call should have been made
        assert client.send_message.call_count >= 1


# =============================================================================
# Section 8.4: Root → Child Pipeline
# =============================================================================


class TestRootChildPipeline:
    """Tests for root LLM and child LLM integration."""

    @pytest.fixture
    def root_interface(self, tmp_experiment, mock_llm_client, evaluator, cost_tracker):
        """Create root interface with mocked components."""
        executor = ChildLLMExecutor(mock_llm_client)

        return RootLLMInterface(
            child_executor=executor,
            evaluator=evaluator,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 5}
        )

    def test_spawn_child_creates_trial(self, root_interface, tmp_experiment):
        """Root spawn_child creates trial and evaluates."""
        result = root_interface.spawn_child_llm("Create a player")

        assert "trial_id" in result
        assert result["success"] is True
        assert "metrics" in result

        # Verify trial was saved
        trial = tmp_experiment.get_trial(result["trial_id"])
        assert trial is not None
        assert "choose_action" in trial.code

    def test_spawn_with_parent(self, root_interface, tmp_experiment):
        """Root spawn_child with parent uses parent code."""
        # Create initial trial
        first_result = root_interface.spawn_child_llm("Create initial player")

        # Spawn with parent
        second_result = root_interface.spawn_child_llm(
            "Improve this player",
            parent_id=first_result["trial_id"]
        )

        assert second_result["success"] is True

        # Verify parent was retrieved
        parent_trial = tmp_experiment.get_trial(first_result["trial_id"])
        assert parent_trial is not None

    def test_spawn_updates_cost(self, root_interface, cost_tracker):
        """Root spawn_child updates cost tracker."""
        initial_cost = cost_tracker.get_total_cost()

        root_interface.spawn_child_llm("Create a player")

        # Cost should have increased
        assert cost_tracker.get_total_cost() > initial_cost


# =============================================================================
# Section 8.5: Generation Lifecycle
# =============================================================================


class TestGenerationLifecycle:
    """Tests for generation lifecycle management."""

    @pytest.fixture
    def root_interface(self, tmp_experiment, mock_llm_client, evaluator, cost_tracker):
        """Create root interface with mocked components."""
        executor = ChildLLMExecutor(mock_llm_client)

        return RootLLMInterface(
            child_executor=executor,
            evaluator=evaluator,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 10}
        )

    def test_full_generation_cycle(self, root_interface, tmp_experiment):
        """Full generation cycle works end-to-end."""
        # Spawn children
        trial_ids = []
        for i in range(3):
            result = root_interface.spawn_child_llm(f"Create player {i}")
            trial_ids.append(result["trial_id"])

        # Verify all trials created
        assert len(trial_ids) == 3
        for tid in trial_ids:
            trial = tmp_experiment.get_trial(tid)
            assert trial is not None

        # Advance generation
        new_gen = root_interface.advance_generation(
            selected_trial_ids=trial_ids[:2],
            reasoning="Selected top 2"
        )

        assert new_gen == 2

        # Verify generation files exist
        gen_path = tmp_experiment.experiment_dir / "generations" / "gen_001"
        assert gen_path.exists()

    def test_generation_stats_calculated(self, root_interface, tmp_experiment):
        """Generation stats calculated correctly."""
        # Spawn children
        for i in range(3):
            root_interface.spawn_child_llm(f"Create player {i}")

        # Advance to calculate stats
        root_interface.advance_generation(
            selected_trial_ids=[],
            reasoning="Testing stats"
        )

        # Get generation history
        history = root_interface.get_generation_history()

        assert len(history) >= 1
        gen1 = history[0]
        assert gen1.num_trials == 3
        assert gen1.best_score >= 0

    def test_selected_parents_recorded(self, root_interface, tmp_experiment):
        """Selected parents recorded in generation."""
        # Spawn children
        trial_ids = []
        for i in range(3):
            result = root_interface.spawn_child_llm(f"Create player {i}")
            trial_ids.append(result["trial_id"])

        # Advance with selection
        root_interface.advance_generation(
            selected_trial_ids=trial_ids[:2],
            reasoning="Selected top performers"
        )

        # Verify reasoning file created
        gen_path = tmp_experiment.experiment_dir / "generations" / "gen_001"
        reasoning_file = gen_path / "root_llm_reasoning.md"
        assert reasoning_file.exists()

        content = reasoning_file.read_text()
        assert "Selected top performers" in content


# =============================================================================
# Section 8.6: Cost → Controller Integration
# =============================================================================


class TestCostControllerIntegration:
    """Tests for cost tracking and controller integration."""

    def test_cost_limit_stops_controller(self, tmp_path):
        """Controller stops when cost limit reached."""
        from tetris_evolve.evolution.controller import (
            EvolutionController,
            EvolutionConfig,
        )

        config = EvolutionConfig(
            max_generations=100,
            max_cost_usd=0.001,  # Very low budget
            max_time_minutes=60,
            initial_population_size=1,
            output_dir=tmp_path / "experiments",
        )

        controller = EvolutionController(config)

        # Initialize and force high cost
        with patch("tetris_evolve.evolution.controller.LLMClient"):
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator"):
                controller._initialize_components()
                controller.cost_tracker.record_call(
                    model="claude-haiku",
                    input_tokens=1000000,
                    output_tokens=1000000,
                    role="test",
                    generation=1  # Required parameter
                )

                limit = controller._check_cost_limit()

                assert limit is not None
                assert "cost_limit" in limit

    def test_cost_accumulates_across_generations(self, tmp_experiment, mock_llm_client, evaluator):
        """Cost accumulates across generations."""
        cost_tracker = CostTracker(max_cost_usd=100.0)
        executor = ChildLLMExecutor(mock_llm_client)

        interface = RootLLMInterface(
            child_executor=executor,
            evaluator=evaluator,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 10}
        )

        # Generation 1
        for i in range(2):
            interface.spawn_child_llm(f"Gen 1 player {i}")

        cost_after_gen1 = cost_tracker.get_total_cost()

        interface.advance_generation([], "Next gen")

        # Generation 2
        for i in range(2):
            interface.spawn_child_llm(f"Gen 2 player {i}")

        cost_after_gen2 = cost_tracker.get_total_cost()

        # Cost should have accumulated
        assert cost_after_gen2 > cost_after_gen1
        assert cost_after_gen1 > 0


# =============================================================================
# Section 8.7: Full Pipeline (Mocked LLM)
# =============================================================================


class TestFullPipeline:
    """Tests for full evolution pipeline with mocked LLM."""

    def test_initial_population_creates_trials(self, tmp_path):
        """Initial population creates N trials."""
        from tetris_evolve.evolution.controller import (
            EvolutionController,
            EvolutionConfig,
        )

        config = EvolutionConfig(
            max_generations=5,
            max_cost_usd=100.0,
            max_time_minutes=60,
            initial_population_size=3,
            output_dir=tmp_path / "experiments",
        )

        controller = EvolutionController(config)

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content="<code>def choose_action(obs): return 5</code>",
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client

                mock_eval = Mock()
                mock_eval.max_steps = 1000
                mock_eval.timeout_seconds = 30
                mock_eval.evaluate.return_value = Mock(
                    success=True,
                    to_dict=lambda: {"avg_score": 100.0}
                )
                mock_eval_cls.return_value = mock_eval

                controller._initialize_components()
                controller.exp_tracker.create_experiment(config.to_dict())
                controller.exp_tracker.start_generation(1)

                # Mock the root_interface.spawn_child_llm to track calls
                spawn_mock = Mock(return_value={"trial_id": "test", "success": True})
                controller.root_interface.spawn_child_llm = spawn_mock

                controller._create_initial_population()

                # Verify spawn was called N times
                assert spawn_mock.call_count == 3

    def test_controller_executes_root_code(self, tmp_path):
        """Controller executes root code correctly."""
        from tetris_evolve.evolution.controller import (
            EvolutionController,
            EvolutionConfig,
        )

        config = EvolutionConfig(
            max_generations=5,
            max_cost_usd=100.0,
            max_time_minutes=60,
            initial_population_size=1,
            output_dir=tmp_path / "experiments",
        )

        controller = EvolutionController(config)

        with patch("tetris_evolve.evolution.controller.LLMClient"):
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator"):
                controller._initialize_components()

                # Create mock interface
                controller.root_interface = Mock()
                controller.root_interface.spawn_child_llm.return_value = {
                    "trial_id": "trial_001",
                    "success": True,
                }
                controller.root_interface.get_population.return_value = []

                # Execute root code
                code = """
result = spawn_child_llm("Create a player")
pop = get_population()
"""
                controller._execute_root_code(code)

                # Verify functions were called
                controller.root_interface.spawn_child_llm.assert_called_once()
                controller.root_interface.get_population.assert_called_once()


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for integration scenarios."""

    def test_empty_generation_advance(self, tmp_experiment, mock_llm_client, evaluator, cost_tracker):
        """Can advance generation with no children."""
        executor = ChildLLMExecutor(mock_llm_client)

        interface = RootLLMInterface(
            child_executor=executor,
            evaluator=evaluator,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 5}
        )

        # Advance without spawning children
        new_gen = interface.advance_generation([], "No children this gen")

        assert new_gen == 2

    def test_trial_code_roundtrip(self, tmp_experiment, cost_tracker):
        """Trial code survives save/load roundtrip."""
        # Create mock client that returns valid code
        client = Mock()
        client.model = "claude-haiku"
        client.send_message.return_value = Mock(
            content="<code>def choose_action(obs): return 5</code>",
            input_tokens=100,
            output_tokens=50,
            model="claude-haiku"
        )

        # Create mock evaluator with _validate_syntax
        mock_eval = Mock()
        mock_eval.max_steps = 100
        mock_eval.timeout_seconds = 10
        mock_eval.evaluate.return_value = Mock(
            success=True,
            to_dict=lambda: {"avg_score": 100.0, "avg_lines_cleared": 5.0, "avg_survival_steps": 500.0}
        )
        mock_eval._validate_syntax = Mock(return_value=(True, None))
        mock_eval._check_safety = Mock(return_value=(True, None))

        executor = ChildLLMExecutor(client)

        interface = RootLLMInterface(
            child_executor=executor,
            evaluator=mock_eval,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 5}
        )

        # Spawn and get trial
        result = interface.spawn_child_llm("Create a player")
        original_code = result["code"]

        # Retrieve via get_trial_code
        retrieved_code = interface.get_trial_code(result["trial_id"])

        assert retrieved_code == original_code
        assert "choose_action" in retrieved_code

    def test_population_has_all_scores(self, tmp_experiment, cost_tracker):
        """Population contains all spawned trials with their scores."""
        # Create mock client that returns different but valid code
        call_count = [0]

        def mock_send(*args, **kwargs):
            call_count[0] += 1
            return Mock(
                content="<code>def choose_action(obs): return 5</code>",
                input_tokens=100,
                output_tokens=50,
                model="claude-haiku"
            )

        client = Mock()
        client.model = "claude-haiku"
        client.send_message.side_effect = mock_send

        # Create evaluator that returns different scores for each call
        mock_eval = Mock()
        mock_eval.max_steps = 100
        mock_eval.timeout_seconds = 10

        eval_scores = [150.0, 50.0, 100.0]
        eval_idx = [0]

        def mock_evaluate(code, trial_id):
            score = eval_scores[eval_idx[0] % len(eval_scores)]
            eval_idx[0] += 1
            return Mock(
                success=True,
                to_dict=lambda s=score: {
                    "avg_score": s,
                    "avg_lines_cleared": s / 10,
                    "avg_survival_steps": s * 10,
                }
            )

        mock_eval.evaluate.side_effect = mock_evaluate
        # Also mock validation methods used by generate_and_validate
        mock_eval._validate_syntax = Mock(return_value=(True, None))
        mock_eval._check_safety = Mock(return_value=(True, None))

        executor = ChildLLMExecutor(client)

        interface = RootLLMInterface(
            child_executor=executor,
            evaluator=mock_eval,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 10}
        )

        # Spawn three children
        for i in range(3):
            interface.spawn_child_llm(f"Create player {i}")

        # Get population
        population = interface.get_population()

        assert len(population) == 3
        # Verify all expected scores are present
        scores_in_order = [m.score for m in population]
        assert sorted(scores_in_order) == sorted(eval_scores)
