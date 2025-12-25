"""
Integration tests for tetris_evolve.

These tests verify that multiple components work together correctly.
All tests use mock LLMs for reproducibility and speed.
"""

import json

import pytest

from tetris_evolve import (
    CostTracker,
    ExperimentLogger,
    MockLLMClient,
    config_from_dict,
)
from tetris_evolve.evaluation.circle_packing import CirclePackingEvaluator
from tetris_evolve.evolution_api import EvolutionAPI
from tetris_evolve.root_llm import RootLLMOrchestrator


@pytest.fixture
def integration_config(sample_config_dict, temp_dir):
    """Config set up for integration testing."""
    sample_config_dict["experiment"]["output_dir"] = str(temp_dir)
    sample_config_dict["root_llm"]["max_iterations"] = 10
    sample_config_dict["budget"]["max_total_cost"] = 5.0
    sample_config_dict["evolution"]["max_generations"] = 3
    sample_config_dict["evolution"]["max_children_per_generation"] = 5
    return config_from_dict(sample_config_dict)


class TestSpawnEvaluateCycle:
    """Tests for the full spawn -> evaluate cycle."""

    def test_spawn_evaluate_cycle(self, integration_config, temp_dir, sample_valid_packing_code):
        """Test that spawn_child_llm correctly spawns, evaluates, and records a trial."""
        # Set up components
        evaluator = CirclePackingEvaluator(n_circles=26)
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses(
            [f"Here's a grid-based solution:\n\n```python\n{sample_valid_packing_code}\n```"]
        )

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=3,
            max_children_per_generation=5,
        )

        # Spawn a child
        result = evolution_api.spawn_child_llm(prompt="Write a circle packing algorithm.")

        # Verify result structure
        assert "trial_id" in result
        assert "code" in result
        assert "metrics" in result
        assert "success" in result

        # Verify evaluation ran
        assert result["metrics"].get("valid") is True
        assert result["metrics"].get("score", 0) > 0
        assert result["success"] is True

        # Verify trial was recorded
        assert result["trial_id"] in evolution_api.all_trials
        assert len(evolution_api.generations[0].trials) == 1

    def test_spawn_with_invalid_code(self, integration_config, sample_invalid_packing_code):
        """Test that invalid code is correctly evaluated and recorded."""
        evaluator = CirclePackingEvaluator(n_circles=26)
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses([f"```python\n{sample_invalid_packing_code}\n```"])

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        result = evolution_api.spawn_child_llm(prompt="Test")

        # Invalid code should be marked as failed
        assert result["metrics"].get("valid") is False
        assert result["success"] is False


class TestMultiGenerationFlow:
    """Tests for multiple generation evolution flows."""

    def test_multi_generation_flow(self, integration_config, temp_dir, sample_valid_packing_code):
        """Test that multiple generations can be advanced correctly."""
        evaluator = CirclePackingEvaluator(n_circles=26)
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        # Set up multiple valid responses
        mock_child.set_responses(
            [
                f"Solution 1:\n```python\n{sample_valid_packing_code}\n```",
                f"Solution 2:\n```python\n{sample_valid_packing_code}\n```",
                f"Solution 3:\n```python\n{sample_valid_packing_code}\n```",
                f"Solution 4:\n```python\n{sample_valid_packing_code}\n```",
            ]
        )

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=3,
        )

        # Generation 0: spawn 2 children
        result1 = evolution_api.spawn_child_llm(prompt="Try approach 1")
        _result2 = evolution_api.spawn_child_llm(prompt="Try approach 2")

        assert evolution_api.current_generation == 0
        assert len(evolution_api.generations[0].trials) == 2

        # Advance to generation 1 (using internal method)
        new_gen = evolution_api._advance_generation()
        assert new_gen == 1
        assert evolution_api.current_generation == 1

        # Generation 1: spawn 2 more children
        result3 = evolution_api.spawn_child_llm(
            prompt="Mutate the best", parent_id=result1["trial_id"]
        )
        _result4 = evolution_api.spawn_child_llm(
            prompt="Try new approach",
        )

        assert len(evolution_api.generations[1].trials) == 2
        assert result3["parent_id"] == result1["trial_id"]

        # Advance to generation 2 (using internal method)
        new_gen = evolution_api._advance_generation()
        assert new_gen == 2

        # Verify generation history
        history = evolution_api._get_generation_history()
        assert len(history) == 3  # gen 0, 1, 2
        assert history[0]["num_trials"] == 2
        assert history[1]["num_trials"] == 2

    def test_get_best_trials_across_generations(
        self, integration_config, sample_valid_packing_code
    ):
        """Test that best trials are correctly ranked across generations."""
        evaluator = CirclePackingEvaluator(n_circles=26)
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses(
            [
                f"```python\n{sample_valid_packing_code}\n```",
                f"```python\n{sample_valid_packing_code}\n```",
            ]
        )

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn trials
        evolution_api.spawn_child_llm(prompt="Test 1")
        evolution_api.spawn_child_llm(prompt="Test 2")

        # Get best trials
        best = evolution_api._get_best_trials(n=5)
        assert len(best) == 2
        # Should be sorted by score descending
        assert best[0]["metrics"]["score"] >= best[1]["metrics"]["score"]


class TestBudgetStopsEvolution:
    """Tests for budget enforcement stopping evolution."""

    def test_budget_stops_evolution(self, integration_config, temp_dir):
        """Test that evolution stops when budget is exceeded."""
        # Set very low budget
        integration_config.budget.max_total_cost = 0.0001

        orchestrator = RootLLMOrchestrator(integration_config)
        cost_tracker = orchestrator.cost_tracker

        # Set up mock LLMs
        mock_root = MockLLMClient(
            model=integration_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_root.set_responses(
            [
                "Starting...\n```repl\nprint('hello')\n```",
                "Continuing...\n```repl\nprint('world')\n```",
            ]
        )
        orchestrator.root_llm = mock_root

        # Pre-spend most of budget
        cost_tracker.record_usage(
            input_tokens=5000,
            output_tokens=5000,
            llm_type="root",
            call_id="pre_spent",
        )

        result = orchestrator.run()

        assert result.terminated is True
        assert "budget" in result.reason.lower()

    def test_spawn_fails_on_budget_exceeded(self, integration_config):
        """Test that spawn_child_llm raises when budget exceeded."""
        from tetris_evolve.exceptions import BudgetExceededError

        evaluator = CirclePackingEvaluator()
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Exhaust the budget
        cost_tracker.record_usage(
            input_tokens=1000000,
            output_tokens=1000000,
            llm_type="child",
            call_id="exhaust",
        )

        with pytest.raises(BudgetExceededError):
            evolution_api.spawn_child_llm(prompt="This should fail")


class TestLoggingComplete:
    """Tests for complete logging of experiments."""

    def test_logging_complete(self, integration_config, temp_dir, sample_valid_packing_code):
        """Test that all expected log files are created."""
        orchestrator = RootLLMOrchestrator(integration_config)
        cost_tracker = orchestrator.cost_tracker

        mock_root = MockLLMClient(
            model=integration_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses([f"```python\n{sample_valid_packing_code}\n```"])
        # Root responses: spawn, selection, terminate
        mock_root.set_responses(
            [
                """Let me spawn a child.

```repl
result = spawn_child_llm("Test prompt")
print(f"Valid: {result['metrics'].get('valid')}")
```
""",
                """```selection
{"selections": [{"trial_id": "trial_0_0", "reasoning": "Only trial", "category": "performance"}], "summary": "Selected"}
```
""",
                """Good. Let me terminate.

```repl
terminate_evolution("test complete")
```
""",
            ]
        )

        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        orchestrator.evolution_api.child_llm = mock_child

        _result = orchestrator.run()

        # Check log files exist
        base_dir = orchestrator.logger.base_dir
        assert base_dir.exists()

        # Config should be saved
        config_path = base_dir / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config_data = json.load(f)
            assert "experiment" in config_data

        # Experiment summary should be saved
        experiment_path = base_dir / "experiment.json"
        assert experiment_path.exists()
        with open(experiment_path) as f:
            exp_data = json.load(f)
            assert "experiment_id" in exp_data
            assert exp_data["termination_reason"] is not None

        # Root LLM log should exist
        root_log = base_dir / "root_llm_log.jsonl"
        assert root_log.exists()
        with open(root_log) as f:
            lines = f.readlines()
            assert len(lines) > 0

        # Cost tracking should be saved
        cost_path = base_dir / "cost_tracking.json"
        assert cost_path.exists()
        with open(cost_path) as f:
            cost_data = json.load(f)
            assert "total_cost" in cost_data

        # Generations directory should exist
        gen_dir = base_dir / "generations"
        assert gen_dir.exists()

    def test_trial_files_created(self, integration_config, temp_dir, sample_valid_packing_code):
        """Test that individual trial files are created correctly."""
        evaluator = CirclePackingEvaluator()
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses([f"```python\n{sample_valid_packing_code}\n```"])

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        result = evolution_api.spawn_child_llm(prompt="Test")

        # Check trial file exists
        trial_file = logger.generations_dir / "gen_0" / f"{result['trial_id']}.json"
        assert trial_file.exists()

        with open(trial_file) as f:
            trial_data = json.load(f)
            assert trial_data["trial_id"] == result["trial_id"]
            assert "code" in trial_data
            assert "metrics" in trial_data


class TestResumeExperiment:
    """Tests for resuming experiments from saved state."""

    def test_resume_experiment(self, integration_config, temp_dir, sample_valid_packing_code):
        """Test that experiment state can be saved and loaded."""
        # First, run a partial experiment
        evaluator = CirclePackingEvaluator()
        cost_tracker = CostTracker(integration_config)
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()

        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses(
            [
                f"```python\n{sample_valid_packing_code}\n```",
                f"```python\n{sample_valid_packing_code}\n```",
            ]
        )

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=mock_child,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Run some trials
        _result1 = evolution_api.spawn_child_llm(prompt="Test 1")
        _result2 = evolution_api.spawn_child_llm(prompt="Test 2")

        # Advance generation (using internal method)
        evolution_api._advance_generation()

        # Save experiment
        logger.save_experiment(termination_reason="paused_for_resume_test")

        # Now load it back
        loaded_logger = ExperimentLogger.from_directory(logger.base_dir)

        # Verify loaded state
        exp_data = loaded_logger.load_experiment()

        assert exp_data["termination_reason"] == "paused_for_resume_test"
        assert exp_data["total_trials"] == 2
        assert len(exp_data["generations"]) == 1  # One completed generation

    def test_load_config_from_saved(self, integration_config, temp_dir):
        """Test that config can be reloaded from saved experiment."""
        logger = ExperimentLogger(integration_config)
        logger.create_experiment_directory()
        logger.save_experiment(termination_reason="config_test")

        # Load from directory
        loaded_logger = ExperimentLogger.from_directory(logger.base_dir)

        # Config should match
        assert loaded_logger.config.experiment.name == integration_config.experiment.name
        assert (
            loaded_logger.config.budget.max_total_cost == integration_config.budget.max_total_cost
        )


class TestEndToEndWithMockLLM:
    """End-to-end tests using mock LLMs."""

    def test_full_evolution_cycle(self, integration_config, temp_dir, sample_valid_packing_code):
        """Test a complete evolution cycle with spawning and automatic generation advance."""
        # Set to 2 generations for this test
        integration_config.evolution.max_generations = 2

        orchestrator = RootLLMOrchestrator(integration_config)
        cost_tracker = orchestrator.cost_tracker

        mock_root = MockLLMClient(
            model=integration_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        # Set up child responses
        mock_child.set_responses(
            [
                f"Grid approach:\n```python\n{sample_valid_packing_code}\n```",
                f"Hex approach:\n```python\n{sample_valid_packing_code}\n```",
            ]
        )

        # Set up root responses: spawn, selection, spawn, selection
        mock_root.set_responses(
            [
                """Generation 0: I'll try a grid-based approach.

```repl
result = spawn_child_llm("Try a grid-based packing approach")
print(f"Trial: valid={result['metrics'].get('valid')}, score={result['metrics'].get('score', 0):.4f}")
```
""",
                """```selection
{"selections": [{"trial_id": "trial_0_0", "reasoning": "Best", "category": "performance"}], "summary": "Selected"}
```
""",
                """Generation 1: Building on previous results with hexagonal approach.

```repl
result = spawn_child_llm("Try a hexagonal packing approach")
print(f"Trial: valid={result['metrics'].get('valid')}, score={result['metrics'].get('score', 0):.4f}")
```
""",
                """```selection
{"selections": [{"trial_id": "trial_1_0", "reasoning": "Best", "category": "performance"}], "summary": "Selected"}
```
""",
            ]
        )

        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        orchestrator.evolution_api.child_llm = mock_child

        result = orchestrator.run()

        # Verify complete flow
        assert result.terminated is True
        assert result.total_trials == 2
        assert result.successful_trials == 2
        assert result.best_score > 0
        assert result.cost_summary is not None
        assert result.cost_summary.get("total_cost", 0) > 0
        assert result.num_generations == 2
        assert "max_generations" in result.reason

    def test_evolution_with_early_termination(
        self, integration_config, temp_dir, sample_valid_packing_code
    ):
        """Test that Root LLM can terminate early."""
        orchestrator = RootLLMOrchestrator(integration_config)
        cost_tracker = orchestrator.cost_tracker

        mock_root = MockLLMClient(
            model=integration_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        mock_child.set_responses(
            [
                f"Valid:\n```python\n{sample_valid_packing_code}\n```",
            ]
        )

        mock_root.set_responses(
            [
                """Testing and terminating early.

```repl
result = spawn_child_llm("Try approach")
print(f"Result: valid={result['success']}")
terminate_evolution("Found good solution, terminating early")
```
""",
            ]
        )

        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        orchestrator.evolution_api.child_llm = mock_child

        result = orchestrator.run()

        assert result.terminated is True
        assert result.total_trials == 1
        assert result.successful_trials == 1
        assert (
            "terminating early" in result.reason.lower() or "evolution_terminated" in result.reason
        )

    def test_evolution_with_mixed_success(
        self, integration_config, temp_dir, sample_valid_packing_code, sample_invalid_packing_code
    ):
        """Test evolution with some successful and some failed trials."""
        # Set to 1 generation for this test
        integration_config.evolution.max_generations = 1

        orchestrator = RootLLMOrchestrator(integration_config)
        cost_tracker = orchestrator.cost_tracker

        mock_root = MockLLMClient(
            model=integration_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_child = MockLLMClient(
            model=integration_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        # Mix of valid and invalid responses
        mock_child.set_responses(
            [
                f"Valid:\n```python\n{sample_valid_packing_code}\n```",
                f"Invalid:\n```python\n{sample_invalid_packing_code}\n```",
            ]
        )

        # Root responses: spawn, selection (at max gen so this triggers termination)
        mock_root.set_responses(
            [
                """Testing strategies.

```repl
result1 = spawn_child_llm("Try valid approach")
result2 = spawn_child_llm("Try risky approach")
print(f"Result 1: valid={result1['success']}")
print(f"Result 2: valid={result2['success']}")
```
""",
                """```selection
{"selections": [{"trial_id": "trial_0_0", "reasoning": "Only valid trial", "category": "performance"}], "summary": "Selected valid trial"}
```
""",
            ]
        )

        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        orchestrator.evolution_api.child_llm = mock_child

        result = orchestrator.run()

        assert result.terminated is True
        assert result.total_trials == 2
        assert result.successful_trials == 1  # Only one was valid
        assert result.best_score > 0  # Should have the valid trial's score
