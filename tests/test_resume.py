"""
Tests for experiment resume functionality.
"""

import json

import pytest

from .helpers import MockLLMClient

from tetris_evolve import config_from_dict
from tetris_evolve.resume import (
    analyze_experiment,
    load_generation_summaries,
    load_trials_from_disk,
    prepare_redo,
)
from tetris_evolve.root_llm import RootLLMOrchestrator


@pytest.fixture
def mock_config_for_resume(sample_config_dict, temp_dir):
    """Config for resume tests with temp directory."""
    sample_config_dict["experiment"]["output_dir"] = str(temp_dir)
    sample_config_dict["evolution"]["max_generations"] = 3
    sample_config_dict["evolution"]["max_children_per_generation"] = 3
    sample_config_dict["budget"]["max_total_cost"] = 10.0
    return config_from_dict(sample_config_dict)


@pytest.fixture
def setup_experiment_dir(temp_dir, mock_config_for_resume):
    """Create a basic experiment directory structure."""
    exp_dir = temp_dir / "test_experiment_20240101_120000"
    exp_dir.mkdir(parents=True)

    # Create config.json
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config_for_resume.to_dict(), f)

    # Create generations directory
    gen_dir = exp_dir / "generations"
    gen_dir.mkdir()

    return exp_dir


@pytest.fixture
def setup_experiment_with_trials(setup_experiment_dir):
    """Create experiment with some completed trials."""
    exp_dir = setup_experiment_dir
    gen0_dir = exp_dir / "generations" / "gen_0"
    gen0_dir.mkdir(parents=True)

    # Create two trial files
    for i in range(2):
        trial_data = {
            "trial_id": f"trial_0_{i}",
            "generation": 0,
            "parent_id": None,
            "code": f"# Trial {i} code\ndef run_packing(): pass",
            "metrics": {"valid": True, "sum_radii": 1.5 + i * 0.1},
            "prompt": f"Test prompt {i}",
            "response": f"Test response {i}",
            "reasoning": f"Test reasoning {i}",
            "timestamp": "2024-01-01T12:00:00",
        }
        trial_path = gen0_dir / f"trial_0_{i}.json"
        with open(trial_path, "w") as f:
            json.dump(trial_data, f)

    return exp_dir


@pytest.fixture
def setup_experiment_with_complete_gen(setup_experiment_with_trials):
    """Create experiment with a complete generation (has summary.json)."""
    exp_dir = setup_experiment_with_trials
    gen0_dir = exp_dir / "generations" / "gen_0"

    # Add third trial to reach max
    trial_data = {
        "trial_id": "trial_0_2",
        "generation": 0,
        "parent_id": None,
        "code": "# Trial 2 code\ndef run_packing(): pass",
        "metrics": {"valid": True, "sum_radii": 1.8},
        "prompt": "Test prompt 2",
        "response": "Test response 2",
        "reasoning": "Test reasoning 2",
        "timestamp": "2024-01-01T12:00:02",
    }
    trial_path = gen0_dir / "trial_0_2.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f)

    # Create summary.json
    summary_data = {
        "generation_num": 0,
        "num_trials": 3,
        "num_successful_trials": 3,
        "best_trial_id": "trial_0_2",
        "best_sum_radii": 1.8,
        "selected_trial_ids": ["trial_0_2", "trial_0_1"],
        "selection_reasoning": "Selected top performers",
        "trial_selections": [
            {"trial_id": "trial_0_2", "reasoning": "Best score", "category": "performance"},
            {"trial_id": "trial_0_1", "reasoning": "Second best", "category": "performance"},
        ],
        "timestamp": "2024-01-01T12:00:03",
    }
    summary_path = gen0_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)

    return exp_dir


class TestAnalyzeExperiment:
    """Tests for analyze_experiment function."""

    def test_analyze_empty_experiment(self, setup_experiment_dir):
        """Test analyzing an experiment with no trials."""
        info = analyze_experiment(setup_experiment_dir)

        assert info.current_generation == 0
        assert info.trials_in_current_gen == 0
        assert info.can_resume is False

    def test_analyze_experiment_with_incomplete_gen(self, setup_experiment_with_trials):
        """Test analyzing an experiment with incomplete generation."""
        info = analyze_experiment(setup_experiment_with_trials)

        assert info.current_generation == 0
        assert info.trials_in_current_gen == 2
        assert info.max_children_per_gen == 3
        assert info.can_resume is True

    def test_analyze_experiment_with_complete_gen(self, setup_experiment_with_complete_gen):
        """Test analyzing an experiment with a complete generation."""
        info = analyze_experiment(setup_experiment_with_complete_gen)

        assert info.current_generation == 1
        assert info.trials_in_current_gen == 0
        assert info.can_resume is True

    def test_analyze_nonexistent_experiment(self, temp_dir):
        """Test analyzing a non-existent experiment directory."""
        with pytest.raises(FileNotFoundError):
            analyze_experiment(temp_dir / "nonexistent")

    def test_analyze_experiment_no_config(self, temp_dir):
        """Test analyzing an experiment without config.json."""
        exp_dir = temp_dir / "no_config_exp"
        exp_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            analyze_experiment(exp_dir)


class TestLoadTrialsFromDisk:
    """Tests for load_trials_from_disk function."""

    def test_load_trials_empty(self, setup_experiment_dir):
        """Test loading trials from empty experiment."""
        trials = load_trials_from_disk(setup_experiment_dir)
        assert len(trials) == 0

    def test_load_trials_with_data(self, setup_experiment_with_trials):
        """Test loading trials from experiment with data."""
        trials = load_trials_from_disk(setup_experiment_with_trials)

        assert len(trials) == 2
        assert "trial_0_0" in trials
        assert "trial_0_1" in trials
        assert trials["trial_0_0"].generation == 0
        assert trials["trial_0_0"].success is True
        assert trials["trial_0_1"].metrics["sum_radii"] == pytest.approx(1.6)


class TestLoadGenerationSummaries:
    """Tests for load_generation_summaries function."""

    def test_load_summaries_no_summary(self, setup_experiment_with_trials):
        """Test loading generation summaries when none exist."""
        generations = load_generation_summaries(setup_experiment_with_trials)

        assert len(generations) == 1
        assert generations[0].generation_num == 0
        assert len(generations[0].trials) == 2

    def test_load_summaries_with_summary(self, setup_experiment_with_complete_gen):
        """Test loading generation summaries with summary.json."""
        generations = load_generation_summaries(setup_experiment_with_complete_gen)

        assert len(generations) == 1
        assert generations[0].generation_num == 0
        assert len(generations[0].selected_trial_ids) == 2
        assert generations[0].best_trial_id == "trial_0_2"
        assert generations[0].best_score == 1.8


class TestPrepareRedo:
    """Tests for prepare_redo function."""

    def test_prepare_redo_removes_current_gen(self, setup_experiment_with_trials):
        """Test that prepare_redo removes current generation directory."""
        gen0_dir = setup_experiment_with_trials / "generations" / "gen_0"
        assert gen0_dir.exists()

        prepare_redo(setup_experiment_with_trials, current_generation=0)

        assert not gen0_dir.exists()

    def test_prepare_redo_nonexistent_gen(self, setup_experiment_dir):
        """Test prepare_redo on nonexistent generation (no-op)."""
        prepare_redo(setup_experiment_dir, current_generation=0)


class TestResumeInfo:
    """Tests for ResumeInfo dataclass."""

    def test_can_resume_with_trials(self, setup_experiment_with_trials):
        """Test can_resume when trials exist."""
        info = analyze_experiment(setup_experiment_with_trials)
        assert info.can_resume is True

    def test_can_resume_empty(self, setup_experiment_dir):
        """Test can_resume when no trials exist."""
        info = analyze_experiment(setup_experiment_dir)
        assert info.can_resume is False


class TestOrchestratorFromResume:
    """Tests for RootLLMOrchestrator.from_resume class method."""

    def test_from_resume(self, setup_experiment_with_trials):
        """Test creating orchestrator from resume."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        # After redo, current gen is cleared so no trials
        assert len(orchestrator.evolution_api.all_trials) == 0
        # Generation should be 0 since we're redoing it
        assert orchestrator.evolution_api.current_generation == 0

    def test_from_resume_with_complete_gen(self, setup_experiment_with_complete_gen):
        """Test resume with a complete generation preserves it."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_complete_gen,
        )

        assert len(orchestrator.evolution_api.all_trials) == 3

    def test_from_resume_starts_fresh_cost(self, setup_experiment_with_trials):
        """Test that resume starts with fresh cost tracker."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        # Cost is not preserved on resume - starts fresh
        assert orchestrator.cost_tracker.total_cost == 0.0
        assert len(orchestrator.cost_tracker.usage_log) == 0

    def test_from_resume_nonexistent_dir(self, temp_dir):
        """Test from_resume with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            RootLLMOrchestrator.from_resume(
                experiment_dir=temp_dir / "nonexistent",
            )

    def test_from_resume_empty_experiment(self, setup_experiment_dir):
        """Test from_resume with empty experiment raises error."""
        with pytest.raises(ValueError, match="no generations have been started"):
            RootLLMOrchestrator.from_resume(
                experiment_dir=setup_experiment_dir,
            )


class TestResumedOrchestrationRun:
    """Tests for running a resumed orchestration."""

    def test_resumed_orchestrator_builds_initial_message(self, setup_experiment_with_trials):
        """Test that resumed orchestrator builds correct initial message."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        messages = orchestrator.build_initial_messages()

        assert len(messages) == 1
        # Should use the current generation number (0 after redo)
        assert "Begin generation 0" in messages[0]["content"]

    def test_resumed_orchestrator_runs_to_completion(
        self, setup_experiment_with_trials, sample_valid_packing_code
    ):
        """Test that a resumed orchestrator can run to completion."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        cost_tracker = orchestrator.cost_tracker
        mock_root = MockLLMClient(
            model=orchestrator.config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_child = MockLLMClient(
            model=orchestrator.config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        mock_child.set_responses([
            f"```python\n{sample_valid_packing_code}\n```",
        ])

        mock_root.set_responses([
            '''Restarting the generation.

```repl
result = spawn_child_llm("Restart generation 0")
print(f"Result: {result['success']}")
```
''',
            '''```selection
{"selections": [{"trial_id": "trial_0_0", "reasoning": "Best", "category": "performance"}], "summary": "Selected best"}
```
''',
            '''Done. Terminating.

```repl
terminate_evolution("Completed via resume")
```
''',
        ])

        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        orchestrator.evolution_api.child_llm = mock_child

        result = orchestrator.run()

        assert result.terminated is True
        assert result.total_trials == 1
