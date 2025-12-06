"""
Tests for experiment resume functionality.
"""

import json
from pathlib import Path

import pytest

from tetris_evolve import CostTracker, MockLLMClient, config_from_dict
from tetris_evolve.resume import (
    ResumeInfo,
    analyze_experiment,
    build_resume_prompt,
    load_cost_data,
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

    # Create cost tracking
    cost_data = {
        "total_cost": 0.05,
        "max_budget": 10.0,
        "usage_log": [
            {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost": 0.025,
                "timestamp": "2024-01-01T12:00:00",
                "llm_type": "child",
                "call_id": "test-1",
            },
            {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost": 0.025,
                "timestamp": "2024-01-01T12:00:01",
                "llm_type": "child",
                "call_id": "test-2",
            },
        ],
    }
    cost_path = exp_dir / "cost_tracking.json"
    with open(cost_path, "w") as f:
        json.dump(cost_data, f)

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
        assert info.generation_complete is False
        assert info.total_cost_spent == 0.0
        assert info.can_resume is False  # Nothing to resume

    def test_analyze_experiment_with_incomplete_gen(self, setup_experiment_with_trials):
        """Test analyzing an experiment with incomplete generation."""
        info = analyze_experiment(setup_experiment_with_trials)

        assert info.current_generation == 0
        assert info.trials_in_current_gen == 2
        assert info.max_children_per_gen == 3
        assert info.generation_complete is False
        assert info.total_cost_spent == 0.05
        assert info.can_resume is True  # Can resume

    def test_analyze_experiment_with_complete_gen(self, setup_experiment_with_complete_gen):
        """Test analyzing an experiment with a complete generation."""
        info = analyze_experiment(setup_experiment_with_complete_gen)

        assert info.current_generation == 1  # Moved to next generation
        assert info.trials_in_current_gen == 0
        assert 0 in info.completed_generation_nums
        assert info.can_resume is True  # Can still resume (gen 1 from scratch)

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

        # Should still create generation from trials
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


class TestLoadCostData:
    """Tests for load_cost_data function."""

    def test_load_cost_data_exists(self, setup_experiment_with_trials):
        """Test loading cost data when file exists."""
        cost_data = load_cost_data(setup_experiment_with_trials)

        assert cost_data["total_cost"] == 0.05
        assert len(cost_data["usage_log"]) == 2

    def test_load_cost_data_missing(self, setup_experiment_dir):
        """Test loading cost data when file doesn't exist."""
        cost_data = load_cost_data(setup_experiment_dir)
        assert cost_data == {}


class TestPrepareRedo:
    """Tests for prepare_redo function."""

    def test_prepare_redo_removes_current_gen(self, setup_experiment_with_trials):
        """Test that prepare_redo removes current generation directory."""
        gen0_dir = setup_experiment_with_trials / "generations" / "gen_0"
        assert gen0_dir.exists()

        prepare_redo(setup_experiment_with_trials)

        assert not gen0_dir.exists()

    def test_prepare_redo_empty_experiment(self, setup_experiment_dir):
        """Test prepare_redo on empty experiment (no-op)."""
        # Should not raise
        prepare_redo(setup_experiment_dir)


class TestBuildResumePrompt:
    """Tests for build_resume_prompt function."""

    def test_build_resume_prompt(self, setup_experiment_with_trials):
        """Test building resume prompt."""
        info = analyze_experiment(setup_experiment_with_trials)
        trials = load_trials_from_disk(setup_experiment_with_trials)

        prompt = build_resume_prompt(info, trials)

        assert "RESUMING EXPERIMENT" in prompt
        assert "restarted" in prompt.lower()
        assert "Spawn up to" in prompt


class TestResumeInfo:
    """Tests for ResumeInfo dataclass."""

    def test_resume_info_str(self, setup_experiment_with_trials):
        """Test ResumeInfo string representation."""
        info = analyze_experiment(setup_experiment_with_trials)
        info_str = str(info)

        assert "Current generation: 0" in info_str
        assert "Trials in current gen: 2/3" in info_str
        assert "Can resume: True" in info_str

    def test_remaining_budget(self, setup_experiment_with_trials):
        """Test remaining_budget property."""
        info = analyze_experiment(setup_experiment_with_trials)
        assert info.remaining_budget == pytest.approx(10.0 - 0.05)


class TestCostTrackerFromDict:
    """Tests for CostTracker.from_dict class method."""

    def test_cost_tracker_from_dict(self, mock_config_for_resume, setup_experiment_with_trials):
        """Test restoring CostTracker from dictionary."""
        cost_data = load_cost_data(setup_experiment_with_trials)
        tracker = CostTracker.from_dict(cost_data, mock_config_for_resume)

        assert tracker.total_cost == 0.05
        assert len(tracker.usage_log) == 2
        assert tracker.usage_log[0].input_tokens == 1000

    def test_cost_tracker_from_empty_dict(self, mock_config_for_resume):
        """Test restoring CostTracker from empty dictionary."""
        tracker = CostTracker.from_dict({}, mock_config_for_resume)

        assert tracker.total_cost == 0.0
        assert len(tracker.usage_log) == 0


class TestOrchestratorFromResume:
    """Tests for RootLLMOrchestrator.from_resume class method."""

    def test_from_resume(self, setup_experiment_with_trials):
        """Test creating orchestrator from resume."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        assert orchestrator._resume_prompt is not None
        assert "RESUMING EXPERIMENT" in orchestrator._resume_prompt
        # Resume clears current generation trials (redo mode)
        assert len(orchestrator.evolution_api.all_trials) == 0

    def test_from_resume_with_complete_gen(self, setup_experiment_with_complete_gen):
        """Test resume with a complete generation preserves it."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_complete_gen,
        )

        # Current generation is 1 (gen 0 is complete), so gen 1 trials would be cleared
        # But since gen 1 had no trials, all_trials should contain gen 0 trials
        assert len(orchestrator.evolution_api.all_trials) == 3  # Gen 0 trials preserved

    def test_from_resume_preserves_cost(self, setup_experiment_with_trials):
        """Test that resume preserves spent cost."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        assert orchestrator.cost_tracker.total_cost == pytest.approx(0.05)
        assert len(orchestrator.cost_tracker.usage_log) == 2

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

    def test_resumed_orchestrator_builds_resume_message(self, setup_experiment_with_trials):
        """Test that resumed orchestrator uses resume prompt."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        messages = orchestrator.build_initial_messages()

        assert len(messages) == 1
        assert "RESUMING EXPERIMENT" in messages[0]["content"]

    def test_resumed_orchestrator_runs_to_completion(
        self, setup_experiment_with_trials, sample_valid_packing_code
    ):
        """Test that a resumed orchestrator can run to completion."""
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=setup_experiment_with_trials,
        )

        # Set up mock LLMs
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

        # Child returns valid packing code
        mock_child.set_responses([
            f"```python\n{sample_valid_packing_code}\n```",
        ])

        # Root spawns a child, then terminates
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
        # Resume clears current gen, so we start fresh - should have 1 new trial
        assert result.total_trials == 1
