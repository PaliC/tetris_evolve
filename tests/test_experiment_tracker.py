"""Tests for Experiment Tracker (Component 3)."""

import json
import pytest
from datetime import datetime
from pathlib import Path

from tetris_evolve.tracking.experiment_tracker import (
    ExperimentTracker,
    TrialData,
    GenerationStats,
)


class TestDataClasses:
    """Tests for data classes."""

    def test_trial_data_creation(self):
        """TrialData dataclass creates correctly."""
        trial = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code="def choose_action(obs): return 5",
            prompt="Create a simple player",
            reasoning="Using hard drop for simplicity",
            llm_response="<code>def choose_action(obs): return 5</code>",
            metrics={"avg_score": 100.0, "avg_lines_cleared": 5.0},
            timestamp=datetime.now(),
        )

        assert trial.trial_id == "trial_001"
        assert trial.generation == 1
        assert trial.parent_id is None
        assert "choose_action" in trial.code

    def test_trial_data_to_dict(self):
        """TrialData converts to dict correctly."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        trial = TrialData(
            trial_id="trial_002",
            generation=2,
            parent_id="trial_001",
            code="def choose_action(obs): return 3",
            prompt="Improve parent",
            reasoning="Adding rotation",
            llm_response="response",
            metrics={"avg_score": 150.0},
            timestamp=timestamp,
        )

        data = trial.to_dict()

        assert data["trial_id"] == "trial_002"
        assert data["parent_id"] == "trial_001"
        assert data["timestamp"] == "2024-01-15T14:30:00"

    def test_trial_data_from_dict(self):
        """TrialData creates from dict correctly."""
        data = {
            "trial_id": "trial_003",
            "generation": 3,
            "parent_id": "trial_002",
            "code": "code",
            "prompt": "prompt",
            "reasoning": "reasoning",
            "llm_response": "response",
            "metrics": {"avg_score": 200.0},
            "timestamp": "2024-01-15T15:00:00",
        }

        trial = TrialData.from_dict(data)

        assert trial.trial_id == "trial_003"
        assert trial.parent_id == "trial_002"

    def test_generation_stats_creation(self):
        """GenerationStats dataclass creates correctly."""
        stats = GenerationStats(
            generation=1,
            num_trials=10,
            best_score=500.0,
            avg_score=250.0,
            best_trial_id="trial_005",
            selected_parent_ids=["trial_005", "trial_003"],
        )

        assert stats.generation == 1
        assert stats.num_trials == 10
        assert stats.best_score == 500.0

    def test_generation_stats_to_dict(self):
        """GenerationStats converts to dict correctly."""
        stats = GenerationStats(
            generation=2,
            num_trials=8,
            best_score=600.0,
            avg_score=300.0,
            best_trial_id="trial_012",
            selected_parent_ids=["trial_012"],
        )

        data = stats.to_dict()

        assert data["generation"] == 2
        assert data["selected_parent_ids"] == ["trial_012"]


class TestExperimentCreation:
    """Tests for experiment creation."""

    def test_create_experiment(self, tmp_path):
        """Creates directory with config."""
        tracker = ExperimentTracker(tmp_path)

        config = {"max_generations": 10, "max_cost": 100.0}
        exp_id = tracker.create_experiment(config)

        assert exp_id.startswith("exp_")
        assert tracker.experiment_dir.exists()
        assert (tracker.experiment_dir / "config.yaml").exists()

    def test_experiment_id_format(self, tmp_path):
        """ID is timestamped correctly."""
        tracker = ExperimentTracker(tmp_path)

        exp_id = tracker.create_experiment({})

        # Format: exp_YYYY_MM_DD_HHMMSS
        assert exp_id.startswith("exp_")
        parts = exp_id.split("_")
        assert len(parts) == 5
        assert len(parts[1]) == 4  # Year
        assert len(parts[2]) == 2  # Month
        assert len(parts[3]) == 2  # Day
        assert len(parts[4]) == 6  # HHMMSS

    def test_experiment_creates_generations_dir(self, tmp_path):
        """Creates generations directory."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})

        assert (tracker.experiment_dir / "generations").exists()

    def test_experiment_saves_config(self, tmp_path):
        """Config is saved as YAML."""
        tracker = ExperimentTracker(tmp_path)

        config = {"max_generations": 50, "child_model": "claude-haiku"}
        tracker.create_experiment(config)

        import yaml
        with open(tracker.experiment_dir / "config.yaml", "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["max_generations"] == 50


class TestGenerationManagement:
    """Tests for generation management."""

    def test_start_generation(self, tmp_path):
        """Creates generation directory structure."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})

        gen_path = tracker.start_generation(1)

        assert gen_path.exists()
        assert (gen_path / "trials").exists()
        assert "gen_001" in str(gen_path)

    def test_start_multiple_generations(self, tmp_path):
        """Can start multiple generations."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})

        gen1 = tracker.start_generation(1)
        gen2 = tracker.start_generation(2)
        gen3 = tracker.start_generation(3)

        assert gen1.name == "gen_001"
        assert gen2.name == "gen_002"
        assert gen3.name == "gen_003"

    def test_complete_generation(self, tmp_path):
        """Saves all generation files."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})
        gen_path = tracker.start_generation(1)

        stats = GenerationStats(
            generation=1,
            num_trials=5,
            best_score=400.0,
            avg_score=200.0,
            best_trial_id="trial_003",
            selected_parent_ids=["trial_003", "trial_001"],
        )

        tracker.complete_generation(
            generation=1,
            selected_ids=["trial_003", "trial_001"],
            root_reasoning="Selected top performers based on score.",
            stats=stats,
        )

        assert (gen_path / "generation_stats.json").exists()
        assert (gen_path / "root_llm_reasoning.md").exists()
        assert (gen_path / "selected_parents.json").exists()


class TestTrialManagement:
    """Tests for trial management."""

    @pytest.fixture
    def tracker_with_gen(self, tmp_path):
        """Create tracker with experiment and generation."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})
        tracker.start_generation(1)
        return tracker

    def test_save_trial_creates_files(self, tracker_with_gen):
        """All trial files are created."""
        trial = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code="def choose_action(obs): return 5",
            prompt="Create a player",
            reasoning="Simple strategy",
            llm_response="<code>def choose_action(obs): return 5</code>",
            metrics={"avg_score": 100.0},
            timestamp=datetime.now(),
        )

        trial_dir = tracker_with_gen.save_trial(trial)

        assert (trial_dir / "code.py").exists()
        assert (trial_dir / "prompt.txt").exists()
        assert (trial_dir / "reasoning.md").exists()
        assert (trial_dir / "llm_response.txt").exists()
        assert (trial_dir / "metrics.json").exists()
        assert (trial_dir / "trial.json").exists()

    def test_save_trial_content(self, tracker_with_gen):
        """File contents match input."""
        trial = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code="def choose_action(obs): return 5",
            prompt="Create a player",
            reasoning="Simple strategy",
            llm_response="response text",
            metrics={"avg_score": 100.0},
            timestamp=datetime.now(),
        )

        trial_dir = tracker_with_gen.save_trial(trial)

        with open(trial_dir / "code.py", "r") as f:
            assert f.read() == "def choose_action(obs): return 5"

        with open(trial_dir / "prompt.txt", "r") as f:
            assert f.read() == "Create a player"

    def test_save_trial_with_parent(self, tracker_with_gen):
        """Parent ID is saved."""
        trial = TrialData(
            trial_id="trial_002",
            generation=1,
            parent_id="trial_001",
            code="code",
            prompt="prompt",
            reasoning="reasoning",
            llm_response="response",
            metrics=None,
            timestamp=datetime.now(),
        )

        trial_dir = tracker_with_gen.save_trial(trial)

        assert (trial_dir / "parent_id.txt").exists()
        with open(trial_dir / "parent_id.txt", "r") as f:
            assert f.read().strip() == "trial_001"

    def test_get_trial_roundtrip(self, tracker_with_gen):
        """Save/get preserves data."""
        original = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id="trial_000",
            code="def choose_action(obs): return 5",
            prompt="Create a player",
            reasoning="Simple strategy",
            llm_response="response",
            metrics={"avg_score": 150.0, "games_played": 10},
            timestamp=datetime.now(),
        )

        tracker_with_gen.save_trial(original)
        loaded = tracker_with_gen.get_trial("trial_001")

        assert loaded.trial_id == original.trial_id
        assert loaded.generation == original.generation
        assert loaded.parent_id == original.parent_id
        assert loaded.code == original.code
        assert loaded.metrics["avg_score"] == original.metrics["avg_score"]


class TestQueries:
    """Tests for query methods."""

    @pytest.fixture
    def tracker_with_trials(self, tmp_path):
        """Create tracker with multiple trials across generations."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})

        # Generation 1
        tracker.start_generation(1)
        for i in range(3):
            trial = TrialData(
                trial_id=f"trial_{i+1:03d}",
                generation=1,
                parent_id=None,
                code=f"code_{i}",
                prompt="prompt",
                reasoning="reasoning",
                llm_response="response",
                metrics={"avg_score": 100.0 * (i + 1)},
                timestamp=datetime.now(),
            )
            tracker.save_trial(trial)

        # Generation 2
        tracker.start_generation(2)
        for i in range(2):
            trial = TrialData(
                trial_id=f"trial_{i+4:03d}",
                generation=2,
                parent_id="trial_003",
                code=f"code_{i+3}",
                prompt="prompt",
                reasoning="reasoning",
                llm_response="response",
                metrics={"avg_score": 200.0 * (i + 1)},
                timestamp=datetime.now(),
            )
            tracker.save_trial(trial)

        return tracker

    def test_get_generation_trials(self, tracker_with_trials):
        """Returns all trials for generation."""
        trials = tracker_with_trials.get_generation_trials(1)

        assert len(trials) == 3
        assert all(t.generation == 1 for t in trials)

    def test_get_all_trials(self, tracker_with_trials):
        """Returns all trials across generations."""
        trials = tracker_with_trials.get_all_trials()

        assert len(trials) == 5
        gen1_trials = [t for t in trials if t.generation == 1]
        gen2_trials = [t for t in trials if t.generation == 2]
        assert len(gen1_trials) == 3
        assert len(gen2_trials) == 2

    def test_get_best_trial(self, tracker_with_trials):
        """Returns highest scoring trial."""
        best = tracker_with_trials.get_best_trial()

        assert best is not None
        assert best.trial_id == "trial_005"  # 400.0 score
        assert best.metrics["avg_score"] == 400.0


class TestExperimentStats:
    """Tests for experiment stats."""

    def test_update_experiment_stats(self, tmp_path):
        """Stats file is updated."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})

        tracker.update_experiment_stats({
            "total_trials": 15,
            "best_score": 500.0,
        })

        stats_path = tracker.experiment_dir / "experiment_stats.json"
        assert stats_path.exists()

        with open(stats_path, "r") as f:
            data = json.load(f)
        assert data["total_trials"] == 15

    def test_get_experiment_stats(self, tmp_path):
        """Stats loaded correctly."""
        tracker = ExperimentTracker(tmp_path)
        tracker.create_experiment({})

        tracker.update_experiment_stats({
            "total_trials": 20,
            "best_score": 600.0,
            "status": "completed",
        })

        stats = tracker.get_experiment_stats()

        assert stats["total_trials"] == 20
        assert stats["status"] == "completed"


class TestResumeSupport:
    """Tests for resuming experiments."""

    def test_load_experiment(self, tmp_path):
        """Can resume experiment from disk."""
        # Create original experiment
        original = ExperimentTracker(tmp_path)
        exp_id = original.create_experiment({"test": True})
        original.start_generation(1)

        for i in range(3):
            trial = TrialData(
                trial_id=f"trial_{i+1:03d}",
                generation=1,
                parent_id=None,
                code=f"code_{i}",
                prompt="prompt",
                reasoning="reasoning",
                llm_response="response",
                metrics={"avg_score": 100.0},
                timestamp=datetime.now(),
            )
            original.save_trial(trial)

        original.start_generation(2)

        # Load experiment
        loaded = ExperimentTracker.load_experiment(original.experiment_dir)

        assert loaded.experiment_id == exp_id
        assert loaded.current_generation == 2
        assert loaded._trial_counter == 3

    def test_load_experiment_preserves_trials(self, tmp_path):
        """Loading preserves access to trials."""
        original = ExperimentTracker(tmp_path)
        original.create_experiment({})
        original.start_generation(1)

        trial = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code="def choose_action(obs): return 5",
            prompt="prompt",
            reasoning="reasoning",
            llm_response="response",
            metrics={"avg_score": 150.0},
            timestamp=datetime.now(),
        )
        original.save_trial(trial)

        # Load and verify
        loaded = ExperimentTracker.load_experiment(original.experiment_dir)
        trials = loaded.get_all_trials()

        assert len(trials) == 1
        assert trials[0].trial_id == "trial_001"
        assert trials[0].metrics["avg_score"] == 150.0

    def test_load_experiment_not_found(self, tmp_path):
        """Missing experiment raises error."""
        with pytest.raises(FileNotFoundError):
            ExperimentTracker.load_experiment(tmp_path / "nonexistent")
