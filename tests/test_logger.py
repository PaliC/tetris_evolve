"""
Tests for mango_evolve.logger module.
"""

import json

from mango_evolve import ExperimentLogger


class TestExperimentLogger:
    """Tests for ExperimentLogger class."""

    def test_create_experiment_directory(self, sample_config, temp_dir):
        """Directory structure created correctly."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")

        result_dir = logger.create_experiment_directory()

        assert result_dir.exists()
        assert (result_dir / "generations").exists()
        assert (result_dir / "config.json").exists()

    def test_log_trial(self, sample_config, temp_dir):
        """Trial JSON written to correct location."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        trial_path = logger.log_trial(
            trial_id="trial_0_0",
            generation=0,
            code="def run_packing(): pass",
            metrics={"valid": True, "score": 1.5},
            prompt="Try grid packing",
            response="Here is the code...",
            reasoning="I used a grid approach",
        )

        assert trial_path.exists()
        with open(trial_path) as f:
            data = json.load(f)

        assert data["trial_id"] == "trial_0_0"
        assert data["generation"] == 0
        assert data["metrics"]["valid"] is True

    def test_log_generation(self, sample_config, temp_dir):
        """Generation summary written."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        trials = [
            {"trial_id": "trial_0_0", "metrics": {"valid": True, "score": 1.5}},
            {"trial_id": "trial_0_1", "metrics": {"valid": False, "score": 0}},
        ]

        summary_path = logger.log_generation(
            generation=0,
            trials=trials,
            selected_trial_ids=["trial_0_0"],
            selection_reasoning="Best score",
            best_trial_id="trial_0_0",
            best_score=1.5,
        )

        assert summary_path.exists()
        with open(summary_path) as f:
            data = json.load(f)

        assert data["generation_num"] == 0
        assert data["num_trials"] == 2
        assert data["num_successful_trials"] == 1
        assert data["best_trial_id"] == "trial_0_0"

    def test_log_root_turn(self, sample_config, temp_dir):
        """Root LLM turn appended to JSONL."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        logger.log_root_turn(
            turn_number=0,
            role="system",
            content="You are an orchestrator...",
        )
        logger.log_root_turn(
            turn_number=1,
            role="assistant",
            content="I'll try grid packing",
            code_executed="spawn_child_llm('grid')",
            execution_result="{'trial_id': 'trial_0_0'}",
        )

        log_path = logger.base_dir / "root_llm_log.jsonl"
        assert log_path.exists()

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        turn0 = json.loads(lines[0])
        turn1 = json.loads(lines[1])

        assert turn0["role"] == "system"
        assert turn1["code_executed"] == "spawn_child_llm('grid')"

    def test_save_experiment(self, sample_config, temp_dir):
        """Full experiment state saved."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        experiment_path = logger.save_experiment(termination_reason="Completed")

        assert experiment_path.exists()
        with open(experiment_path) as f:
            data = json.load(f)

        assert data["termination_reason"] == "Completed"
        assert data["end_time"] is not None

    def test_load_experiment(self, sample_config, temp_dir):
        """Can reload saved experiment."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        # Log some data
        logger.log_trial(
            trial_id="trial_0_0",
            generation=0,
            code="pass",
            metrics={"valid": True},
            prompt="test",
            response="test",
            reasoning="test",
        )
        logger.save_experiment(termination_reason="Test")

        # Create new logger and load
        logger2 = ExperimentLogger(sample_config, run_id="test_run")
        logger2.base_dir = logger.base_dir
        logger2.generations_dir = logger.generations_dir

        data = logger2.load_experiment()

        assert data["termination_reason"] == "Test"
        assert data["total_trials"] == 1

    def test_log_cost_tracking(self, sample_config, temp_dir):
        """Cost data saved correctly."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        cost_data = {
            "total_cost": 1.50,
            "max_budget": 10.0,
            "usage_log": [],
        }

        cost_path = logger.log_cost_tracking(cost_data)

        assert cost_path.exists()
        with open(cost_path) as f:
            data = json.load(f)

        assert data["total_cost"] == 1.50

    def test_best_trial_tracking(self, sample_config, temp_dir):
        """Best trial is tracked across generations."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()

        # First generation
        logger.log_generation(
            generation=0,
            trials=[],
            selected_trial_ids=[],
            selection_reasoning="",
            best_trial_id="trial_0_0",
            best_score=1.5,
        )

        # Second generation with better result
        logger.log_generation(
            generation=1,
            trials=[],
            selected_trial_ids=[],
            selection_reasoning="",
            best_trial_id="trial_1_0",
            best_score=2.0,
        )

        logger.save_experiment()

        data = logger.load_experiment()
        assert data["best_trial"]["trial_id"] == "trial_1_0"
        assert data["best_trial"]["score"] == 2.0

    def test_from_directory(self, sample_config, temp_dir):
        """Create logger from existing directory."""
        sample_config.experiment.output_dir = str(temp_dir)
        logger = ExperimentLogger(sample_config, run_id="test_run")
        logger.create_experiment_directory()
        logger.save_experiment(termination_reason="Original")

        # Load from directory
        loaded = ExperimentLogger.from_directory(logger.base_dir)

        assert loaded.config.experiment.name == sample_config.experiment.name
        data = loaded.load_experiment()
        assert data["termination_reason"] == "Original"
