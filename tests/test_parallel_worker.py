"""
Tests for tetris_evolve.parallel_worker module.
"""

import json

from tetris_evolve.parallel_worker import _write_trial_file


class TestWriteTrialFile:
    """Tests for _write_trial_file function."""

    def test_writes_trial_json(self, temp_dir):
        """Trial JSON file is written to correct location."""
        _write_trial_file(
            trial_id="trial_0_0",
            generation=0,
            experiment_dir=str(temp_dir),
            code="def run_packing(): pass",
            metrics={"valid": True, "score": 1.5},
            prompt="Test prompt",
            response="Test response",
            reasoning="Test reasoning",
            parent_id=None,
        )

        trial_path = temp_dir / "generations" / "gen_0" / "trial_0_0.json"
        assert trial_path.exists()

        with open(trial_path) as f:
            data = json.load(f)

        assert data["trial_id"] == "trial_0_0"
        assert data["generation"] == 0
        assert data["code"] == "def run_packing(): pass"
        assert data["metrics"]["valid"] is True
        assert data["metrics"]["score"] == 1.5
        assert data["prompt"] == "Test prompt"
        assert data["response"] == "Test response"
        assert data["reasoning"] == "Test reasoning"
        assert data["parent_id"] is None
        assert "timestamp" in data

    def test_writes_with_parent_id(self, temp_dir):
        """Trial JSON includes parent_id when provided."""
        _write_trial_file(
            trial_id="trial_1_0",
            generation=1,
            experiment_dir=str(temp_dir),
            code="def run_packing(): pass",
            metrics={"valid": True},
            prompt="Improve parent",
            response="Improved code",
            reasoning="Made it better",
            parent_id="trial_0_2",
        )

        trial_path = temp_dir / "generations" / "gen_1" / "trial_1_0.json"
        with open(trial_path) as f:
            data = json.load(f)

        assert data["parent_id"] == "trial_0_2"

    def test_creates_generation_directory(self, temp_dir):
        """Generation directory is created if it doesn't exist."""
        # Ensure generations dir doesn't exist
        gen_dir = temp_dir / "generations" / "gen_5"
        assert not gen_dir.exists()

        _write_trial_file(
            trial_id="trial_5_0",
            generation=5,
            experiment_dir=str(temp_dir),
            code="pass",
            metrics={},
            prompt="test",
            response="test",
            reasoning="test",
            parent_id=None,
        )

        assert gen_dir.exists()
        assert (gen_dir / "trial_5_0.json").exists()

    def test_writes_failed_trial(self, temp_dir):
        """Failed trials are written with error info in metrics."""
        _write_trial_file(
            trial_id="trial_0_1",
            generation=0,
            experiment_dir=str(temp_dir),
            code="",
            metrics={"valid": False, "error": "No code found"},
            prompt="Test",
            response="Here's an explanation without code",
            reasoning="",
            parent_id=None,
        )

        trial_path = temp_dir / "generations" / "gen_0" / "trial_0_1.json"
        with open(trial_path) as f:
            data = json.load(f)

        assert data["code"] == ""
        assert data["metrics"]["valid"] is False
        assert "error" in data["metrics"]


class TestTrialFileTracking:
    """Tests for tracking trial completion via file count."""

    def test_count_completed_trials(self, temp_dir):
        """Can count completed trials by counting JSON files."""
        gen_dir = temp_dir / "generations" / "gen_0"

        # Write 3 trial files
        for i in range(3):
            _write_trial_file(
                trial_id=f"trial_0_{i}",
                generation=0,
                experiment_dir=str(temp_dir),
                code=f"def run_packing_{i}(): pass",
                metrics={"valid": True, "score": 1.0 + i * 0.1},
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                reasoning=f"Reasoning {i}",
                parent_id=None,
            )

        # Count JSON files (excluding summary.json)
        trial_files = list(gen_dir.glob("trial_*.json"))
        assert len(trial_files) == 3

    def test_mixed_success_failure_tracking(self, temp_dir):
        """Both successful and failed trials are tracked."""
        # Write 2 successful, 1 failed
        _write_trial_file(
            trial_id="trial_0_0",
            generation=0,
            experiment_dir=str(temp_dir),
            code="def run_packing(): return valid()",
            metrics={"valid": True, "score": 2.0},
            prompt="Test",
            response="Code here",
            reasoning="Good",
            parent_id=None,
        )
        _write_trial_file(
            trial_id="trial_0_1",
            generation=0,
            experiment_dir=str(temp_dir),
            code="",
            metrics={"valid": False, "error": "No code"},
            prompt="Test",
            response="No code sorry",
            reasoning="",
            parent_id=None,
        )
        _write_trial_file(
            trial_id="trial_0_2",
            generation=0,
            experiment_dir=str(temp_dir),
            code="def run_packing(): return another()",
            metrics={"valid": True, "score": 1.8},
            prompt="Test",
            response="More code",
            reasoning="Also good",
            parent_id=None,
        )

        gen_dir = temp_dir / "generations" / "gen_0"
        trial_files = list(gen_dir.glob("trial_*.json"))
        assert len(trial_files) == 3

        # Can also check success rate by reading files
        success_count = 0
        for f in trial_files:
            with open(f) as fp:
                data = json.load(fp)
                if data["metrics"].get("valid"):
                    success_count += 1
        assert success_count == 2
