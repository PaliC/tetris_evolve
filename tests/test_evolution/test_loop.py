"""
Tests for the evolution loop.

These tests define the expected behavior of the main evolution
orchestration system.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

from tetris_evolve.evolution import (
    EvolutionRunner,
    EvolutionConfig,
    EvolutionLogger,
    GenerationSummary,
)
from tetris_evolve.database import ProgramDatabase, Program
from tetris_evolve.environment import TetrisConfig


class TestEvolutionConfig:
    """Tests for EvolutionConfig."""

    def test_from_dict(self):
        """Should create config from dictionary."""
        d = {
            "max_generations": 50,
            "max_child_rllms_per_generation": 20,
            "episodes_per_evaluation": 10,
            "initial_population_size": 5,
        }

        config = EvolutionConfig.from_dict(d)

        assert config.max_generations == 50
        assert config.max_child_rllms_per_generation == 20

    def test_from_yaml_file(self):
        """Should load config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
evolution:
  max_generations: 100
  initial_population_size: 10
""")
            f.flush()

            config = EvolutionConfig.from_yaml(f.name)

            assert config.max_generations == 100
            assert config.initial_population_size == 10

    def test_default_values(self):
        """Should have sensible defaults."""
        config = EvolutionConfig()

        assert config.max_generations > 0
        assert config.max_child_rllms_per_generation > 0
        assert config.episodes_per_evaluation > 0


class TestEvolutionLogger:
    """Tests for EvolutionLogger."""

    @pytest.fixture
    def logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EvolutionLogger(run_dir=Path(tmpdir))

    def test_log_event(self, logger):
        """Should log evolution events."""
        logger.log_event("test_event", {"key": "value"})

        # Should create log file
        assert (logger.run_dir / "evolution_log.jsonl").exists()

    def test_log_generation_summary(self, logger):
        """Should log generation summaries."""
        summary = GenerationSummary(
            generation=0,
            num_programs=10,
            best_score=100.0,
            avg_score=50.0,
            duration_seconds=60.0,
        )

        logger.log_generation_summary(summary)

        # Should create generation directory
        gen_dir = logger.run_dir / "generations" / "gen_000"
        assert gen_dir.exists()

    def test_save_program(self, logger):
        """Should save program with metadata."""
        program = Program(
            program_id="prog_001",
            generation=0,
            code="class Player: pass",
        )

        logger.save_program(program)

        # Check files created
        prog_dir = logger.run_dir / "generations" / "gen_000" / "programs" / "prog_001"
        assert (prog_dir / "code.py").exists()
        assert (prog_dir / "metadata.json").exists()

    def test_create_checkpoint(self, logger):
        """Should create checkpoint files."""
        state = {
            "generation": 5,
            "program_ids": ["p1", "p2"],
        }

        logger.create_checkpoint(generation=5, state=state)

        checkpoint_file = logger.run_dir / "checkpoints" / "checkpoint_gen_005.json"
        assert checkpoint_file.exists()


class TestEvolutionRunner:
    """Tests for EvolutionRunner."""

    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        client.generate = Mock(return_value='''
```python
class TetrisPlayer:
    def select_action(self, obs):
        return 5
    def reset(self):
        pass
```
''')
        return client

    @pytest.fixture
    def runner(self, mock_llm_client):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvolutionConfig(
                max_generations=3,
                initial_population_size=2,
                episodes_per_evaluation=2,
                max_child_rllms_per_generation=5,
            )
            yield EvolutionRunner(
                config=config,
                env_config=TetrisConfig(),
                llm_client=mock_llm_client,
                run_dir=Path(tmpdir),
            )

    def test_runner_creation(self, runner):
        """Should create runner with all components."""
        assert runner.config is not None
        assert runner.database is not None
        assert runner.logger is not None

    def test_initialize_population(self, runner):
        """Should create initial population."""
        runner.initialize_population()

        programs = runner.database.get_programs_by_generation(0)
        assert len(programs) == runner.config.initial_population_size

    def test_run_generation(self, runner):
        """Should run a single generation."""
        runner.initialize_population()

        summary = runner.run_generation()

        assert isinstance(summary, GenerationSummary)
        assert summary.generation == 0
        assert summary.num_programs > 0

    def test_run_evolution_respects_max_generations(self, runner):
        """Should stop at max generations."""
        result = runner.run()

        assert result["generations_completed"] <= runner.config.max_generations

    def test_run_creates_logs(self, runner):
        """Should create log files during run."""
        runner.run()

        assert (runner.logger.run_dir / "evolution_log.jsonl").exists()

    def test_run_saves_best_program(self, runner):
        """Should save the best program at the end."""
        result = runner.run()

        assert "best_program_id" in result
        assert result["best_program_id"] is not None


class TestGenerationSummary:
    """Tests for GenerationSummary."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        summary = GenerationSummary(
            generation=5,
            num_programs=20,
            best_score=500.0,
            avg_score=200.0,
            std_score=50.0,
            duration_seconds=120.0,
            rllms_spawned=15,
        )

        d = summary.to_dict()

        assert d["generation"] == 5
        assert d["best_score"] == 500.0
        assert d["duration_seconds"] == 120.0


class TestEvolutionCheckpointing:
    """Tests for checkpoint/resume functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        client.generate = Mock(return_value='''
```python
class TetrisPlayer:
    def select_action(self, obs): return 5
    def reset(self): pass
```
''')
        return client

    def test_save_and_resume(self, mock_llm_client):
        """Should save and resume evolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Run for a bit
            config = EvolutionConfig(
                max_generations=5,
                initial_population_size=2,
                episodes_per_evaluation=1,
            )
            runner1 = EvolutionRunner(
                config=config,
                env_config=TetrisConfig(),
                llm_client=mock_llm_client,
                run_dir=run_dir,
            )
            runner1.initialize_population()
            runner1.run_generation()
            runner1.save_checkpoint()

            # Resume
            runner2 = EvolutionRunner.resume(
                run_dir=run_dir,
                llm_client=mock_llm_client,
            )

            assert runner2.current_generation == runner1.current_generation
            assert len(runner2.database.get_all_programs()) > 0


class TestEvolutionIntegration:
    """Integration tests for full evolution loop."""

    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        # Return slightly different code each time
        call_count = [0]

        def generate(prompt):
            call_count[0] += 1
            return f'''
```python
class TetrisPlayer:
    def __init__(self):
        self.version = {call_count[0]}
    def select_action(self, obs):
        return 5
    def reset(self):
        pass
```
'''
        client.generate = generate
        return client

    def test_full_evolution_run(self, mock_llm_client):
        """Should complete a full evolution run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvolutionConfig(
                max_generations=2,
                initial_population_size=2,
                episodes_per_evaluation=1,
                max_child_rllms_per_generation=3,
            )

            runner = EvolutionRunner(
                config=config,
                env_config=TetrisConfig(),
                llm_client=mock_llm_client,
                run_dir=Path(tmpdir),
            )

            result = runner.run()

            assert result["success"] is True
            assert result["generations_completed"] > 0
            assert result["best_program_id"] is not None

    def test_evolution_improves_or_maintains(self, mock_llm_client):
        """Evolution should not degrade (best score should be maintained)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvolutionConfig(
                max_generations=2,
                initial_population_size=3,
                episodes_per_evaluation=2,
            )

            runner = EvolutionRunner(
                config=config,
                env_config=TetrisConfig(),
                llm_client=mock_llm_client,
                run_dir=Path(tmpdir),
            )

            result = runner.run()

            # Check that we tracked best scores
            assert "best_scores" in result
            # Best score should be maintained or improved
            if len(result["best_scores"]) >= 2:
                assert result["best_scores"][-1] >= result["best_scores"][0] * 0.9  # Allow 10% variance
