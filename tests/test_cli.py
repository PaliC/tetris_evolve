"""Tests for CLI entry point (__main__.py)."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tetris_evolve.__main__ import (
    create_parser,
    load_config,
    run_evolution,
    main,
)
from tetris_evolve.evolution.controller import EvolutionConfig, EvolutionResult


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_creation(self):
        """Parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "tetris_evolve"

    def test_parser_config_argument(self):
        """Parser accepts --config argument."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.yaml"])
        assert args.config == Path("config.yaml")

    def test_parser_config_short_argument(self):
        """Parser accepts -c short argument."""
        parser = create_parser()
        args = parser.parse_args(["-c", "config.yaml"])
        assert args.config == Path("config.yaml")

    def test_parser_resume_argument(self):
        """Parser accepts --resume argument."""
        parser = create_parser()
        args = parser.parse_args(["--resume", "/path/to/experiment"])
        assert args.resume == Path("/path/to/experiment")

    def test_parser_resume_short_argument(self):
        """Parser accepts -r short argument."""
        parser = create_parser()
        args = parser.parse_args(["-r", "/path/to/experiment"])
        assert args.resume == Path("/path/to/experiment")

    def test_parser_max_generations(self):
        """Parser accepts --max-generations argument."""
        parser = create_parser()
        args = parser.parse_args(["--max-generations", "25"])
        assert args.max_generations == 25

    def test_parser_max_cost(self):
        """Parser accepts --max-cost argument."""
        parser = create_parser()
        args = parser.parse_args(["--max-cost", "50.5"])
        assert args.max_cost == 50.5

    def test_parser_max_time(self):
        """Parser accepts --max-time argument."""
        parser = create_parser()
        args = parser.parse_args(["--max-time", "120"])
        assert args.max_time == 120

    def test_parser_output_dir(self):
        """Parser accepts --output-dir argument."""
        parser = create_parser()
        args = parser.parse_args(["--output-dir", "/tmp/experiments"])
        assert args.output_dir == Path("/tmp/experiments")

    def test_parser_output_dir_short(self):
        """Parser accepts -o short argument."""
        parser = create_parser()
        args = parser.parse_args(["-o", "/tmp/experiments"])
        assert args.output_dir == Path("/tmp/experiments")

    def test_parser_root_model(self):
        """Parser accepts --root-model argument."""
        parser = create_parser()
        args = parser.parse_args(["--root-model", "claude-sonnet-4"])
        assert args.root_model == "claude-sonnet-4"

    def test_parser_child_model(self):
        """Parser accepts --child-model argument."""
        parser = create_parser()
        args = parser.parse_args(["--child-model", "claude-haiku"])
        assert args.child_model == "claude-haiku"

    def test_parser_version_flag(self):
        """Parser accepts --version flag."""
        parser = create_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_version_short_flag(self):
        """Parser accepts -v short flag."""
        parser = create_parser()
        args = parser.parse_args(["-v"])
        assert args.version is True

    def test_parser_defaults(self):
        """Parser defaults are None."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.config is None
        assert args.resume is None
        assert args.max_generations is None
        assert args.max_cost is None
        assert args.max_time is None
        assert args.output_dir is None
        assert args.root_model is None
        assert args.child_model is None
        assert args.version is False

    def test_parser_multiple_arguments(self):
        """Parser accepts multiple arguments together."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.yaml",
            "--max-generations", "10",
            "--max-cost", "5.0",
            "--root-model", "claude-haiku",
            "-o", "/tmp/out",
        ])
        assert args.config == Path("config.yaml")
        assert args.max_generations == 10
        assert args.max_cost == 5.0
        assert args.root_model == "claude-haiku"
        assert args.output_dir == Path("/tmp/out")


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_config_defaults(self):
        """Load defaults when no config file specified."""
        parser = create_parser()
        args = parser.parse_args([])

        config = load_config(args)

        assert config.max_generations == 50
        assert config.max_cost_usd == 100.0
        assert config.max_time_minutes == 60

    def test_load_config_from_yaml(self, tmp_path):
        """Load config from YAML file."""
        yaml_content = """
max_generations: 10
max_cost_usd: 25.0
max_time_minutes: 30
initial_population_size: 3
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        parser = create_parser()
        args = parser.parse_args(["--config", str(config_path)])

        config = load_config(args)

        assert config.max_generations == 10
        assert config.max_cost_usd == 25.0
        assert config.max_time_minutes == 30
        assert config.initial_population_size == 3

    def test_load_config_cli_overrides_yaml(self, tmp_path):
        """CLI arguments override YAML config."""
        yaml_content = """
max_generations: 10
max_cost_usd: 25.0
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        parser = create_parser()
        args = parser.parse_args([
            "--config", str(config_path),
            "--max-generations", "5",
            "--max-cost", "10.0",
        ])

        config = load_config(args)

        assert config.max_generations == 5  # CLI override
        assert config.max_cost_usd == 10.0  # CLI override

    def test_load_config_cli_overrides_defaults(self):
        """CLI arguments override defaults."""
        parser = create_parser()
        args = parser.parse_args([
            "--max-generations", "15",
            "--max-time", "45",
            "--output-dir", "/custom/path",
        ])

        config = load_config(args)

        assert config.max_generations == 15
        assert config.max_time_minutes == 45
        assert config.output_dir == Path("/custom/path")

    def test_load_config_model_overrides(self):
        """Model arguments are applied."""
        parser = create_parser()
        args = parser.parse_args([
            "--root-model", "claude-opus",
            "--child-model", "claude-haiku",
        ])

        config = load_config(args)

        assert config.root_model == "claude-opus"
        assert config.child_model == "claude-haiku"


class TestRunEvolution:
    """Tests for run_evolution function."""

    @patch("tetris_evolve.__main__.EvolutionController")
    def test_run_evolution_success(self, mock_controller_cls, capsys):
        """Successful evolution run returns 0."""
        mock_result = EvolutionResult(
            experiment_id="exp_001",
            generations_completed=5,
            total_cost_usd=15.50,
            total_time_minutes=10.5,
            best_trial_id="trial_010",
            best_score=500.0,
            best_code="def choose_action(obs): return 5",
            termination_reason="generation_limit",
        )
        mock_controller = Mock()
        mock_controller.run.return_value = mock_result
        mock_controller_cls.return_value = mock_controller

        config = EvolutionConfig(max_generations=5)
        exit_code = run_evolution(config)

        assert exit_code == 0
        mock_controller.run.assert_called_once()

        captured = capsys.readouterr()
        assert "Evolution Complete" in captured.out
        assert "exp_001" in captured.out
        assert "500.0" in captured.out

    @patch("tetris_evolve.__main__.EvolutionController")
    def test_run_evolution_resume(self, mock_controller_cls, tmp_path, capsys):
        """Resume from existing experiment."""
        mock_result = EvolutionResult(
            experiment_id="exp_resumed",
            generations_completed=10,
            total_cost_usd=25.0,
            total_time_minutes=15.0,
            best_trial_id="trial_020",
            best_score=750.0,
            best_code="def choose_action(obs): return 3",
            termination_reason="cost_limit",
        )
        mock_controller = Mock()
        mock_controller.resume.return_value = mock_result
        mock_controller_cls.return_value = mock_controller

        config = EvolutionConfig(max_generations=20)
        resume_dir = tmp_path / "old_experiment"
        exit_code = run_evolution(config, resume_dir)

        assert exit_code == 0
        mock_controller.resume.assert_called_once_with(resume_dir)

        captured = capsys.readouterr()
        assert "Resuming from" in captured.out
        assert "exp_resumed" in captured.out

    def test_run_evolution_invalid_config(self, capsys):
        """Invalid config returns error code 1."""
        config = EvolutionConfig(max_generations=0)  # Invalid

        exit_code = run_evolution(config)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Configuration error" in captured.err

    @patch("tetris_evolve.__main__.EvolutionController")
    def test_run_evolution_exception(self, mock_controller_cls, capsys):
        """Exception during evolution returns error code 1."""
        mock_controller = Mock()
        mock_controller.run.side_effect = RuntimeError("Test error")
        mock_controller_cls.return_value = mock_controller

        config = EvolutionConfig(max_generations=5)
        exit_code = run_evolution(config)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error during evolution" in captured.err

    @patch("tetris_evolve.__main__.EvolutionController")
    def test_run_evolution_no_best_trial(self, mock_controller_cls, capsys):
        """Run with no successful trials shows appropriate message."""
        mock_result = EvolutionResult(
            experiment_id="exp_002",
            generations_completed=1,
            total_cost_usd=5.0,
            total_time_minutes=2.0,
            best_trial_id=None,
            best_score=None,
            best_code=None,
            termination_reason="time_limit",
        )
        mock_controller = Mock()
        mock_controller.run.return_value = mock_result
        mock_controller_cls.return_value = mock_controller

        config = EvolutionConfig(max_generations=5)
        exit_code = run_evolution(config)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No successful trials found" in captured.out

    @patch("tetris_evolve.__main__.EvolutionController")
    def test_run_evolution_truncates_long_code(self, mock_controller_cls, capsys):
        """Long code is truncated in output."""
        long_code = "x = 1\n" * 200  # Much longer than 500 chars
        mock_result = EvolutionResult(
            experiment_id="exp_003",
            generations_completed=5,
            total_cost_usd=10.0,
            total_time_minutes=5.0,
            best_trial_id="trial_005",
            best_score=300.0,
            best_code=long_code,
            termination_reason="completed",
        )
        mock_controller = Mock()
        mock_controller.run.return_value = mock_result
        mock_controller_cls.return_value = mock_controller

        config = EvolutionConfig(max_generations=5)
        exit_code = run_evolution(config)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "(truncated)" in captured.out


class TestMain:
    """Tests for main entry point."""

    def test_main_version(self, capsys):
        """--version prints version and exits."""
        exit_code = main(["--version"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "tetris_evolve" in captured.out

    @patch("tetris_evolve.__main__.run_evolution")
    def test_main_with_config(self, mock_run, tmp_path):
        """main() loads config and runs evolution."""
        yaml_content = "max_generations: 10\n"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        mock_run.return_value = 0

        exit_code = main(["--config", str(config_path)])

        assert exit_code == 0
        mock_run.assert_called_once()
        call_config = mock_run.call_args[0][0]
        assert call_config.max_generations == 10

    @patch("tetris_evolve.__main__.run_evolution")
    def test_main_with_overrides(self, mock_run):
        """main() applies CLI overrides."""
        mock_run.return_value = 0

        exit_code = main([
            "--max-generations", "15",
            "--max-cost", "20.0",
        ])

        assert exit_code == 0
        call_config = mock_run.call_args[0][0]
        assert call_config.max_generations == 15
        assert call_config.max_cost_usd == 20.0

    @patch("tetris_evolve.__main__.run_evolution")
    def test_main_with_resume(self, mock_run, tmp_path):
        """main() passes resume directory."""
        mock_run.return_value = 0
        resume_dir = tmp_path / "experiment"

        exit_code = main(["--resume", str(resume_dir)])

        assert exit_code == 0
        assert mock_run.call_args[0][1] == resume_dir

    @patch("tetris_evolve.__main__.run_evolution")
    def test_main_config_and_resume_warning(self, mock_run, tmp_path, capsys):
        """main() warns when both config and resume specified."""
        yaml_content = "max_generations: 10\n"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        mock_run.return_value = 0
        resume_dir = tmp_path / "experiment"

        exit_code = main([
            "--config", str(config_path),
            "--resume", str(resume_dir),
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @patch("tetris_evolve.__main__.run_evolution")
    def test_main_no_arguments(self, mock_run):
        """main() works with no arguments (uses defaults)."""
        mock_run.return_value = 0

        exit_code = main([])

        assert exit_code == 0
        call_config = mock_run.call_args[0][0]
        assert call_config.max_generations == 50  # default


class TestCLIIntegration:
    """Integration tests for CLI."""

    @patch("tetris_evolve.__main__.EvolutionController")
    def test_full_cli_flow(self, mock_controller_cls, tmp_path, capsys):
        """Full CLI flow from config to result."""
        # Create config file
        yaml_content = """
max_generations: 5
max_cost_usd: 10.0
max_time_minutes: 5
initial_population_size: 2
root_model: claude-haiku
child_model: claude-haiku
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        # Mock controller
        mock_result = EvolutionResult(
            experiment_id="exp_test",
            generations_completed=5,
            total_cost_usd=8.50,
            total_time_minutes=4.2,
            best_trial_id="trial_015",
            best_score=450.0,
            best_code="def choose_action(obs):\n    return obs['piece'] % 4",
            termination_reason="generation_limit (5)",
        )
        mock_controller = Mock()
        mock_controller.run.return_value = mock_result
        mock_controller_cls.return_value = mock_controller

        # Run CLI
        exit_code = main(["--config", str(config_path)])

        # Verify
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Starting evolution" in captured.out
        assert "claude-haiku" in captured.out
        assert "Evolution Complete" in captured.out
        assert "exp_test" in captured.out
        assert "450.0" in captured.out

    def test_cli_missing_config_file(self, capsys):
        """CLI handles missing config file gracefully."""
        with pytest.raises(FileNotFoundError):
            main(["--config", "/nonexistent/config.yaml"])
