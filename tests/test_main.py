"""
Tests for the CLI entry point.
"""

from unittest.mock import MagicMock, patch

import pytest

from mango_evolve.main import main, parse_args


class TestParseArgs:
    """Tests for argument parsing."""

    def test_cli_help(self, capsys):
        """Test that --help works."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "mango_evolve" in captured.out
        assert "--config" in captured.out

    def test_cli_config_required(self, capsys):
        """Test that config file is required."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args([])

        # argparse exits with code 2 for missing required args
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "required" in captured.err.lower()

    def test_parse_args_config(self):
        """Test parsing config argument."""
        args = parse_args(["--config", "test.yaml"])
        assert args.config == "test.yaml"

    def test_parse_args_config_short(self):
        """Test parsing -c shorthand."""
        args = parse_args(["-c", "test.yaml"])
        assert args.config == "test.yaml"

    def test_parse_args_verbose(self):
        """Test parsing verbose flag."""
        args = parse_args(["-c", "test.yaml", "--verbose"])
        assert args.verbose is True

    def test_parse_args_verbose_short(self):
        """Test parsing -v shorthand."""
        args = parse_args(["-c", "test.yaml", "-v"])
        assert args.verbose is True


class TestMain:
    """Tests for the main function."""

    def test_cli_config_not_found(self, capsys):
        """Test error when config file doesn't exist."""
        exit_code = main(["--config", "/nonexistent/path/config.yaml"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_cli_invalid_config(self, temp_dir, capsys):
        """Test error with invalid config file."""
        # Create an invalid config file
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text("invalid: yaml\nwithout: required fields")

        exit_code = main(["--config", str(config_path)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "invalid" in captured.err.lower() or "missing" in captured.err.lower()

    def test_cli_loads_config(self, temp_dir, sample_config_dict, capsys):
        """Test that config is loaded correctly."""
        import yaml

        # Update output dir to temp
        sample_config_dict["experiment"]["output_dir"] = str(temp_dir)

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        # Mock the orchestrator to avoid actual LLM calls
        with patch("mango_evolve.main.RootLLMOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                reason="test",
                num_iterations=1,
                total_trials=0,
                successful_trials=0,
                best_score=0.0,
                cost_summary={"total_cost": 0.0},
            )
            mock_instance.logger.base_dir = temp_dir / "experiment"
            mock_orch.return_value = mock_instance

            exit_code = main(["--config", str(config_path)])

            assert exit_code == 0
            mock_orch.assert_called_once()

    def test_cli_verbose_output(self, temp_dir, sample_config_dict, capsys):
        """Test verbose mode output."""
        import yaml

        sample_config_dict["experiment"]["output_dir"] = str(temp_dir)

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        with patch("mango_evolve.main.RootLLMOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                reason="test",
                num_iterations=1,
                total_trials=0,
                successful_trials=0,
                best_score=0.0,
                cost_summary={"total_cost": 0.0},
            )
            mock_instance.logger.base_dir = temp_dir / "experiment"
            mock_orch.return_value = mock_instance

            exit_code = main(["--config", str(config_path), "--verbose"])

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Loaded configuration" in captured.out
            assert "Experiment:" in captured.out

    def test_cli_creates_experiment_dir(self, temp_dir, sample_config_dict):
        """Test that experiment directory is created."""
        import yaml

        sample_config_dict["experiment"]["output_dir"] = str(temp_dir)
        sample_config_dict["root_llm"]["max_iterations"] = 1

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        with patch("mango_evolve.main.RootLLMOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                reason="test",
                num_iterations=1,
                total_trials=0,
                successful_trials=0,
                best_score=0.0,
                cost_summary={"total_cost": 0.0},
            )
            experiment_dir = temp_dir / "test_experiment_12345"
            experiment_dir.mkdir()
            mock_instance.logger.base_dir = experiment_dir
            mock_orch.return_value = mock_instance

            exit_code = main(["--config", str(config_path)])

            assert exit_code == 0
            # The orchestrator should have been created with the config
            mock_orch.assert_called_once()

    def test_cli_runs_orchestrator(self, temp_dir, sample_config_dict):
        """Test that orchestrator.run() is called."""
        import yaml

        sample_config_dict["experiment"]["output_dir"] = str(temp_dir)

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        with patch("mango_evolve.main.RootLLMOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                reason="completed",
                num_iterations=5,
                total_trials=10,
                successful_trials=8,
                best_score=2.0,
                cost_summary={"total_cost": 1.5},
            )
            mock_instance.logger.base_dir = temp_dir / "experiment"
            mock_orch.return_value = mock_instance

            exit_code = main(["--config", str(config_path)])

            assert exit_code == 0
            mock_instance.run.assert_called_once()

    def test_cli_prints_results(self, temp_dir, sample_config_dict, capsys):
        """Test that results are printed."""
        import yaml

        sample_config_dict["experiment"]["output_dir"] = str(temp_dir)

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        with patch("mango_evolve.main.RootLLMOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                reason="max_iterations_reached",
                num_iterations=5,
                total_trials=10,
                successful_trials=8,
                best_score=2.0,
                cost_summary={"total_cost": 1.5},
            )
            mock_instance.logger.base_dir = temp_dir / "experiment"
            mock_orch.return_value = mock_instance

            exit_code = main(["--config", str(config_path)])

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "EVOLUTION COMPLETE" in captured.out
            assert "max_iterations_reached" in captured.out
            assert "Total trials: 10" in captured.out
            assert "Successful trials: 8" in captured.out
            assert "Best score: 2.0" in captured.out


class TestMainModule:
    """Test running as module."""

    def test_module_runnable(self, temp_dir, sample_config_dict):
        """Test that module can be run with python -m."""
        import yaml

        sample_config_dict["experiment"]["output_dir"] = str(temp_dir)

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        # Just verify the module structure is correct
        from mango_evolve import main as main_module

        assert hasattr(main_module, "main")
        assert callable(main_module.main)
