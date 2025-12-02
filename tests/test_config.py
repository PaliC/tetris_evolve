"""
Tests for tetris_evolve.config module.
"""

import pytest
import tempfile
from pathlib import Path

from tetris_evolve import (
    Config,
    ExperimentConfig,
    LLMConfig,
    EvolutionConfig,
    BudgetConfig,
    EvaluationConfig,
    load_config,
    config_from_dict,
    ConfigValidationError,
)


class TestConfigFromDict:
    """Tests for config_from_dict function."""

    def test_load_valid_config(self, sample_config_dict):
        """Load a valid configuration dictionary."""
        config = config_from_dict(sample_config_dict)

        assert config.experiment.name == "test_experiment"
        assert config.experiment.output_dir == "./test_experiments"
        assert config.root_llm.model == "claude-sonnet-4-20250514"
        assert config.child_llm.model == "claude-sonnet-4-20250514"
        assert config.budget.max_total_cost == 10.0

    def test_load_with_defaults(self, sample_config_dict):
        """Load config with missing optional fields uses defaults."""
        # Remove optional sections
        del sample_config_dict["evolution"]
        del sample_config_dict["budget"]
        del sample_config_dict["evaluation"]

        config = config_from_dict(sample_config_dict)

        # Check defaults are applied
        assert config.evolution.max_generations == 10
        assert config.evolution.max_children_per_generation == 10
        assert config.budget.max_total_cost == 20.0
        assert config.evaluation.n_circles == 26
        assert config.evaluation.target_sum == 2.635
        assert config.evaluation.timeout_seconds == 30

    def test_validation_missing_required(self, sample_config_dict):
        """Raise on missing required fields."""
        del sample_config_dict["experiment"]

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "experiment" in str(exc_info.value)

    def test_validation_missing_required_nested(self, sample_config_dict):
        """Raise on missing required nested fields."""
        del sample_config_dict["experiment"]["name"]

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "name" in str(exc_info.value)

    def test_validation_invalid_types(self, sample_config_dict):
        """Raise on wrong types."""
        sample_config_dict["root_llm"]["max_iterations"] = "not an int"

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "max_iterations" in str(exc_info.value)

    def test_config_from_dict_not_dict(self):
        """Raise when input is not a dictionary."""
        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict("not a dict")

        assert "dictionary" in str(exc_info.value)


class TestConfigToDict:
    """Tests for Config.to_dict method."""

    def test_config_to_dict(self, sample_config):
        """Serialize config to dictionary."""
        result = sample_config.to_dict()

        assert isinstance(result, dict)
        assert result["experiment"]["name"] == "test_experiment"
        assert result["root_llm"]["model"] == "claude-sonnet-4-20250514"
        assert result["budget"]["max_total_cost"] == 10.0

    def test_roundtrip(self, sample_config_dict):
        """Config can be serialized and deserialized."""
        config = config_from_dict(sample_config_dict)
        result = config.to_dict()
        config2 = config_from_dict(result)

        assert config.experiment.name == config2.experiment.name
        assert config.root_llm.model == config2.root_llm.model
        assert config.budget.max_total_cost == config2.budget.max_total_cost


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_file(self, sample_config_dict, temp_dir):
        """Load configuration from YAML file."""
        import yaml

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = load_config(config_path)

        assert config.experiment.name == "test_experiment"
        assert config.root_llm.model == "claude-sonnet-4-20250514"

    def test_load_nonexistent_file(self):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_path_object(self, sample_config_dict, temp_dir):
        """Accept Path objects."""
        import yaml

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = load_config(Path(config_path))

        assert config.experiment.name == "test_experiment"


class TestNumericTypeHandling:
    """Tests for numeric type handling."""

    def test_int_accepted_for_float_fields(self, sample_config_dict):
        """Integer values should be accepted for float fields."""
        sample_config_dict["budget"]["max_total_cost"] = 10  # int, not float
        sample_config_dict["root_llm"]["cost_per_input_token"] = 0  # int

        config = config_from_dict(sample_config_dict)

        assert config.budget.max_total_cost == 10.0
        assert isinstance(config.budget.max_total_cost, float)
