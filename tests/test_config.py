"""
Tests for tetris_evolve.config module.
"""

from pathlib import Path

import pytest

from tetris_evolve import (
    ConfigValidationError,
    EvaluationConfig,
    config_from_dict,
    load_config,
    load_evaluator,
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
        assert config.evaluation.evaluator_fn == "tetris_evolve.evaluation.circle_packing:CirclePackingEvaluator"

    def test_load_with_defaults(self, sample_config_dict):
        """Load config with missing optional fields uses defaults."""
        # Remove optional sections
        del sample_config_dict["evolution"]
        del sample_config_dict["budget"]

        config = config_from_dict(sample_config_dict)

        # Check defaults are applied
        assert config.evolution.max_generations == 10
        assert config.evolution.max_children_per_generation == 10
        assert config.budget.max_total_cost == 20.0

    def test_validation_missing_required(self, sample_config_dict):
        """Raise on missing required fields."""
        del sample_config_dict["experiment"]

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "experiment" in str(exc_info.value)

    def test_validation_missing_evaluation(self, sample_config_dict):
        """Raise on missing evaluation section."""
        del sample_config_dict["evaluation"]

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "evaluation" in str(exc_info.value)

    def test_validation_missing_evaluator_fn(self, sample_config_dict):
        """Raise on missing evaluator_fn in evaluation section."""
        del sample_config_dict["evaluation"]["evaluator_fn"]

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "evaluator_fn" in str(exc_info.value)

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
        assert config.evaluation.evaluator_fn == config2.evaluation.evaluator_fn


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
        sample_config_dict["root_llm"]["cost_per_million_input_tokens"] = 0  # int

        config = config_from_dict(sample_config_dict)

        assert config.budget.max_total_cost == 10.0
        assert isinstance(config.budget.max_total_cost, float)


class TestLoadEvaluator:
    """Tests for load_evaluator function."""

    def test_load_circle_packing_evaluator(self, sample_config):
        """Load the CirclePackingEvaluator."""
        evaluator = load_evaluator(sample_config.evaluation)

        # Should be a CirclePackingEvaluator instance
        assert hasattr(evaluator, "evaluate")
        assert evaluator.n_circles == 26
        assert evaluator.target == 2.635

    def test_load_evaluator_invalid_format(self):
        """Raise on invalid evaluator_fn format."""
        config = EvaluationConfig(
            evaluator_fn="invalid_format_no_colon",
            evaluator_kwargs={},
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            load_evaluator(config)

        assert "Invalid evaluator_fn format" in str(exc_info.value)

    def test_load_evaluator_module_not_found(self):
        """Raise on non-existent module."""
        config = EvaluationConfig(
            evaluator_fn="nonexistent.module:SomeClass",
            evaluator_kwargs={},
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            load_evaluator(config)

        assert "Cannot import" in str(exc_info.value)

    def test_load_evaluator_attribute_not_found(self):
        """Raise on non-existent attribute."""
        config = EvaluationConfig(
            evaluator_fn="tetris_evolve.evaluation.circle_packing:NonExistentClass",
            evaluator_kwargs={},
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            load_evaluator(config)

        assert "no attribute" in str(exc_info.value)

    def test_load_evaluator_with_kwargs(self):
        """Load evaluator with custom kwargs."""
        config = EvaluationConfig(
            evaluator_fn="tetris_evolve.evaluation.circle_packing:CirclePackingEvaluator",
            evaluator_kwargs={"n_circles": 10, "target": 1.5, "timeout_seconds": 5},
        )

        evaluator = load_evaluator(config)

        assert evaluator.n_circles == 10
        assert evaluator.target == 1.5
        assert evaluator.timeout_seconds == 5


class TestProviderConfig:
    """Tests for LLM provider configuration."""

    def test_default_provider_is_anthropic(self, sample_config_dict):
        """Default provider should be anthropic when not specified."""
        config = config_from_dict(sample_config_dict)

        assert config.root_llm.provider == "anthropic"
        assert config.child_llm.provider == "anthropic"

    def test_explicit_anthropic_provider(self, sample_config_dict):
        """Explicit anthropic provider should be accepted."""
        sample_config_dict["root_llm"]["provider"] = "anthropic"
        sample_config_dict["child_llm"]["provider"] = "anthropic"

        config = config_from_dict(sample_config_dict)

        assert config.root_llm.provider == "anthropic"
        assert config.child_llm.provider == "anthropic"

    def test_openrouter_provider(self, sample_config_dict):
        """OpenRouter provider should be accepted."""
        sample_config_dict["root_llm"]["provider"] = "openrouter"
        sample_config_dict["child_llm"]["provider"] = "openrouter"

        config = config_from_dict(sample_config_dict)

        assert config.root_llm.provider == "openrouter"
        assert config.child_llm.provider == "openrouter"

    def test_mixed_providers(self, sample_config_dict):
        """Different providers for root and child should work."""
        sample_config_dict["root_llm"]["provider"] = "anthropic"
        sample_config_dict["child_llm"]["provider"] = "openrouter"

        config = config_from_dict(sample_config_dict)

        assert config.root_llm.provider == "anthropic"
        assert config.child_llm.provider == "openrouter"

    def test_invalid_provider_raises(self, sample_config_dict):
        """Invalid provider should raise ConfigValidationError."""
        sample_config_dict["root_llm"]["provider"] = "invalid_provider"

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "Invalid provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)
        assert "anthropic" in str(exc_info.value)
        assert "openrouter" in str(exc_info.value)

    def test_provider_type_validation(self, sample_config_dict):
        """Provider must be a string."""
        sample_config_dict["root_llm"]["provider"] = 123

        with pytest.raises(ConfigValidationError) as exc_info:
            config_from_dict(sample_config_dict)

        assert "provider" in str(exc_info.value)

    def test_provider_preserved_in_to_dict(self, sample_config_dict):
        """Provider should be preserved when serializing to dict."""
        sample_config_dict["root_llm"]["provider"] = "openrouter"

        config = config_from_dict(sample_config_dict)
        result = config.to_dict()

        assert result["root_llm"]["provider"] == "openrouter"

    def test_provider_roundtrip(self, sample_config_dict):
        """Provider should survive serialization and deserialization."""
        sample_config_dict["root_llm"]["provider"] = "openrouter"
        sample_config_dict["child_llm"]["provider"] = "anthropic"

        config = config_from_dict(sample_config_dict)
        result = config.to_dict()
        config2 = config_from_dict(result)

        assert config.root_llm.provider == config2.root_llm.provider
        assert config.child_llm.provider == config2.child_llm.provider
