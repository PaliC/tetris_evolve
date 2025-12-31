"""
Configuration system for mango_evolve.

Loads and validates YAML configuration files into typed dataclasses.
"""

import importlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigValidationError


@dataclass
class ExperimentConfig:
    """Experiment-level configuration."""

    name: str
    output_dir: str = "./experiments"
    seed: int | None = None


@dataclass
class ReasoningConfig:
    """Configuration for reasoning/thinking tokens (OpenRouter only).

    Enables extended reasoning for models that support it like Gemini Flash,
    DeepSeek, OpenAI o-series, etc.
    """

    enabled: bool = False
    effort: str | None = None  # "minimal", "low", "medium", "high", "xhigh", or None
    max_tokens: int | None = None  # Direct token limit for reasoning (alternative to effort)
    exclude: bool = False  # If True, model uses reasoning but doesn't return it


@dataclass
class LLMConfig:
    """Configuration for an LLM (root only)."""

    model: str
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float
    provider: str = "anthropic"  # "anthropic" or "openrouter"
    max_iterations: int | None = None  # Only used for root LLM
    reasoning: ReasoningConfig | None = None  # OpenRouter reasoning config


@dataclass
class ChildLLMConfig:
    """Configuration for a child LLM with optional alias and calibration settings."""

    model: str
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float
    alias: str | None = None  # Optional, defaults to model name
    provider: str = "anthropic"  # "anthropic" or "openrouter"
    calibration_calls: int = 3  # Number of test calls during calibration phase
    reasoning: ReasoningConfig | None = None  # OpenRouter reasoning config

    @property
    def effective_alias(self) -> str:
        """Return the alias to use (explicit alias or model name)."""
        return self.alias if self.alias else self.model


@dataclass
class CalibrationNotes:
    """Calibration notes file format.

    The notes field contains observations written by the root LLM during calibration.
    The metadata field stores config info for validation but is NOT exposed to the root LLM.
    """

    notes: str  # The actual notes written by root LLM
    metadata: dict[str, Any]  # Config metadata (models, costs, etc.)
    created_at: str  # ISO timestamp
    experiment_name: str | None = None


@dataclass
class EvolutionConfig:
    """Evolution process configuration."""

    max_generations: int = 10
    max_children_per_generation: int = 10


@dataclass
class BudgetConfig:
    """Budget configuration."""

    max_total_cost: float = 20.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration.

    The evaluator_fn should be a module path to an evaluator function or class,
    e.g., "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator"

    The evaluator_kwargs are passed to the evaluator function/class constructor.
    """

    evaluator_fn: str  # Module path like "module.path:ClassName" or "module.path:function_name"
    evaluator_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration container."""

    experiment: ExperimentConfig
    root_llm: LLMConfig
    child_llms: list[ChildLLMConfig]
    evaluation: EvaluationConfig
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    default_child_llm_alias: str | None = None
    calibration_notes_file: str | None = None  # Path to pre-existing notes (skips calibration)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return asdict(self)


def _validate_required_fields(data: dict[str, Any], required: list[str], section: str) -> None:
    """Validate that required fields are present."""
    for field_name in required:
        if field_name not in data:
            raise ConfigValidationError(
                f"Missing required field '{field_name}' in section '{section}'"
            )


def _validate_types(data: dict[str, Any], type_map: dict[str, type], section: str) -> None:
    """Validate field types."""
    for field_name, expected_type in type_map.items():
        if field_name in data and data[field_name] is not None:
            value = data[field_name]
            # Handle numeric types - allow int where float is expected
            if expected_type is float and isinstance(value, int):
                continue
            if not isinstance(value, expected_type):
                raise ConfigValidationError(
                    f"Field '{field_name}' in section '{section}' must be "
                    f"{expected_type.__name__}, got {type(value).__name__}"
                )


def _parse_experiment_config(data: dict[str, Any]) -> ExperimentConfig:
    """Parse experiment configuration section."""
    _validate_required_fields(data, ["name"], "experiment")
    _validate_types(
        data,
        {"name": str, "output_dir": str, "seed": int},
        "experiment",
    )
    return ExperimentConfig(
        name=data["name"],
        output_dir=data.get("output_dir", "./experiments"),
        seed=data.get("seed"),
    )


def _parse_reasoning_config(data: dict[str, Any] | None, section: str) -> ReasoningConfig | None:
    """Parse reasoning configuration section."""
    if data is None:
        return None

    _validate_types(
        data,
        {
            "enabled": bool,
            "effort": str,
            "max_tokens": int,
            "exclude": bool,
        },
        f"{section}.reasoning",
    )

    # Validate effort value if provided
    effort = data.get("effort")
    if effort is not None:
        valid_efforts = {"minimal", "low", "medium", "high", "xhigh", "none"}
        if effort not in valid_efforts:
            raise ConfigValidationError(
                f"Invalid reasoning effort '{effort}' in section '{section}'. "
                f"Must be one of: {', '.join(sorted(valid_efforts))}"
            )

    return ReasoningConfig(
        enabled=data.get("enabled", False),
        effort=effort,
        max_tokens=data.get("max_tokens"),
        exclude=data.get("exclude", False),
    )


def _parse_child_llm_config(data: dict[str, Any], index: int) -> ChildLLMConfig:
    """Parse a single child LLM configuration."""
    section = f"child_llms[{index}]"
    _validate_required_fields(
        data,
        ["model", "cost_per_million_input_tokens", "cost_per_million_output_tokens"],
        section,
    )
    _validate_types(
        data,
        {
            "model": str,
            "alias": str,
            "cost_per_million_input_tokens": float,
            "cost_per_million_output_tokens": float,
            "provider": str,
            "calibration_calls": int,
        },
        section,
    )

    # Validate provider value
    provider = data.get("provider", "anthropic")
    valid_providers = {"anthropic", "openrouter"}
    if provider not in valid_providers:
        raise ConfigValidationError(
            f"Invalid provider '{provider}' in section '{section}'. "
            f"Must be one of: {', '.join(sorted(valid_providers))}"
        )

    # Parse reasoning config
    reasoning = _parse_reasoning_config(data.get("reasoning"), section)

    return ChildLLMConfig(
        model=data["model"],
        cost_per_million_input_tokens=float(data["cost_per_million_input_tokens"]),
        cost_per_million_output_tokens=float(data["cost_per_million_output_tokens"]),
        alias=data.get("alias"),
        provider=provider,
        calibration_calls=data.get("calibration_calls", 3),
        reasoning=reasoning,
    )


def _parse_child_llm_configs(data: list[dict[str, Any]]) -> list[ChildLLMConfig]:
    """Parse list of child LLM configurations."""
    if not isinstance(data, list):
        raise ConfigValidationError("child_llms must be a list")
    if len(data) == 0:
        raise ConfigValidationError("child_llms must contain at least one configuration")

    configs = []
    seen_aliases: set[str] = set()

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigValidationError(f"child_llms[{i}] must be a dictionary")
        config = _parse_child_llm_config(item, i)
        effective_alias = config.effective_alias

        if effective_alias in seen_aliases:
            raise ConfigValidationError(
                f"Duplicate child LLM alias '{effective_alias}' in child_llms[{i}]. "
                "Each child LLM must have a unique alias (or unique model name if no alias)."
            )
        seen_aliases.add(effective_alias)
        configs.append(config)

    return configs


def _parse_llm_config(data: dict[str, Any], section: str) -> LLMConfig:
    """Parse LLM configuration section."""
    _validate_required_fields(
        data,
        ["model", "cost_per_million_input_tokens", "cost_per_million_output_tokens"],
        section,
    )
    _validate_types(
        data,
        {
            "model": str,
            "cost_per_million_input_tokens": float,
            "cost_per_million_output_tokens": float,
            "provider": str,
            "max_iterations": int,
        },
        section,
    )
    # Validate provider value
    provider = data.get("provider", "anthropic")
    valid_providers = {"anthropic", "openrouter"}
    if provider not in valid_providers:
        raise ConfigValidationError(
            f"Invalid provider '{provider}' in section '{section}'. "
            f"Must be one of: {', '.join(sorted(valid_providers))}"
        )

    # Parse reasoning config
    reasoning = _parse_reasoning_config(data.get("reasoning"), section)

    return LLMConfig(
        model=data["model"],
        cost_per_million_input_tokens=float(data["cost_per_million_input_tokens"]),
        cost_per_million_output_tokens=float(data["cost_per_million_output_tokens"]),
        provider=provider,
        max_iterations=data.get("max_iterations"),
        reasoning=reasoning,
    )


def _parse_evolution_config(data: dict[str, Any] | None) -> EvolutionConfig:
    """Parse evolution configuration section."""
    if data is None:
        return EvolutionConfig()
    _validate_types(
        data,
        {"max_generations": int, "max_children_per_generation": int},
        "evolution",
    )
    return EvolutionConfig(
        max_generations=data.get("max_generations", 10),
        max_children_per_generation=data.get("max_children_per_generation", 10),
    )


def _parse_budget_config(data: dict[str, Any] | None) -> BudgetConfig:
    """Parse budget configuration section."""
    if data is None:
        return BudgetConfig()
    _validate_types(data, {"max_total_cost": float}, "budget")
    return BudgetConfig(
        max_total_cost=float(data.get("max_total_cost", 20.0)),
    )


def _parse_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    """Parse evaluation configuration section."""
    _validate_required_fields(data, ["evaluator_fn"], "evaluation")
    _validate_types(
        data,
        {"evaluator_fn": str, "evaluator_kwargs": dict},
        "evaluation",
    )
    return EvaluationConfig(
        evaluator_fn=data["evaluator_fn"],
        evaluator_kwargs=data.get("evaluator_kwargs", {}),
    )


def load_evaluator(config: EvaluationConfig) -> Any:
    """
    Load an evaluator from its module path.

    Args:
        config: EvaluationConfig with evaluator_fn like "module.path:ClassName"

    Returns:
        Instantiated evaluator object

    Raises:
        ConfigValidationError: If the evaluator cannot be loaded
    """
    try:
        module_path, obj_name = config.evaluator_fn.rsplit(":", 1)
    except ValueError:
        raise ConfigValidationError(
            f"Invalid evaluator_fn format: {config.evaluator_fn}. "
            "Expected 'module.path:ClassName' or 'module.path:function_name'"
        ) from None

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ConfigValidationError(f"Cannot import evaluator module: {e}") from e

    try:
        evaluator_class_or_fn = getattr(module, obj_name)
    except AttributeError:
        raise ConfigValidationError(
            f"Module '{module_path}' has no attribute '{obj_name}'"
        ) from None

    try:
        return evaluator_class_or_fn(**config.evaluator_kwargs)
    except TypeError as e:
        raise ConfigValidationError(f"Cannot instantiate evaluator: {e}") from e


def load_config(path: str | Path) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Validated Config object

    Raises:
        ConfigValidationError: If configuration is invalid
        FileNotFoundError: If configuration file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> Config:
    """
    Create a Config from a dictionary.

    Args:
        data: Configuration dictionary

    Returns:
        Validated Config object

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    if not isinstance(data, dict):
        raise ConfigValidationError("Configuration must be a dictionary")

    # Validate required top-level sections
    for section in ["experiment", "root_llm", "child_llms", "evaluation"]:
        if section not in data:
            raise ConfigValidationError(f"Missing required section '{section}'")

    # Parse child LLMs
    child_llms = _parse_child_llm_configs(data["child_llms"])

    # Validate default_child_llm_alias if provided
    default_alias = data.get("default_child_llm_alias")
    if default_alias is not None:
        valid_aliases = {cfg.effective_alias for cfg in child_llms}
        if default_alias not in valid_aliases:
            raise ConfigValidationError(
                f"default_child_llm_alias '{default_alias}' not found in child_llms. "
                f"Available aliases: {', '.join(sorted(valid_aliases))}"
            )

    # Validate calibration_notes_file if provided
    calibration_notes_file = data.get("calibration_notes_file")
    if calibration_notes_file is not None:
        if not isinstance(calibration_notes_file, str):
            raise ConfigValidationError("calibration_notes_file must be a string path")

    return Config(
        experiment=_parse_experiment_config(data["experiment"]),
        root_llm=_parse_llm_config(data["root_llm"], "root_llm"),
        child_llms=child_llms,
        evaluation=_parse_evaluation_config(data["evaluation"]),
        evolution=_parse_evolution_config(data.get("evolution")),
        budget=_parse_budget_config(data.get("budget")),
        default_child_llm_alias=default_alias,
        calibration_notes_file=calibration_notes_file,
    )


def save_calibration_notes(
    path: str | Path,
    notes: str,
    child_llm_configs: list[ChildLLMConfig],
    experiment_name: str | None = None,
) -> None:
    """
    Save calibration notes to a YAML file.

    Args:
        path: Path to save the notes file
        notes: The calibration notes content (written by root LLM)
        child_llm_configs: List of child LLM configurations for metadata
        experiment_name: Optional experiment name for reference
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata from child LLM configs
    metadata = {
        "child_llms": [
            {
                "alias": cfg.alias,
                "model": cfg.model,
                "provider": cfg.provider,
                "cost_per_million_input_tokens": cfg.cost_per_million_input_tokens,
                "cost_per_million_output_tokens": cfg.cost_per_million_output_tokens,
            }
            for cfg in child_llm_configs
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if experiment_name:
        metadata["experiment_name"] = experiment_name

    data = {
        "notes": notes,
        "metadata": metadata,
    }

    with open(path, "w") as f:
        # Add header comment
        f.write("# MangoEvolve Calibration Notes\n")
        f.write("# The root LLM only sees the 'notes' section during evolution.\n")
        f.write("# Metadata is stored for validation purposes.\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_calibration_notes(path: str | Path) -> CalibrationNotes:
    """
    Load calibration notes from a YAML file.

    Args:
        path: Path to the notes file

    Returns:
        CalibrationNotes object

    Raises:
        FileNotFoundError: If the notes file doesn't exist
        ConfigValidationError: If the notes file format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration notes file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ConfigValidationError("Calibration notes file must contain a dictionary")

    if "notes" not in data:
        raise ConfigValidationError("Calibration notes file missing 'notes' field")

    if "metadata" not in data:
        raise ConfigValidationError("Calibration notes file missing 'metadata' field")

    metadata = data["metadata"]
    if not isinstance(metadata, dict):
        raise ConfigValidationError("Calibration notes 'metadata' must be a dictionary")

    return CalibrationNotes(
        notes=data["notes"],
        metadata=metadata,
        created_at=metadata.get("created_at", ""),
        experiment_name=metadata.get("experiment_name"),
    )
