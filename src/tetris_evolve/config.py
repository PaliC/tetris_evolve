"""
Configuration system for tetris_evolve.

Loads and validates YAML configuration files into typed dataclasses.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import importlib
import yaml

from .exceptions import ConfigValidationError


@dataclass
class ExperimentConfig:
    """Experiment-level configuration."""

    name: str
    output_dir: str = "./experiments"
    seed: Optional[int] = None


@dataclass
class LLMConfig:
    """Configuration for an LLM (root or child)."""

    model: str
    cost_per_input_token: float
    cost_per_output_token: float
    max_iterations: Optional[int] = None  # Only used for root LLM


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
    e.g., "tetris_evolve.evaluation.circle_packing:CirclePackingEvaluator"

    The evaluator_kwargs are passed to the evaluator function/class constructor.
    """

    evaluator_fn: str  # Module path like "module.path:ClassName" or "module.path:function_name"
    evaluator_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration container."""

    experiment: ExperimentConfig
    root_llm: LLMConfig
    child_llm: LLMConfig
    evaluation: EvaluationConfig
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return asdict(self)


def _validate_required_fields(
    data: Dict[str, Any], required: list[str], section: str
) -> None:
    """Validate that required fields are present."""
    for field_name in required:
        if field_name not in data:
            raise ConfigValidationError(
                f"Missing required field '{field_name}' in section '{section}'"
            )


def _validate_types(
    data: Dict[str, Any], type_map: Dict[str, type], section: str
) -> None:
    """Validate field types."""
    for field_name, expected_type in type_map.items():
        if field_name in data and data[field_name] is not None:
            value = data[field_name]
            # Handle numeric types - allow int where float is expected
            if expected_type == float and isinstance(value, int):
                continue
            if not isinstance(value, expected_type):
                raise ConfigValidationError(
                    f"Field '{field_name}' in section '{section}' must be "
                    f"{expected_type.__name__}, got {type(value).__name__}"
                )


def _parse_experiment_config(data: Dict[str, Any]) -> ExperimentConfig:
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


def _parse_llm_config(data: Dict[str, Any], section: str) -> LLMConfig:
    """Parse LLM configuration section."""
    _validate_required_fields(
        data,
        ["model", "cost_per_input_token", "cost_per_output_token"],
        section,
    )
    _validate_types(
        data,
        {
            "model": str,
            "cost_per_input_token": float,
            "cost_per_output_token": float,
            "max_iterations": int,
        },
        section,
    )
    return LLMConfig(
        model=data["model"],
        cost_per_input_token=float(data["cost_per_input_token"]),
        cost_per_output_token=float(data["cost_per_output_token"]),
        max_iterations=data.get("max_iterations"),
    )


def _parse_evolution_config(data: Optional[Dict[str, Any]]) -> EvolutionConfig:
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


def _parse_budget_config(data: Optional[Dict[str, Any]]) -> BudgetConfig:
    """Parse budget configuration section."""
    if data is None:
        return BudgetConfig()
    _validate_types(data, {"max_total_cost": float}, "budget")
    return BudgetConfig(
        max_total_cost=float(data.get("max_total_cost", 20.0)),
    )


def _parse_evaluation_config(data: Dict[str, Any]) -> EvaluationConfig:
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
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ConfigValidationError(f"Cannot import evaluator module: {e}")

    try:
        evaluator_class_or_fn = getattr(module, obj_name)
    except AttributeError:
        raise ConfigValidationError(
            f"Module '{module_path}' has no attribute '{obj_name}'"
        )

    try:
        return evaluator_class_or_fn(**config.evaluator_kwargs)
    except TypeError as e:
        raise ConfigValidationError(f"Cannot instantiate evaluator: {e}")


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


def config_from_dict(data: Dict[str, Any]) -> Config:
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
    for section in ["experiment", "root_llm", "child_llm", "evaluation"]:
        if section not in data:
            raise ConfigValidationError(f"Missing required section '{section}'")

    return Config(
        experiment=_parse_experiment_config(data["experiment"]),
        root_llm=_parse_llm_config(data["root_llm"], "root_llm"),
        child_llm=_parse_llm_config(data["child_llm"], "child_llm"),
        evaluation=_parse_evaluation_config(data["evaluation"]),
        evolution=_parse_evolution_config(data.get("evolution")),
        budget=_parse_budget_config(data.get("budget")),
    )
