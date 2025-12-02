"""
Configuration system for tetris_evolve.

Loads and validates YAML configuration files into typed dataclasses.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
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
    """Evaluation configuration."""

    n_circles: int = 26
    target_sum: float = 2.635
    timeout_seconds: int = 30


@dataclass
class Config:
    """Main configuration container."""

    experiment: ExperimentConfig
    root_llm: LLMConfig
    child_llm: LLMConfig
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

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


def _parse_evaluation_config(data: Optional[Dict[str, Any]]) -> EvaluationConfig:
    """Parse evaluation configuration section."""
    if data is None:
        return EvaluationConfig()
    _validate_types(
        data,
        {"n_circles": int, "target_sum": float, "timeout_seconds": int},
        "evaluation",
    )
    return EvaluationConfig(
        n_circles=data.get("n_circles", 26),
        target_sum=float(data.get("target_sum", 2.635)),
        timeout_seconds=data.get("timeout_seconds", 30),
    )


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
    for section in ["experiment", "root_llm", "child_llm"]:
        if section not in data:
            raise ConfigValidationError(f"Missing required section '{section}'")

    return Config(
        experiment=_parse_experiment_config(data["experiment"]),
        root_llm=_parse_llm_config(data["root_llm"], "root_llm"),
        child_llm=_parse_llm_config(data["child_llm"], "child_llm"),
        evolution=_parse_evolution_config(data.get("evolution")),
        budget=_parse_budget_config(data.get("budget")),
        evaluation=_parse_evaluation_config(data.get("evaluation")),
    )
