"""Cost tracker for LLM API calls with budget enforcement."""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class LLMCall:
    """Record of a single LLM API call."""

    timestamp: datetime
    model: str
    role: str  # "root" or "child"
    generation: int
    trial_id: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "role": self.role,
            "generation": self.generation,
            "trial_id": self.trial_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMCall":
        """Create from dict."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model=data["model"],
            role=data["role"],
            generation=data["generation"],
            trial_id=data.get("trial_id"),
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost_usd=data["cost_usd"],
        )


# Cost per 1K tokens in USD
DEFAULT_COST_CONFIG = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
    # Aliases for convenience
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "claude-haiku": {"input": 0.0008, "output": 0.004},
}


class UnknownModelError(Exception):
    """Raised when cost calculation is attempted for unknown model."""

    pass


class CostTracker:
    """Tracks LLM API costs with budget enforcement."""

    def __init__(
        self, max_cost_usd: float, cost_config: Optional[dict] = None
    ) -> None:
        """Initialize cost tracker.

        Args:
            max_cost_usd: Maximum allowed total cost in USD.
            cost_config: Optional cost config per model. Uses DEFAULT_COST_CONFIG if not provided.
        """
        self.max_cost_usd = max_cost_usd
        self.cost_config = cost_config or DEFAULT_COST_CONFIG
        self.calls: list[LLMCall] = []

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for given model and token counts.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.

        Raises:
            UnknownModelError: If model is not in cost config.
        """
        if model not in self.cost_config:
            raise UnknownModelError(
                f"Unknown model '{model}'. Known models: {list(self.cost_config.keys())}"
            )

        config = self.cost_config[model]
        input_cost = (input_tokens / 1000) * config["input"]
        output_cost = (output_tokens / 1000) * config["output"]
        return input_cost + output_cost

    def record_call(
        self,
        model: str,
        role: str,
        generation: int,
        input_tokens: int,
        output_tokens: int,
        trial_id: Optional[str] = None,
    ) -> LLMCall:
        """Record an LLM API call.

        Args:
            model: Model name used.
            role: Either "root" or "child".
            generation: Current generation number.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            trial_id: Optional trial ID if this was a child call.

        Returns:
            The recorded LLMCall.
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        call = LLMCall(
            timestamp=datetime.now(),
            model=model,
            role=role,
            generation=generation,
            trial_id=trial_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.calls.append(call)
        return call

    def get_total_cost(self) -> float:
        """Get total cost of all recorded calls."""
        return sum(call.cost_usd for call in self.calls)

    def get_remaining_budget(self) -> float:
        """Get remaining budget in USD."""
        return self.max_cost_usd - self.get_total_cost()

    def would_exceed_budget(self, estimated_cost: float) -> bool:
        """Check if adding estimated_cost would exceed budget."""
        return self.get_total_cost() + estimated_cost > self.max_cost_usd

    def get_summary(self) -> dict:
        """Get cost summary broken down by role.

        Returns:
            Dict with 'root_llm' and 'child_llm' breakdowns.
        """
        summary = {
            "root_llm": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            "child_llm": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        }

        for call in self.calls:
            key = "root_llm" if call.role == "root" else "child_llm"
            summary[key]["calls"] += 1
            summary[key]["input_tokens"] += call.input_tokens
            summary[key]["output_tokens"] += call.output_tokens
            summary[key]["cost_usd"] += call.cost_usd

        return summary

    def get_per_generation_costs(self) -> list[dict]:
        """Get cost breakdown per generation.

        Returns:
            List of dicts with generation, cost_usd, and trials count.
        """
        gen_costs: dict[int, dict] = {}

        for call in self.calls:
            gen = call.generation
            if gen not in gen_costs:
                gen_costs[gen] = {"generation": gen, "cost_usd": 0.0, "trials": set()}
            gen_costs[gen]["cost_usd"] += call.cost_usd
            if call.trial_id:
                gen_costs[gen]["trials"].add(call.trial_id)

        # Convert trials set to count
        result = []
        for gen in sorted(gen_costs.keys()):
            entry = gen_costs[gen]
            result.append({
                "generation": entry["generation"],
                "cost_usd": entry["cost_usd"],
                "trials": len(entry["trials"]),
            })

        return result

    def save(self, path: Path) -> None:
        """Save cost tracker state to JSON file.

        Args:
            path: Path to save JSON file.
        """
        data = {
            "max_cost_usd": self.max_cost_usd,
            "total_cost_usd": self.get_total_cost(),
            "calls": [call.to_dict() for call in self.calls],
            "summary": self.get_summary(),
            "per_generation": self.get_per_generation_costs(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls, path: Path, cost_config: Optional[dict] = None
    ) -> "CostTracker":
        """Load cost tracker from JSON file.

        Args:
            path: Path to JSON file.
            cost_config: Optional cost config to use.

        Returns:
            CostTracker instance with loaded state.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(path, "r") as f:
            data = json.load(f)

        tracker = cls(data["max_cost_usd"], cost_config)
        tracker.calls = [LLMCall.from_dict(c) for c in data["calls"]]

        return tracker
