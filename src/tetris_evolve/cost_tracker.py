"""
Cost tracking system for tetris_evolve.

Tracks token usage and enforces budget limits.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from .config import Config
from .exceptions import BudgetExceededError


@dataclass
class TokenUsage:
    """Record of a single LLM API call's token usage."""

    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
    llm_type: str  # "root" or "child"
    call_id: str


@dataclass
class CostSummary:
    """Summary of cost tracking data."""

    total_cost: float
    remaining_budget: float
    total_input_tokens: int
    total_output_tokens: int
    root_cost: float
    child_cost: float
    root_calls: int
    child_calls: int


class CostTracker:
    """
    Tracks token usage and enforces budget limits.

    Supports different pricing for root and child LLMs.
    """

    def __init__(self, config: Config):
        """
        Initialize the cost tracker.

        Args:
            config: Configuration containing LLM pricing and budget info
        """
        self.config = config
        self.usage_log: list[TokenUsage] = []
        self.total_cost: float = 0.0

        # Cache pricing info (convert from per-million to per-token)
        self._pricing = {
            "root": {
                "input": config.root_llm.cost_per_million_input_tokens / 1_000_000,
                "output": config.root_llm.cost_per_million_output_tokens / 1_000_000,
            },
            "child": {
                "input": config.child_llm.cost_per_million_input_tokens / 1_000_000,
                "output": config.child_llm.cost_per_million_output_tokens / 1_000_000,
            },
        }
        self._max_budget = config.budget.max_total_cost

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_type: str,
        call_id: str | None = None,
    ) -> TokenUsage:
        """
        Record token usage and compute cost.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            llm_type: Either "root" or "child"
            call_id: Optional unique identifier for this call

        Returns:
            TokenUsage record with computed cost
        """
        if llm_type not in self._pricing:
            raise ValueError(f"Invalid llm_type: {llm_type}. Must be 'root' or 'child'")

        pricing = self._pricing[llm_type]
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=datetime.now(),
            llm_type=llm_type,
            call_id=call_id or str(uuid.uuid4()),
        )

        self.usage_log.append(usage)
        self.total_cost += cost

        return usage

    def check_budget(self) -> bool:
        """
        Check if we're still within budget.

        Returns:
            True if within budget, False if exceeded
        """
        return self.total_cost <= self._max_budget

    def get_remaining_budget(self) -> float:
        """
        Get remaining budget in USD.

        Returns:
            Remaining budget (can be negative if exceeded)
        """
        return self._max_budget - self.total_cost

    def raise_if_over_budget(self) -> None:
        """
        Raise BudgetExceededError if over budget.

        Raises:
            BudgetExceededError: If budget is exceeded
        """
        if not self.check_budget():
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self.total_cost:.4f} of ${self._max_budget:.2f}"
            )

    def get_summary(self) -> CostSummary:
        """
        Get a summary of all cost tracking data.

        Returns:
            CostSummary with aggregated statistics
        """
        root_usage = [u for u in self.usage_log if u.llm_type == "root"]
        child_usage = [u for u in self.usage_log if u.llm_type == "child"]

        return CostSummary(
            total_cost=self.total_cost,
            remaining_budget=self.get_remaining_budget(),
            total_input_tokens=sum(u.input_tokens for u in self.usage_log),
            total_output_tokens=sum(u.output_tokens for u in self.usage_log),
            root_cost=sum(u.cost for u in root_usage),
            child_cost=sum(u.cost for u in child_usage),
            root_calls=len(root_usage),
            child_calls=len(child_usage),
        )

    def to_dict(self) -> dict:
        """
        Serialize cost tracker state to dictionary.

        Returns:
            Dictionary representation of the cost tracker
        """
        return {
            "total_cost": self.total_cost,
            "max_budget": self._max_budget,
            "usage_log": [
                {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost": u.cost,
                    "timestamp": u.timestamp.isoformat(),
                    "llm_type": u.llm_type,
                    "call_id": u.call_id,
                }
                for u in self.usage_log
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, config: "Config") -> "CostTracker":
        """
        Restore cost tracker state from a dictionary.

        Args:
            data: Dictionary from to_dict() or cost_tracking.json
            config: Config object for pricing information

        Returns:
            CostTracker with restored state
        """
        tracker = cls(config)

        # Restore usage log
        for entry in data.get("usage_log", []):
            usage = TokenUsage(
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
                cost=entry["cost"],
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                llm_type=entry["llm_type"],
                call_id=entry["call_id"],
            )
            tracker.usage_log.append(usage)

        # Restore total cost
        tracker.total_cost = data.get("total_cost", 0.0)

        return tracker
