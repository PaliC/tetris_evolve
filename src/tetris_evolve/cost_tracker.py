"""
Cost tracking system for tetris_evolve.

Tracks token usage and enforces budget limits.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

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
        self.usage_log: List[TokenUsage] = []
        self.total_cost: float = 0.0

        # Cache pricing info
        self._pricing = {
            "root": {
                "input": config.root_llm.cost_per_input_token,
                "output": config.root_llm.cost_per_output_token,
            },
            "child": {
                "input": config.child_llm.cost_per_input_token,
                "output": config.child_llm.cost_per_output_token,
            },
        }
        self._max_budget = config.budget.max_total_cost

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_type: str,
        call_id: Optional[str] = None,
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

    def to_dict(self) -> Dict:
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
