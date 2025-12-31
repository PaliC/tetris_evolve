"""
Cost tracking system for mango_evolve.

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
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class CostSummary:
    """Summary of cost tracking data."""

    total_cost: float
    remaining_budget: float
    total_input_tokens: int
    total_output_tokens: int
    root_cost: float
    root_calls: int
    # Per-model child costs (alias -> cost/calls)
    child_costs: dict[str, float]
    child_calls: dict[str, int]
    # Aggregate child stats
    total_child_cost: float
    total_child_calls: int
    # Cache statistics
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    cache_savings: float = 0.0  # Estimated savings from caching


class CostTracker:
    """
    Tracks token usage and enforces budget limits.

    Supports different pricing for root and child LLMs (per-model).
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
        # Root LLM pricing
        self._pricing: dict[str, dict[str, float]] = {
            "root": {
                "input": config.root_llm.cost_per_million_input_tokens / 1_000_000,
                "output": config.root_llm.cost_per_million_output_tokens / 1_000_000,
            },
        }

        # Add pricing for each child LLM by alias (format: "child:<alias>")
        self._child_aliases: list[str] = []
        for child_config in config.child_llms:
            alias = child_config.effective_alias
            self._child_aliases.append(alias)
            self._pricing[f"child:{alias}"] = {
                "input": child_config.cost_per_million_input_tokens / 1_000_000,
                "output": child_config.cost_per_million_output_tokens / 1_000_000,
            }

        self._max_budget = config.budget.max_total_cost

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_type: str,
        call_id: str | None = None,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> TokenUsage:
        """
        Record token usage and compute cost.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            llm_type: "root" or "child:<alias>" (e.g., "child:fast_model")
            call_id: Optional unique identifier for this call
            cache_creation_input_tokens: Tokens written to cache (25% markup)
            cache_read_input_tokens: Tokens read from cache (90% discount)

        Returns:
            TokenUsage record with computed cost

        Note:
            Anthropic's cache pricing:
            - Cache creation: 25% more than base input price
            - Cache read: 90% discount (10% of base price)
            - Regular input tokens: counted in input_tokens but excludes cached
        """
        if llm_type not in self._pricing:
            valid_types = ["root"] + [f"child:{alias}" for alias in self._child_aliases]
            raise ValueError(
                f"Invalid llm_type: {llm_type}. Must be one of: {', '.join(valid_types)}"
            )

        pricing = self._pricing[llm_type]

        # Calculate cost with cache pricing
        # input_tokens from API is the total, but we need to apply different rates
        # for cached vs non-cached tokens
        #
        # Anthropic's response:
        # - cache_read_input_tokens: tokens served from cache (90% discount)
        # - cache_creation_input_tokens: tokens written to cache (25% markup)
        # - input_tokens: total input tokens (regular rate for non-cached portion)
        #
        # Cost = (non_cached * base) + (cache_creation * 1.25 * base) + (cache_read * 0.1 * base)
        non_cached_tokens = input_tokens - cache_creation_input_tokens - cache_read_input_tokens
        non_cached_tokens = max(0, non_cached_tokens)  # Safety check

        cost = (
            (non_cached_tokens * pricing["input"])
            + (cache_creation_input_tokens * pricing["input"] * 1.25)
            + (cache_read_input_tokens * pricing["input"] * 0.10)
            + (output_tokens * pricing["output"])
        )

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=datetime.now(),
            llm_type=llm_type,
            call_id=call_id or str(uuid.uuid4()),
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
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

        # Per-model child costs and calls
        child_costs: dict[str, float] = {}
        child_calls: dict[str, int] = {}
        for alias in self._child_aliases:
            llm_type = f"child:{alias}"
            usage = [u for u in self.usage_log if u.llm_type == llm_type]
            child_costs[alias] = sum(u.cost for u in usage)
            child_calls[alias] = len(usage)

        total_child_cost = sum(child_costs.values())
        total_child_calls = sum(child_calls.values())

        # Calculate cache statistics
        total_cache_creation = sum(u.cache_creation_input_tokens for u in self.usage_log)
        total_cache_read = sum(u.cache_read_input_tokens for u in self.usage_log)

        # Calculate cache savings (what we saved by reading from cache)
        # Savings = cache_read_tokens * base_price * 0.90 (the 90% discount)
        # We use root pricing as a proxy since that's where most caching happens
        cache_savings = 0.0
        for u in self.usage_log:
            pricing = self._pricing.get(u.llm_type, self._pricing["root"])
            # Savings = tokens that would have been full price but were cached
            cache_savings += u.cache_read_input_tokens * pricing["input"] * 0.90

        return CostSummary(
            total_cost=self.total_cost,
            remaining_budget=self.get_remaining_budget(),
            total_input_tokens=sum(u.input_tokens for u in self.usage_log),
            total_output_tokens=sum(u.output_tokens for u in self.usage_log),
            root_cost=sum(u.cost for u in root_usage),
            root_calls=len(root_usage),
            child_costs=child_costs,
            child_calls=child_calls,
            total_child_cost=total_child_cost,
            total_child_calls=total_child_calls,
            total_cache_creation_tokens=total_cache_creation,
            total_cache_read_tokens=total_cache_read,
            cache_savings=cache_savings,
        )

    def to_dict(self) -> dict:
        """
        Serialize cost tracker state to dictionary.

        Returns:
            Dictionary representation of the cost tracker
        """
        summary = self.get_summary()
        return {
            "total_cost": self.total_cost,
            "max_budget": self._max_budget,
            "root_cost": summary.root_cost,
            "root_calls": summary.root_calls,
            "child_costs": summary.child_costs,
            "child_calls": summary.child_calls,
            "total_child_cost": summary.total_child_cost,
            "total_child_calls": summary.total_child_calls,
            "cache_savings": summary.cache_savings,
            "total_cache_creation_tokens": summary.total_cache_creation_tokens,
            "total_cache_read_tokens": summary.total_cache_read_tokens,
            "usage_log": [
                {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost": u.cost,
                    "timestamp": u.timestamp.isoformat(),
                    "llm_type": u.llm_type,
                    "call_id": u.call_id,
                    "cache_creation_input_tokens": u.cache_creation_input_tokens,
                    "cache_read_input_tokens": u.cache_read_input_tokens,
                }
                for u in self.usage_log
            ],
        }
