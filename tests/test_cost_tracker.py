"""
Tests for mango_evolve.cost_tracker module.
"""

import pytest

from mango_evolve import CostTracker, config_from_dict
from mango_evolve.exceptions import BudgetExceededError


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_record_usage_basic(self, sample_config):
        """Record a single usage."""
        tracker = CostTracker(sample_config)

        usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=500,
            llm_type="root",
        )

        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.llm_type == "root"
        assert usage.cost > 0
        assert len(tracker.usage_log) == 1

    def test_cost_computation_correct(self, sample_config):
        """Verify cost math is correct."""
        tracker = CostTracker(sample_config)

        # Root LLM pricing: input=3.0 per million, output=15.0 per million
        # Which is 0.000003 per token and 0.000015 per token
        usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=1000,
            llm_type="root",
        )

        expected_cost = (1000 * 0.000003) + (1000 * 0.000015)
        assert abs(usage.cost - expected_cost) < 1e-10

    def test_different_pricing_root_child(self, sample_config_dict):
        """Root and child use different pricing."""
        # Set different pricing for child
        sample_config_dict["child_llm"]["cost_per_million_input_tokens"] = 1.0
        sample_config_dict["child_llm"]["cost_per_million_output_tokens"] = 5.0
        config = config_from_dict(sample_config_dict)

        tracker = CostTracker(config)

        root_usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=1000,
            llm_type="root",
        )

        child_usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=1000,
            llm_type="child",
        )

        # Root should cost more with the default pricing
        assert root_usage.cost > child_usage.cost

    def test_budget_check_within(self, sample_config):
        """Returns True when within budget."""
        tracker = CostTracker(sample_config)

        # Small usage, should be within $10 budget
        tracker.record_usage(
            input_tokens=100,
            output_tokens=100,
            llm_type="root",
        )

        assert tracker.check_budget() is True

    def test_budget_check_exceeded(self, sample_config_dict):
        """Returns False when exceeded."""
        sample_config_dict["budget"]["max_total_cost"] = 0.001
        config = config_from_dict(sample_config_dict)

        tracker = CostTracker(config)

        # Large usage to exceed tiny budget
        tracker.record_usage(
            input_tokens=100000,
            output_tokens=100000,
            llm_type="root",
        )

        assert tracker.check_budget() is False

    def test_raise_if_over_budget(self, sample_config_dict):
        """Raises BudgetExceededError when over budget."""
        sample_config_dict["budget"]["max_total_cost"] = 0.001
        config = config_from_dict(sample_config_dict)

        tracker = CostTracker(config)

        tracker.record_usage(
            input_tokens=100000,
            output_tokens=100000,
            llm_type="root",
        )

        with pytest.raises(BudgetExceededError):
            tracker.raise_if_over_budget()

    def test_get_summary(self, sample_config):
        """Summary has all expected fields."""
        tracker = CostTracker(sample_config)

        tracker.record_usage(1000, 500, "root")
        tracker.record_usage(2000, 1000, "child")

        summary = tracker.get_summary()

        assert summary.total_cost > 0
        assert summary.remaining_budget > 0
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500
        assert summary.root_cost > 0
        assert summary.child_cost > 0
        assert summary.root_calls == 1
        assert summary.child_calls == 1

    def test_remaining_budget(self, sample_config):
        """Remaining budget decreases correctly."""
        tracker = CostTracker(sample_config)

        initial_remaining = tracker.get_remaining_budget()
        assert initial_remaining == 10.0

        tracker.record_usage(10000, 10000, "root")

        new_remaining = tracker.get_remaining_budget()
        assert new_remaining < initial_remaining

    def test_to_dict(self, sample_config):
        """Serialization to dict works."""
        tracker = CostTracker(sample_config)
        tracker.record_usage(1000, 500, "root", call_id="test-call-1")

        data = tracker.to_dict()

        assert "total_cost" in data
        assert "max_budget" in data
        assert "usage_log" in data
        assert len(data["usage_log"]) == 1
        assert data["usage_log"][0]["call_id"] == "test-call-1"

    def test_invalid_llm_type(self, sample_config):
        """Raises ValueError for invalid llm_type."""
        tracker = CostTracker(sample_config)

        with pytest.raises(ValueError) as exc_info:
            tracker.record_usage(1000, 500, "invalid_type")

        assert "invalid_type" in str(exc_info.value)
