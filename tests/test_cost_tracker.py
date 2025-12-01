"""Tests for Cost Tracker (Component 2)."""

import json
import pytest
from datetime import datetime
from pathlib import Path

from tetris_evolve.tracking.cost_tracker import (
    CostTracker,
    LLMCall,
    UnknownModelError,
    DEFAULT_COST_CONFIG,
)


class TestLLMCall:
    """Tests for LLMCall dataclass."""

    def test_llm_call_creation(self):
        """LLMCall dataclass creates correctly."""
        call = LLMCall(
            timestamp=datetime.now(),
            model="claude-sonnet-4",
            role="root",
            generation=1,
            trial_id=None,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0105,
        )

        assert call.model == "claude-sonnet-4"
        assert call.role == "root"
        assert call.input_tokens == 1000
        assert call.output_tokens == 500

    def test_llm_call_to_dict(self):
        """LLMCall converts to dict correctly."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        call = LLMCall(
            timestamp=timestamp,
            model="claude-sonnet-4",
            role="child",
            generation=2,
            trial_id="trial_001",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.0045,
        )

        data = call.to_dict()

        assert data["timestamp"] == "2024-01-15T14:30:00"
        assert data["model"] == "claude-sonnet-4"
        assert data["trial_id"] == "trial_001"

    def test_llm_call_from_dict(self):
        """LLMCall creates from dict correctly."""
        data = {
            "timestamp": "2024-01-15T14:30:00",
            "model": "claude-haiku",
            "role": "child",
            "generation": 3,
            "trial_id": "trial_002",
            "input_tokens": 300,
            "output_tokens": 150,
            "cost_usd": 0.00084,
        }

        call = LLMCall.from_dict(data)

        assert call.model == "claude-haiku"
        assert call.generation == 3
        assert call.trial_id == "trial_002"


class TestCostCalculation:
    """Tests for cost calculation logic."""

    def test_cost_calculation_claude_sonnet(self):
        """Correct cost calculation for claude-sonnet-4."""
        tracker = CostTracker(max_cost_usd=100.0)

        # 1000 input tokens * $0.003/1K + 500 output tokens * $0.015/1K
        # = $0.003 + $0.0075 = $0.0105
        cost = tracker._calculate_cost("claude-sonnet-4", 1000, 500)

        assert abs(cost - 0.0105) < 0.0001

    def test_cost_calculation_claude_haiku(self):
        """Correct cost calculation for claude-haiku."""
        tracker = CostTracker(max_cost_usd=100.0)

        # 1000 input * $0.0008/1K + 500 output * $0.004/1K
        # = $0.0008 + $0.002 = $0.0028
        cost = tracker._calculate_cost("claude-haiku", 1000, 500)

        assert abs(cost - 0.0028) < 0.0001

    def test_unknown_model_raises(self):
        """Unknown model raises UnknownModelError."""
        tracker = CostTracker(max_cost_usd=100.0)

        with pytest.raises(UnknownModelError):
            tracker._calculate_cost("unknown-model", 1000, 500)


class TestCostTrackerCore:
    """Tests for CostTracker core functionality."""

    def test_record_call(self):
        """Records call with correct cost."""
        tracker = CostTracker(max_cost_usd=100.0)

        call = tracker.record_call(
            model="claude-sonnet-4",
            role="root",
            generation=1,
            input_tokens=1000,
            output_tokens=500,
        )

        assert len(tracker.calls) == 1
        assert call.cost_usd > 0
        assert call.model == "claude-sonnet-4"

    def test_record_multiple_calls(self):
        """Multiple calls tracked correctly."""
        tracker = CostTracker(max_cost_usd=100.0)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)
        tracker.record_call("claude-haiku", "child", 1, 500, 200, "trial_001")
        tracker.record_call("claude-haiku", "child", 1, 500, 200, "trial_002")

        assert len(tracker.calls) == 3


class TestCostTrackerQueries:
    """Tests for CostTracker query methods."""

    def test_get_total_cost(self):
        """Total cost sums correctly."""
        tracker = CostTracker(max_cost_usd=100.0)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)
        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

        total = tracker.get_total_cost()

        # Two identical calls
        assert abs(total - 0.021) < 0.001

    def test_get_remaining_budget(self):
        """Remaining budget calculated correctly."""
        tracker = CostTracker(max_cost_usd=1.0)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

        remaining = tracker.get_remaining_budget()

        assert remaining < 1.0
        assert remaining > 0.98

    def test_would_exceed_budget_true(self):
        """Detects when estimated cost would exceed budget."""
        tracker = CostTracker(max_cost_usd=0.02)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

        # Adding another $0.0105 would exceed $0.02
        assert tracker.would_exceed_budget(0.0105) is True

    def test_would_exceed_budget_false(self):
        """Allows when estimated cost is within budget."""
        tracker = CostTracker(max_cost_usd=100.0)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

        assert tracker.would_exceed_budget(0.01) is False


class TestCostTrackerSummaries:
    """Tests for CostTracker summary methods."""

    def test_get_summary(self):
        """Summary has correct structure and values."""
        tracker = CostTracker(max_cost_usd=100.0)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)
        tracker.record_call("claude-haiku", "child", 1, 500, 200, "trial_001")
        tracker.record_call("claude-haiku", "child", 1, 600, 300, "trial_002")

        summary = tracker.get_summary()

        assert "root_llm" in summary
        assert "child_llm" in summary
        assert summary["root_llm"]["calls"] == 1
        assert summary["child_llm"]["calls"] == 2
        assert summary["root_llm"]["input_tokens"] == 1000
        assert summary["child_llm"]["input_tokens"] == 1100

    def test_get_per_generation_costs(self):
        """Per-generation breakdown is correct."""
        tracker = CostTracker(max_cost_usd=100.0)

        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)
        tracker.record_call("claude-haiku", "child", 1, 500, 200, "trial_001")
        tracker.record_call("claude-sonnet-4", "root", 2, 1000, 500)
        tracker.record_call("claude-haiku", "child", 2, 500, 200, "trial_002")
        tracker.record_call("claude-haiku", "child", 2, 500, 200, "trial_003")

        per_gen = tracker.get_per_generation_costs()

        assert len(per_gen) == 2
        assert per_gen[0]["generation"] == 1
        assert per_gen[0]["trials"] == 1  # trial_001
        assert per_gen[1]["generation"] == 2
        assert per_gen[1]["trials"] == 2  # trial_002, trial_003


class TestCostTrackerPersistence:
    """Tests for CostTracker save/load functionality."""

    def test_save_creates_file(self, tmp_path):
        """Save creates JSON file."""
        tracker = CostTracker(max_cost_usd=100.0)
        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

        path = tmp_path / "costs.json"
        tracker.save(path)

        assert path.exists()

    def test_save_valid_json(self, tmp_path):
        """Saved file is valid JSON."""
        tracker = CostTracker(max_cost_usd=100.0)
        tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

        path = tmp_path / "costs.json"
        tracker.save(path)

        with open(path) as f:
            data = json.load(f)

        assert "max_cost_usd" in data
        assert "calls" in data
        assert "summary" in data

    def test_load_roundtrip(self, tmp_path):
        """Save then load preserves state."""
        tracker1 = CostTracker(max_cost_usd=50.0)
        tracker1.record_call("claude-sonnet-4", "root", 1, 1000, 500)
        tracker1.record_call("claude-haiku", "child", 1, 500, 200, "trial_001")

        path = tmp_path / "costs.json"
        tracker1.save(path)

        tracker2 = CostTracker.load(path)

        assert tracker2.max_cost_usd == 50.0
        assert len(tracker2.calls) == 2
        assert abs(tracker2.get_total_cost() - tracker1.get_total_cost()) < 0.0001

    def test_load_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CostTracker.load(Path("/nonexistent/costs.json"))
