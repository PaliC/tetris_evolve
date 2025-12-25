"""
Tests for the LLM Client module.
"""

from unittest.mock import MagicMock, patch

import pytest

from mango_evolve import (
    BudgetExceededError,
    CostTracker,
)
from mango_evolve.llm import LLMClient, LLMResponse, MockLLMClient


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_generate_returns_response(self, sample_config):
        """Test that generate returns a response."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Hello, world!"],
        )

        response = client.generate(
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello, world!"
        assert response.model == "claude-test"

    def test_generate_multiple_responses(self, sample_config):
        """Test that multiple calls return responses in order."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["First", "Second", "Third"],
        )

        r1 = client.generate(messages=[{"role": "user", "content": "1"}])
        r2 = client.generate(messages=[{"role": "user", "content": "2"}])
        r3 = client.generate(messages=[{"role": "user", "content": "3"}])

        assert r1.content == "First"
        assert r2.content == "Second"
        assert r3.content == "Third"

    def test_cost_recorded(self, sample_config):
        """Test that usage is recorded in cost tracker."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Response with some tokens"],
        )

        client.generate(
            messages=[{"role": "user", "content": "Test message"}],
        )

        assert len(cost_tracker.usage_log) == 1
        assert cost_tracker.usage_log[0].llm_type == "child"
        assert cost_tracker.total_cost > 0

    def test_budget_exceeded_raises(self, sample_config):
        """Test that BudgetExceededError is raised when budget exceeded."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Response"],
        )

        # Artificially exceed budget
        cost_tracker.total_cost = sample_config.budget.max_total_cost + 1

        with pytest.raises(BudgetExceededError):
            client.generate(
                messages=[{"role": "user", "content": "Test"}],
            )

    def test_add_response(self, sample_config):
        """Test adding responses dynamically."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        client.add_response("Dynamic response")
        response = client.generate(
            messages=[{"role": "user", "content": "Test"}],
        )

        assert response.content == "Dynamic response"

    def test_set_responses(self, sample_config):
        """Test setting responses list."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Old response"],
        )

        client.set_responses(["New response 1", "New response 2"])
        r1 = client.generate(messages=[{"role": "user", "content": "1"}])
        r2 = client.generate(messages=[{"role": "user", "content": "2"}])

        assert r1.content == "New response 1"
        assert r2.content == "New response 2"

    def test_call_history_recorded(self, sample_config):
        """Test that call history is recorded."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Response"],
        )

        client.generate(
            messages=[{"role": "user", "content": "Test message"}],
            system="System prompt",
            max_tokens=1000,
            temperature=0.5,
        )

        assert len(client.call_history) == 1
        call = client.call_history[0]
        assert call["messages"] == [{"role": "user", "content": "Test message"}]
        assert call["system"] == "System prompt"
        assert call["max_tokens"] == 1000
        assert call["temperature"] == 0.5

    def test_no_more_responses_raises(self, sample_config):
        """Test that IndexError is raised when no more responses."""
        cost_tracker = CostTracker(sample_config)
        client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Only one"],
        )

        client.generate(messages=[{"role": "user", "content": "1"}])

        with pytest.raises(IndexError, match="No more mock responses"):
            client.generate(messages=[{"role": "user", "content": "2"}])

    def test_different_llm_types(self, sample_config):
        """Test root vs child LLM types use different pricing."""
        cost_tracker = CostTracker(sample_config)

        root_client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="root",
            responses=["Root response"],
        )

        child_client = MockLLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["Child response"],
        )

        root_client.generate(messages=[{"role": "user", "content": "Root"}])
        child_client.generate(messages=[{"role": "user", "content": "Child"}])

        assert len(cost_tracker.usage_log) == 2
        assert cost_tracker.usage_log[0].llm_type == "root"
        assert cost_tracker.usage_log[1].llm_type == "child"


class TestLLMClient:
    """Tests for the real LLM Client (with mocked API)."""

    def test_budget_check_before_call(self, sample_config):
        """Test that budget is checked before making API call."""
        cost_tracker = CostTracker(sample_config)
        client = LLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        # Exceed budget
        cost_tracker.total_cost = sample_config.budget.max_total_cost + 1

        with pytest.raises(BudgetExceededError):
            client.generate(
                messages=[{"role": "user", "content": "Test"}],
            )

    @patch("mango_evolve.llm.providers.anthropic.anthropic.Anthropic")
    def test_generate_with_mock(self, mock_anthropic_class, sample_config):
        """Test generate with mocked Anthropic API."""
        # Set up mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        cost_tracker = CostTracker(sample_config)
        client = LLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        response = client.generate(
            messages=[{"role": "user", "content": "Test"}],
        )

        assert response.content == "Test response"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert len(cost_tracker.usage_log) == 1

    @patch("mango_evolve.llm.providers.anthropic.anthropic.Anthropic")
    def test_cost_recorded_correctly(self, mock_anthropic_class, sample_config):
        """Test that cost is computed correctly."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        cost_tracker = CostTracker(sample_config)
        client = LLMClient(
            model="claude-test",
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        client.generate(messages=[{"role": "user", "content": "Test"}])

        # Verify cost calculation
        expected_cost = (
            1000 * sample_config.child_llm.cost_per_million_input_tokens / 1_000_000
            + 500 * sample_config.child_llm.cost_per_million_output_tokens / 1_000_000
        )
        assert abs(cost_tracker.total_cost - expected_cost) < 1e-10


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLMResponse."""
        response = LLMResponse(
            content="Test content",
            input_tokens=100,
            output_tokens=50,
            model="claude-test",
            call_id="test-id",
            stop_reason="end_turn",
        )

        assert response.content == "Test content"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model == "claude-test"
        assert response.call_id == "test-id"
        assert response.stop_reason == "end_turn"

    def test_optional_stop_reason(self):
        """Test that stop_reason is optional."""
        response = LLMResponse(
            content="Test",
            input_tokens=10,
            output_tokens=5,
            model="claude-test",
            call_id="test-id",
        )

        assert response.stop_reason is None
