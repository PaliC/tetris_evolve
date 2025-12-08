"""
Test utilities and mock classes for tetris_evolve tests.
"""

import uuid
from typing import Any

from tetris_evolve.cost_tracker import CostTracker
from tetris_evolve.llm.client import LLMResponse


class MockLLMClient:
    """
    Mock LLM client for testing.

    Returns predefined responses without making API calls.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: CostTracker,
        llm_type: str,
        responses: list[str] | None = None,
    ):
        """
        Initialize the mock client.

        Args:
            model: Model identifier
            cost_tracker: CostTracker instance
            llm_type: Either "root" or "child"
            responses: List of responses to return in order
        """
        self.model = model
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self._responses = responses or []
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    def set_responses(self, responses: list[str]) -> None:
        """Set the list of responses to return."""
        self._responses = responses
        self._call_count = 0

    def add_response(self, response: str) -> None:
        """Add a response to the queue."""
        self._responses.append(response)

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a mock response.

        Args:
            messages: List of message dicts
            system: Optional system prompt
            max_tokens: Maximum tokens (ignored)
            temperature: Temperature (ignored)

        Returns:
            LLMResponse with mock content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded
            IndexError: If no more responses are available
        """
        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Record the call
        self.call_history.append({
            "messages": messages,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        # Get the response
        if self._call_count >= len(self._responses):
            raise IndexError(
                f"No more mock responses available. "
                f"Called {self._call_count + 1} times but only {len(self._responses)} responses set."
            )

        content = self._responses[self._call_count]
        self._call_count += 1

        call_id = str(uuid.uuid4())

        # Estimate token counts (rough approximation)
        input_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        if system:
            input_tokens += len(system) // 4
        output_tokens = len(content) // 4

        # Record usage
        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_type=self.llm_type,
            call_id=call_id,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            call_id=call_id,
            stop_reason="end_turn",
        )
