"""
LLM Client for tetris_evolve.

Wraps the Anthropic API with cost tracking and budget enforcement.
Supports prompt caching for reduced costs on repeated prefixes.
"""

import uuid
from dataclasses import dataclass
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..cost_tracker import CostTracker


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    call_id: str
    stop_reason: str | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class LLMClient:
    """
    Client for making LLM API calls with cost tracking.

    Integrates with CostTracker for budget enforcement.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: CostTracker,
        llm_type: str,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
        """
        self.model = model
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self.max_retries = max_retries
        self._client = anthropic.Anthropic()

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with "role" and "content".
                      Content can be a string or list of content blocks.
                      For caching, use content blocks with cache_control.
            system: Optional system prompt. Can be:
                    - A plain string (will be converted to cacheable block if enable_caching=True)
                    - A list of content blocks for fine-grained cache control
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_caching: Whether to enable prompt caching (default True)

        Returns:
            LLMResponse with content and token usage (including cache stats)

        Raises:
            BudgetExceededError: If budget is exceeded before the call
            anthropic.APIError: If the API call fails after retries
        """
        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())

        # Prepare API call kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            if isinstance(system, str) and enable_caching:
                # Convert plain string to cacheable content block
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system

        if temperature is not None:
            kwargs["temperature"] = temperature

        # Make the API call with retries
        response = self._call_with_retry(**kwargs)

        # Extract content
        content = ""
        if response.content:
            content = response.content[0].text

        # Record usage (including cache tokens)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Extract cache statistics from response
        cache_creation_input_tokens = getattr(
            response.usage, "cache_creation_input_tokens", 0
        ) or 0
        cache_read_input_tokens = getattr(
            response.usage, "cache_read_input_tokens", 0
        ) or 0

        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_type=self.llm_type,
            call_id=call_id,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            call_id=call_id,
            stop_reason=response.stop_reason,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    def _call_with_retry(self, **kwargs) -> anthropic.types.Message:
        """
        Make an API call with retry logic for transient errors.

        Args:
            **kwargs: Arguments to pass to the API

        Returns:
            API response

        Raises:
            anthropic.APIError: If all retries fail
        """

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (
                    anthropic.RateLimitError,
                    anthropic.APIConnectionError,
                )
            ),
            reraise=True,
        )
        def _make_call():
            return self._client.messages.create(**kwargs)

        return _make_call()


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
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,
    ) -> LLMResponse:
        """
        Generate a mock response.

        Args:
            messages: List of message dicts
            system: Optional system prompt (string or content blocks)
            max_tokens: Maximum tokens (ignored)
            temperature: Temperature (ignored)
            enable_caching: Whether to enable caching (ignored in mock)

        Returns:
            LLMResponse with mock content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded
            IndexError: If no more responses are available
        """
        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Record the call
        self.call_history.append(
            {
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

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
            if isinstance(system, str):
                input_tokens += len(system) // 4
            elif isinstance(system, list):
                # Handle content blocks format
                for block in system:
                    if isinstance(block, dict) and "text" in block:
                        input_tokens += len(block["text"]) // 4
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
