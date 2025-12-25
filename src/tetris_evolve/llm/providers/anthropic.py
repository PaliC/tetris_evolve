"""Anthropic Claude API provider."""

import uuid
from typing import TYPE_CHECKING, Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMProvider, LLMResponse

if TYPE_CHECKING:
    from ...cost_tracker import CostTracker


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider with prompt caching support.

    Integrates with CostTracker for budget enforcement.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
        llm_type: str,
        max_retries: int = 3,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
        """
        super().__init__(model, cost_tracker, llm_type, max_retries)
        self._client = anthropic.Anthropic()

    @property
    def supports_caching(self) -> bool:
        """Anthropic supports ephemeral prompt caching."""
        return True

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,
    ) -> LLMResponse:
        """
        Generate a response from Claude.

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
