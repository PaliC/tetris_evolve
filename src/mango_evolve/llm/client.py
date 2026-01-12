"""
LLM Client for mango_evolve.

Provides a factory function to create LLM clients for different providers.
Supports Anthropic Claude (with prompt caching) and OpenRouter (100+ models).
"""

import uuid
from typing import TYPE_CHECKING, Any

from .providers.anthropic import AnthropicProvider
from .providers.base import BaseLLMProvider, LLMResponse
from .providers.google import GoogleProvider
from .providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from ..cost_tracker import CostTracker

# Re-export LLMResponse for backwards compatibility
__all__ = ["LLMResponse", "LLMClient", "MockLLMClient", "create_llm_client", "BaseLLMProvider"]


def create_llm_client(
    provider: str,
    model: str,
    cost_tracker: "CostTracker",
    llm_type: str,
    max_retries: int = 3,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseLLMProvider:
    """
    Factory function to create an LLM client for the specified provider.

    Args:
        provider: Provider name ("anthropic" or "openrouter")
        model: Model identifier
        cost_tracker: CostTracker instance for budget enforcement
        llm_type: Either "root" or "child" - used for cost tracking
        max_retries: Maximum number of retries on transient errors
        reasoning_config: Optional reasoning configuration (OpenRouter and Google only)

    Returns:
        LLM client instance (AnthropicProvider, GoogleProvider, or OpenRouterProvider)

    Raises:
        ValueError: If provider is unknown
    """
    providers = {
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "openrouter": OpenRouterProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: '{provider}'. Supported providers: {list(providers.keys())}"
        )

    # OpenRouter and Google support reasoning config
    if provider in ("openrouter", "google"):
        return providers[provider](
            model=model,
            cost_tracker=cost_tracker,
            llm_type=llm_type,
            max_retries=max_retries,
            reasoning_config=reasoning_config,
        )

    return providers[provider](
        model=model,
        cost_tracker=cost_tracker,
        llm_type=llm_type,
        max_retries=max_retries,
    )


# Backwards compatibility alias
LLMClient = AnthropicProvider


class MockLLMClient:
    """
    Mock LLM client for testing.

    Returns predefined responses without making API calls.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
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
        max_tokens: int = 8192,
        temperature: float = 0.7,
        enable_caching: bool = True,  # noqa: ARG002
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
