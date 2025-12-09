"""
LLM Client facade for tetris_evolve.

This module provides the main interface for LLM clients:
- Factory function to create clients for different providers
- Re-exports for backwards compatibility
- Type aliases for common patterns
"""

from typing import TYPE_CHECKING

# Re-export base types for backwards compatibility
from .base import (
    BaseLLMClient,
    CacheHint,
    LLMResponse,
    Message,
    MockLLMClient,
    SystemPrompt,
)

# Import provider implementations
from .anthropic import AnthropicLLMClient

if TYPE_CHECKING:
    from ..cost_tracker import CostTracker

# Type alias for backwards compatibility
LLMClient = AnthropicLLMClient

# Supported providers
SUPPORTED_PROVIDERS = {"anthropic"}

__all__ = [
    # Base types
    "BaseLLMClient",
    "CacheHint",
    "LLMResponse",
    "Message",
    "MockLLMClient",
    "SystemPrompt",
    # Provider implementations
    "AnthropicLLMClient",
    # Backwards compatibility
    "LLMClient",
    # Factory
    "create_llm_client",
    "SUPPORTED_PROVIDERS",
]


def create_llm_client(
    provider: str,
    model: str,
    cost_tracker: "CostTracker",
    llm_type: str,
    max_retries: int = 3,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client for the specified provider.

    This function creates the appropriate LLM client based on the provider.
    Each provider implementation handles prompt caching in its own way:
    - Anthropic: Uses cache_control blocks with "ephemeral" type
    - OpenAI: Automatic prefix caching (future implementation)
    - Google: Context caching (future implementation)

    Args:
        provider: Provider name ("anthropic", "openai", etc.)
        model: Model identifier (e.g., "claude-sonnet-4-20250514")
        cost_tracker: CostTracker instance for budget enforcement
        llm_type: Either "root" or "child" - used for cost tracking
        max_retries: Maximum number of retries on transient errors

    Returns:
        An LLM client instance implementing BaseLLMClient

    Raises:
        ValueError: If the provider is not supported

    Example:
        ```python
        from tetris_evolve.llm.client import create_llm_client

        client = create_llm_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_tracker=cost_tracker,
            llm_type="root",
        )

        response = client.generate(
            messages=[Message(role="user", content="Hello!")],
            system=SystemPrompt.from_string("You are helpful."),
        )
        ```
    """
    provider = provider.lower()

    if provider == "anthropic":
        return AnthropicLLMClient(
            model=model,
            cost_tracker=cost_tracker,
            llm_type=llm_type,
            max_retries=max_retries,
        )
    # Future providers can be added here:
    # elif provider == "openai":
    #     return OpenAILLMClient(...)
    # elif provider == "google":
    #     return GoogleLLMClient(...)
    else:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Supported providers: {supported}"
        )
