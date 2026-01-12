"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...cost_tracker import CostTracker


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
    reasoning_content: str | None = None  # Reasoning/thinking content (OpenRouter)
    reasoning_tokens: int = 0  # Number of reasoning tokens used


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement the generate() method with a consistent interface.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
        llm_type: str,
        max_retries: int = 3,
    ):
        """
        Initialize the provider.

        Args:
            model: Model identifier
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
        """
        self.model = model
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self.max_retries = max_retries

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        enable_caching: bool = True,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with "role" and "content"
            system: Optional system prompt (string or content blocks)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_caching: Whether to enable prompt caching (provider-specific)

        Returns:
            LLMResponse with content and token usage
        """
        pass

    @property
    def supports_caching(self) -> bool:
        """Whether this provider supports prompt caching."""
        return False
