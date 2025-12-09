"""
Base classes and types for LLM clients.

This module provides the abstract base class and shared types for LLM clients,
enabling support for multiple providers (Anthropic, OpenAI, etc.) with a
unified interface that handles prompt caching transparently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..cost_tracker import CostTracker


class CacheHint(Enum):
    """
    Hints for prompt caching behavior.

    Different LLM providers implement caching differently:
    - Anthropic: Uses explicit cache_control blocks with "ephemeral" type
    - OpenAI: Automatic prefix caching (hints are informational)
    - Google: Context caching with explicit cache creation

    The LLM client implementations translate these abstract hints
    to provider-specific formats.
    """

    NONE = "none"  # No caching - content may change frequently
    EPHEMERAL = "ephemeral"  # Cache for session duration (recommended for system prompts)


@dataclass
class Message:
    """
    A message in the conversation with optional caching hint.

    Attributes:
        role: The role of the message sender ("user", "assistant", or "system")
        content: The text content of the message
        cache_hint: Optional hint for caching behavior (provider-specific handling)
    """

    role: str  # "user" | "assistant" | "system"
    content: str
    cache_hint: CacheHint = CacheHint.NONE


@dataclass
class SystemPrompt:
    """
    A system prompt with support for cacheable sections.

    System prompts can be split into cacheable and non-cacheable parts.
    The cacheable parts (typically static content) will be cached by
    providers that support it.

    Attributes:
        parts: List of content parts with individual cache hints
    """

    parts: list[Message] = field(default_factory=list)

    @classmethod
    def from_string(cls, text: str, cache: bool = True) -> "SystemPrompt":
        """
        Create a system prompt from a plain string.

        Args:
            text: The system prompt text
            cache: Whether to mark the content as cacheable

        Returns:
            SystemPrompt instance
        """
        hint = CacheHint.EPHEMERAL if cache else CacheHint.NONE
        return cls(parts=[Message(role="system", content=text, cache_hint=hint)])

    @classmethod
    def from_parts(
        cls, static_content: str, dynamic_content: str | None = None
    ) -> "SystemPrompt":
        """
        Create a system prompt with static (cached) and dynamic parts.

        Args:
            static_content: Static content that should be cached
            dynamic_content: Dynamic content that changes per request

        Returns:
            SystemPrompt instance with appropriate cache hints
        """
        parts = [
            Message(role="system", content=static_content, cache_hint=CacheHint.EPHEMERAL)
        ]
        if dynamic_content:
            parts.append(
                Message(role="system", content=dynamic_content, cache_hint=CacheHint.NONE)
            )
        return cls(parts=parts)


@dataclass
class LLMResponse:
    """
    Response from an LLM API call.

    Attributes:
        content: The generated text content
        input_tokens: Total input tokens processed
        output_tokens: Output tokens generated
        model: Model identifier used
        call_id: Unique identifier for this call
        stop_reason: Why generation stopped (e.g., "end_turn", "max_tokens")
        cache_creation_input_tokens: Tokens written to cache (if supported)
        cache_read_input_tokens: Tokens read from cache (if supported)
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    call_id: str
    stop_reason: str | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    This class defines the interface that all LLM client implementations
    must follow. Each provider-specific implementation handles:
    - API authentication and calls
    - Prompt caching in provider-specific format
    - Cost tracking integration
    - Retry logic for transient errors

    Subclasses must implement:
    - generate(): Make an LLM API call
    - _apply_message_caching(): Convert abstract cache hints to provider format
    - _apply_system_prompt_caching(): Convert system prompt to provider format
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
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

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any] | Message],
        system: str | SystemPrompt | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        The implementation should:
        1. Apply caching to messages and system prompt
        2. Make the API call with retries
        3. Record usage with cost tracker
        4. Return standardized response

        Args:
            messages: List of conversation messages. Can be:
                      - List of Message objects (recommended)
                      - List of dicts with "role" and "content" keys
            system: Optional system prompt. Can be:
                    - A plain string (will use default caching behavior)
                    - A SystemPrompt object with explicit cache hints
                    - A list of content blocks (provider-specific, for backwards compat)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded before the call
            Provider-specific errors: If the API call fails after retries
        """
        pass

    @abstractmethod
    def _apply_message_caching(
        self, messages: list[dict[str, Any] | Message]
    ) -> list[dict[str, Any]]:
        """
        Convert messages with cache hints to provider-specific format.

        Args:
            messages: List of messages (Message objects or dicts)

        Returns:
            Provider-specific message format with caching applied
        """
        pass

    @abstractmethod
    def _apply_system_prompt_caching(
        self, system: str | SystemPrompt | list[dict[str, Any]] | None
    ) -> Any:
        """
        Convert system prompt to provider-specific format with caching.

        Args:
            system: System prompt in various formats

        Returns:
            Provider-specific system prompt format
        """
        pass

    def _normalize_messages(
        self, messages: list[dict[str, Any] | Message]
    ) -> list[Message]:
        """
        Normalize messages to Message objects.

        Args:
            messages: Mixed list of Message objects and dicts

        Returns:
            List of Message objects
        """
        result = []
        for msg in messages:
            if isinstance(msg, Message):
                result.append(msg)
            elif isinstance(msg, dict):
                # Handle dict format - check if content is already structured
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Content blocks format - extract text and check for cache_control
                    text_parts = []
                    has_cache = False
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            if "cache_control" in block:
                                has_cache = True
                    content = "\n".join(text_parts)
                    hint = CacheHint.EPHEMERAL if has_cache else CacheHint.NONE
                else:
                    hint = CacheHint.NONE

                result.append(
                    Message(
                        role=msg.get("role", "user"),
                        content=content,
                        cache_hint=hint,
                    )
                )
            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")
        return result


class MockLLMClient(BaseLLMClient):
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
        max_retries: int = 3,
    ):
        """
        Initialize the mock client.

        Args:
            model: Model identifier
            cost_tracker: CostTracker instance
            llm_type: Either "root" or "child"
            responses: List of responses to return in order
            max_retries: Ignored (for interface compatibility)
        """
        super().__init__(model, cost_tracker, llm_type, max_retries)
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
        messages: list[dict[str, Any] | Message],
        system: str | SystemPrompt | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a mock response."""
        import uuid

        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Normalize messages for recording
        normalized = self._normalize_messages(messages)

        # Extract system text
        system_text = None
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, SystemPrompt):
            system_text = " ".join(p.content for p in system.parts)
        elif isinstance(system, list):
            system_text = " ".join(
                b.get("text", "") for b in system if isinstance(b, dict)
            )

        # Record the call
        self.call_history.append(
            {
                "messages": [{"role": m.role, "content": m.content} for m in normalized],
                "system": system_text,
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
        input_tokens = sum(len(m.content) // 4 for m in normalized)
        if system_text:
            input_tokens += len(system_text) // 4
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

    def _apply_message_caching(
        self, messages: list[dict[str, Any] | Message]
    ) -> list[dict[str, Any]]:
        """Mock implementation - just normalize to dicts."""
        normalized = self._normalize_messages(messages)
        return [{"role": m.role, "content": m.content} for m in normalized]

    def _apply_system_prompt_caching(
        self, system: str | SystemPrompt | list[dict[str, Any]] | None
    ) -> str | None:
        """Mock implementation - just extract text."""
        if system is None:
            return None
        if isinstance(system, str):
            return system
        if isinstance(system, SystemPrompt):
            return " ".join(p.content for p in system.parts)
        if isinstance(system, list):
            return " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        return None
