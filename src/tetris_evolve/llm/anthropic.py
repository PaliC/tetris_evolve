"""
Anthropic LLM client implementation.

This module provides the Anthropic-specific LLM client that handles
prompt caching using Anthropic's cache_control blocks.
"""

import uuid
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import (
    BaseLLMClient,
    CacheHint,
    LLMResponse,
    Message,
    SystemPrompt,
)


class AnthropicLLMClient(BaseLLMClient):
    """
    Anthropic-specific LLM client with prompt caching support.

    Implements prompt caching using Anthropic's cache_control blocks:
    - System prompts with CacheHint.EPHEMERAL get cache_control: {"type": "ephemeral"}
    - Message history caching: marks second-to-last user message for prefix caching
    - Cache statistics are tracked and reported in LLMResponse
    """

    def __init__(
        self,
        model: str,
        cost_tracker: Any,  # CostTracker, avoiding circular import
        llm_type: str,
        max_retries: int = 3,
    ):
        """
        Initialize the Anthropic LLM client.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
        """
        super().__init__(model, cost_tracker, llm_type, max_retries)
        self._client = anthropic.Anthropic()

    def generate(
        self,
        messages: list[dict[str, Any] | Message],
        system: str | SystemPrompt | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a response from the Anthropic API.

        Applies Anthropic-specific prompt caching:
        - System prompts are converted to content blocks with cache_control
        - Message history gets cache_control on second-to-last user message

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and token usage (including cache stats)

        Raises:
            BudgetExceededError: If budget is exceeded before the call
            anthropic.APIError: If the API call fails after retries
        """
        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())

        # Apply caching to messages and system prompt
        cached_messages = self._apply_message_caching(messages)
        cached_system = self._apply_system_prompt_caching(system)

        # Prepare API call kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": cached_messages,
        }

        if cached_system:
            kwargs["system"] = cached_system

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

    def _apply_message_caching(
        self, messages: list[dict[str, Any] | Message]
    ) -> list[dict[str, Any]]:
        """
        Convert messages with cache hints to Anthropic format.

        Applies two caching strategies:
        1. Respects CacheHint.EPHEMERAL on individual messages
        2. Automatically adds cache_control to second-to-last user message
           to cache the conversation prefix (most effective for multi-turn)

        Args:
            messages: List of messages (Message objects or dicts)

        Returns:
            Anthropic-formatted messages with cache_control blocks
        """
        # First normalize all messages
        normalized = self._normalize_messages(messages)

        if not normalized:
            return []

        # Find user message indices for automatic prefix caching
        user_indices = [i for i, m in enumerate(normalized) if m.role == "user"]

        # Determine which message to cache for prefix caching
        # Cache the second-to-last user message to cache all prior context
        auto_cache_index = user_indices[-2] if len(user_indices) >= 2 else None

        result = []
        for i, msg in enumerate(normalized):
            # Determine if this message should be cached
            should_cache = (
                msg.cache_hint == CacheHint.EPHEMERAL or i == auto_cache_index
            )

            if should_cache:
                result.append({
                    "role": msg.role,
                    "content": [
                        {
                            "type": "text",
                            "text": msg.content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                })
            else:
                result.append({"role": msg.role, "content": msg.content})

        return result

    def _apply_system_prompt_caching(
        self, system: str | SystemPrompt | list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """
        Convert system prompt to Anthropic format with caching.

        Handles multiple input formats:
        - Plain string: Converted to cacheable content block
        - SystemPrompt: Each part converted with its cache hint
        - List of dicts: Passed through (assumed to be Anthropic format already)

        Args:
            system: System prompt in various formats

        Returns:
            Anthropic-formatted system content blocks, or None
        """
        if system is None:
            return None

        if isinstance(system, str):
            # Plain string - cache by default
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if isinstance(system, SystemPrompt):
            # Convert SystemPrompt parts to Anthropic format
            result = []
            for part in system.parts:
                block: dict[str, Any] = {
                    "type": "text",
                    "text": part.content,
                }
                if part.cache_hint == CacheHint.EPHEMERAL:
                    block["cache_control"] = {"type": "ephemeral"}
                result.append(block)
            return result

        if isinstance(system, list):
            # Already in Anthropic format - pass through
            return system

        raise TypeError(f"Unsupported system prompt type: {type(system)}")

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
