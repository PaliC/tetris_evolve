"""OpenRouter API provider."""

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...cost_tracker import CostTracker


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API provider.

    Uses OpenAI SDK with custom base_url for OpenRouter.
    Supports 100+ models including Gemini, Claude, Llama, Mistral, etc.

    Environment variable: OPENROUTER_API_KEY
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
        llm_type: str,
        max_retries: int = 3,
        reasoning_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the OpenRouter provider.

        Args:
            model: Model identifier (e.g., "google/gemini-2.0-flash-001")
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
            reasoning_config: Optional reasoning configuration dict with keys:
                - enabled: bool - Enable reasoning at medium effort
                - effort: str - "minimal", "low", "medium", "high", "xhigh"
                - max_tokens: int - Direct token limit for reasoning
                - exclude: bool - Use reasoning but don't return it

        Raises:
            ValueError: If OPENROUTER_API_KEY is not set
        """
        super().__init__(model, cost_tracker, llm_type, max_retries)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your API key from https://openrouter.ai/keys"
            )

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._reasoning_config = reasoning_config

    @property
    def supports_caching(self) -> bool:
        """OpenRouter doesn't support Anthropic-style prompt caching."""
        return False

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,  # noqa: ARG002 - Ignored for OpenRouter
    ) -> LLMResponse:
        """
        Generate a response using OpenRouter API.

        Args:
            messages: List of message dicts with "role" and "content"
            system: Optional system prompt (string or content blocks)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_caching: Ignored (OpenRouter doesn't support caching)

        Returns:
            LLMResponse with content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded before the call
        """
        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())

        # Build messages list with system prompt
        api_messages: list[dict[str, str]] = []

        # Handle system prompt (convert to first message if present)
        if system:
            system_text = self._extract_system_text(system)
            api_messages.append({"role": "system", "content": system_text})

        # Add conversation messages
        for msg in messages:
            content = msg.get("content", "")
            # Handle Anthropic-style content blocks
            if isinstance(content, list):
                content = self._extract_text_from_blocks(content)
            api_messages.append({"role": msg["role"], "content": content})

        # Make the API call with retries
        response = self._call_with_retry(
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract content
        content = response.choices[0].message.content or ""

        # Extract reasoning content if present
        reasoning_content = None
        reasoning_tokens = 0
        message = response.choices[0].message

        # OpenRouter returns reasoning in message.reasoning field
        if hasattr(message, "reasoning") and message.reasoning:
            reasoning_content = message.reasoning

        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Extract reasoning tokens if available
        if response.usage and hasattr(response.usage, "reasoning_tokens"):
            reasoning_tokens = response.usage.reasoning_tokens or 0

        # Record usage (no cache stats for OpenRouter)
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
            stop_reason=response.choices[0].finish_reason,
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
        )

    def _extract_system_text(self, system: str | list[dict[str, Any]]) -> str:
        """Extract text from system prompt (handles both string and content blocks)."""
        if isinstance(system, str):
            return system
        # Handle Anthropic-style content blocks
        texts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    def _extract_text_from_blocks(self, blocks: list[dict[str, Any]]) -> str:
        """Extract text from content blocks."""
        texts = []
        for block in blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    def _build_reasoning_param(self) -> dict[str, Any] | None:
        """Build the reasoning parameter for the API call by filtering out None/False values."""
        if not self._reasoning_config:
            return None

        # Filter out None, False, and "none" effort - keep only truthy values
        reasoning = {
            k: v
            for k, v in self._reasoning_config.items()
            if v is not None and v is not False and v != "none"
        }
        return reasoning or None

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ):
        """Make API call with retry logic for errors and empty responses."""

        def _is_empty_response(response) -> bool:
            """Check if response content is empty (triggers retry)."""
            if not response.choices:
                return True
            content = response.choices[0].message.content
            return not content or not content.strip()

        def _log_retry(retry_state: RetryCallState) -> None:
            """Log retry attempts with appropriate context."""
            outcome = retry_state.outcome
            if outcome is not None and outcome.failed:
                logger.warning(
                    "API error for %s, retrying (attempt %d/%d): %s",
                    self.model,
                    retry_state.attempt_number,
                    self.max_retries + 1,
                    outcome.exception(),
                )
            else:
                # Empty response - log reasoning info for debugging
                reasoning_info = ""
                if outcome is not None:
                    response = outcome.result()
                    if response and response.choices:
                        message = response.choices[0].message
                        reasoning = getattr(message, "reasoning", None)
                        if reasoning:
                            reasoning_len = len(reasoning) if reasoning else 0
                            reasoning_info = f" (has {reasoning_len} chars of reasoning)"
                logger.warning(
                    "Empty response from %s, retrying (attempt %d/%d)%s",
                    self.model,
                    retry_state.attempt_number,
                    self.max_retries + 1,
                    reasoning_info,
                )

        # Build reasoning parameter
        reasoning = self._build_reasoning_param()

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=(
                retry_if_exception_type((RateLimitError, APIConnectionError))
                | retry_if_result(_is_empty_response)
            ),
            before_sleep=_log_retry,
            reraise=True,
        )
        def _make_call():
            # Build extra body with reasoning if configured
            extra_body = {}
            if reasoning:
                extra_body["reasoning"] = reasoning

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            return self._client.chat.completions.create(**kwargs)

        return _make_call()
