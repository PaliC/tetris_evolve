"""Google Gemini API provider."""

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, cast

from google import genai
from google.genai import types
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


class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini API provider with thinking/reasoning support.

    Uses the google-genai SDK directly for native Gemini access.
    Environment variable: GEMINI_API_KEY
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
        Initialize the Google Gemini provider.

        Args:
            model: Model identifier (e.g., "gemini-2.5-flash")
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
            reasoning_config: Optional reasoning configuration dict with keys:
                - enabled: bool - Enable reasoning/thinking
                - effort: str - "none", "minimal", "low", "medium", "high", "xhigh"
                - max_tokens: int - Direct token limit for thinking (overrides effort)

        Raises:
            ValueError: If GEMINI_API_KEY is not set
        """
        super().__init__(model, cost_tracker, llm_type, max_retries)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your API key from https://aistudio.google.com/apikey"
            )

        self._client = genai.Client(api_key=api_key)
        self._reasoning_config = reasoning_config

    @property
    def supports_caching(self) -> bool:
        """Google Gemini has context caching but different from Anthropic's ephemeral caching."""
        return False

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        enable_caching: bool = True,  # noqa: ARG002 - Ignored for Google
    ) -> LLMResponse:
        """
        Generate a response using Google Gemini API.

        Args:
            messages: List of message dicts with "role" and "content"
            system: Optional system prompt (string or content blocks)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_caching: Ignored (Google uses different caching mechanism)

        Returns:
            LLMResponse with content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded before the call
        """
        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())

        # Convert messages to Gemini format
        contents = self._convert_messages_to_contents(messages)

        # Build generation config
        config = self._build_generate_config(
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Make the API call with retries
        response = self._call_with_retry(contents=contents, config=config)

        # Extract content and reasoning
        content, reasoning_content = self._extract_content(response)

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0

        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            # Gemini may report thinking tokens separately
            reasoning_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0

        # Record usage
        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_type=self.llm_type,
            call_id=call_id,
        )

        # Determine stop reason
        stop_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            stop_reason = finish_reason.name if finish_reason else None

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            call_id=call_id,
            stop_reason=stop_reason,
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
        )

    def _convert_messages_to_contents(
        self, messages: list[dict[str, Any]]
    ) -> list[types.Content]:
        """Convert Anthropic-style messages to Gemini Content objects."""
        contents = []
        for msg in messages:
            # Gemini uses "user" and "model" roles (not "assistant")
            role = "user" if msg["role"] == "user" else "model"
            text = msg.get("content", "")

            # Handle Anthropic-style content blocks
            if isinstance(text, list):
                text = self._extract_text_from_blocks(text)

            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        return contents

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
        """Extract text from Anthropic-style content blocks."""
        texts = []
        for block in blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    def _compute_thinking_budget(self, max_tokens: int) -> int:
        """Convert effort level to thinking_budget as percentage of max_tokens."""
        if not self._reasoning_config:
            return 0

        # Direct token limit takes precedence
        if self._reasoning_config.get("max_tokens"):
            return self._reasoning_config["max_tokens"]

        effort = self._reasoning_config.get("effort", "medium")
        effort_percentages = {
            "none": 0,
            "minimal": 0.10,  # ~10% of max_tokens
            "low": 0.20,  # ~20%
            "medium": 0.50,  # ~50%
            "high": 0.80,  # ~80%
            "xhigh": 0.95,  # ~95%
        }
        return int(max_tokens * effort_percentages.get(effort, 0.5))

    def _build_generate_config(
        self,
        system: str | list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
    ) -> types.GenerateContentConfig:
        """Build the Gemini generation config."""
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add system instruction if provided
        if system:
            system_text = self._extract_system_text(system)
            config_kwargs["system_instruction"] = system_text

        # Add thinking config if reasoning is enabled
        if self._reasoning_config and self._reasoning_config.get("enabled"):
            thinking_budget = self._compute_thinking_budget(max_tokens)
            if thinking_budget > 0:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

        return types.GenerateContentConfig(**config_kwargs)

    def _extract_content(self, response) -> tuple[str, str | None]:
        """Extract regular content and thinking content from response."""
        content = ""
        reasoning_content = None

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Check if this is thinking/reasoning content
                    if hasattr(part, "thought") and part.thought:
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += part.text or ""
                    else:
                        content += part.text or ""

        return content, reasoning_content

    def _call_with_retry(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ):
        """Make API call with retry logic for errors and empty responses."""

        def _is_empty_response(response) -> bool:
            """Check if response content is empty (triggers retry)."""
            if not response.candidates:
                return True
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return True
            # Check if there's any non-empty text
            text = "".join(
                part.text or ""
                for part in candidate.content.parts
                if not (hasattr(part, "thought") and part.thought)
            )
            return not text.strip()

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
                logger.warning(
                    "Empty response from %s, retrying (attempt %d/%d)",
                    self.model,
                    retry_state.attempt_number,
                    self.max_retries + 1,
                )

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=(
                retry_if_exception_type(
                    (
                        # Google API common transient errors
                        ConnectionError,
                        TimeoutError,
                    )
                )
                | retry_if_result(_is_empty_response)
            ),
            before_sleep=_log_retry,
            reraise=True,
        )
        def _make_call():
            # Cast needed because ty has issues with google-genai's complex union types
            # list[Content] is valid per SDK docs and runtime tests pass
            return self._client.models.generate_content(
                model=self.model,
                contents=cast(Any, contents),
                config=config,
            )

        return _make_call()
