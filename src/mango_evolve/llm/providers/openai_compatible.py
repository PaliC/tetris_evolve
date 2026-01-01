"""Shared OpenAI-compatible provider base class."""

import logging
import uuid
from abc import ABC, abstractmethod
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


class OpenAICompatibleProvider(BaseLLMProvider, ABC):
    """
    Base class for OpenAI-compatible chat completion providers.

    Subclasses supply API key/base URL and any extra request body fields.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
        llm_type: str,
        max_retries: int = 3,
    ):
        super().__init__(model, cost_tracker, llm_type, max_retries)

        api_key = self._get_api_key()
        base_url = self._get_base_url()
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        else:
            self._client = OpenAI(api_key=api_key)

    @property
    def supports_caching(self) -> bool:
        """OpenAI-compatible providers don't support Anthropic-style prompt caching."""
        return False

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,  # noqa: ARG002 - Ignored for OpenAI-compatible providers
    ) -> LLMResponse:
        """
        Generate a response using an OpenAI-compatible API.
        """
        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())
        api_messages = self._build_api_messages(messages, system)

        response = self._call_with_retry(
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        message = response.choices[0].message
        content = message.content or ""
        reasoning_content, reasoning_tokens = self._extract_reasoning(message, response)

        input_tokens, output_tokens = self._extract_usage(response)

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

    @abstractmethod
    def _get_api_key(self) -> str:
        """Return API key or raise ValueError."""

    def _get_base_url(self) -> str | None:
        """Return base URL if needed, else None."""
        return None

    def _build_extra_body(self) -> dict[str, Any] | None:
        """Return extra body fields for the API call."""
        return None

    def _build_request_overrides(self) -> dict[str, Any] | None:
        """Return extra top-level request parameters for the API call."""
        return None

    def _extract_reasoning(self, message: Any, response: Any) -> tuple[str | None, int]:
        """Extract optional reasoning content and tokens."""
        return None, 0

    @staticmethod
    def _extract_system_text(system: str | list[dict[str, Any]]) -> str:
        """Extract text from system prompt (handles both string and content blocks)."""
        if isinstance(system, str):
            return system
        texts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    @staticmethod
    def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
        """Extract text from content blocks."""
        texts = []
        for block in blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    def _build_api_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Convert system + messages into OpenAI-compatible message list."""
        api_messages: list[dict[str, str]] = []
        if system:
            system_text = self._extract_system_text(system)
            api_messages.append({"role": "system", "content": system_text})

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = self._extract_text_from_blocks(content)
            api_messages.append({"role": msg["role"], "content": content})

        return api_messages

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int, int]:
        """Extract input/output token usage from response."""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return input_tokens, output_tokens

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ):
        """Make API call with retry logic for errors and empty responses."""

        def _is_empty_response(response) -> bool:
            if not response.choices:
                return True
            content = response.choices[0].message.content
            return not content or not content.strip()

        def _log_retry(retry_state: RetryCallState) -> None:
            if retry_state.outcome.failed:
                logger.warning(
                    "API error for %s, retrying (attempt %d/%d): %s",
                    self.model,
                    retry_state.attempt_number,
                    self.max_retries + 1,
                    retry_state.outcome.exception(),
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
                retry_if_exception_type((RateLimitError, APIConnectionError))
                | retry_if_result(_is_empty_response)
            ),
            before_sleep=_log_retry,
            reraise=True,
        )
        def _make_call():
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            overrides = self._build_request_overrides()
            if overrides:
                kwargs.update(overrides)
            extra_body = self._build_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body
            return self._client.chat.completions.create(**kwargs)

        return _make_call()
