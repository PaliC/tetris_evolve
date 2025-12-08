"""
LLM Client for tetris_evolve.

Wraps the Anthropic API with cost tracking and budget enforcement.
Also supports OpenRouter for accessing OpenAI models (GPT-5.1, GPT-OSS, etc.).
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any

import anthropic
import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..cost_tracker import CostTracker

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    call_id: str
    stop_reason: str | None = None


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
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with "role" and "content"
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and token usage

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
            kwargs["system"] = system

        if temperature is not None:
            kwargs["temperature"] = temperature

        # Make the API call with retries
        response = self._call_with_retry(**kwargs)

        # Extract content
        content = ""
        if response.content:
            content = response.content[0].text

        # Record usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

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
            stop_reason=response.stop_reason,
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
            retry=retry_if_exception_type((
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
            )),
            reraise=True,
        )
        def _make_call():
            return self._client.messages.create(**kwargs)

        return _make_call()


class OpenRouterClient(LLMClient):
    """
    Client for making LLM API calls via OpenRouter.

    OpenRouter provides access to various LLM providers including OpenAI models
    (GPT-5.1, GPT-OSS, etc.) through a unified OpenAI-compatible API.

    Subclasses LLMClient (which is the Anthropic client) to maintain the same
    interface while routing requests through OpenRouter.

    Environment variables:
        OPENROUTER_API_KEY: Required API key for OpenRouter
        OPENROUTER_SITE_URL: Optional site URL for request attribution
        OPENROUTER_APP_NAME: Optional app name for request attribution
    """

    def __init__(
        self,
        model: str,
        cost_tracker: CostTracker,
        llm_type: str,
        max_retries: int = 3,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
    ):
        """
        Initialize the OpenRouter client.

        Args:
            model: Model identifier (e.g., "openai/gpt-4o", "openai/gpt-5.1")
            cost_tracker: CostTracker instance for budget enforcement
            llm_type: Either "root" or "child" - used for cost tracking
            max_retries: Maximum number of retries on transient errors
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            site_url: Optional site URL for request attribution
            app_name: Optional app name for request attribution
        """
        # Initialize base attributes without calling parent __init__
        # since parent creates an Anthropic client we don't need
        self.model = model
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self.max_retries = max_retries

        # OpenRouter-specific configuration
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        self._app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "tetris_evolve")

        # Create OpenAI client configured for OpenRouter
        self._client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self._api_key,
            default_headers={
                "HTTP-Referer": self._site_url,
                "X-Title": self._app_name,
            },
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a response from the LLM via OpenRouter.

        Args:
            messages: List of message dicts with "role" and "content"
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded before the call
            openai.APIError: If the API call fails after retries
        """
        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())

        # Build messages list for OpenAI format
        openai_messages: list[dict[str, str]] = []

        # Add system message if provided
        if system:
            openai_messages.append({"role": "system", "content": system})

        # Add the conversation messages
        openai_messages.extend(messages)

        # Prepare API call kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": openai_messages,
            "temperature": temperature,
        }

        # Make the API call with retries
        response = self._call_with_retry(**kwargs)

        # Extract content from OpenAI response format
        content = ""
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content

        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Record usage
        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_type=self.llm_type,
            call_id=call_id,
        )

        # Map finish_reason to stop_reason
        stop_reason = None
        if response.choices:
            finish_reason = response.choices[0].finish_reason
            # Map OpenAI finish reasons to Anthropic-style stop reasons
            if finish_reason == "stop":
                stop_reason = "end_turn"
            elif finish_reason == "length":
                stop_reason = "max_tokens"
            else:
                stop_reason = finish_reason

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            call_id=call_id,
            stop_reason=stop_reason,
        )

    def _call_with_retry(self, **kwargs) -> openai.types.chat.ChatCompletion:
        """
        Make an API call with retry logic for transient errors.

        Args:
            **kwargs: Arguments to pass to the API

        Returns:
            OpenAI ChatCompletion response

        Raises:
            openai.APIError: If all retries fail
        """

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((
                openai.RateLimitError,
                openai.APIConnectionError,
            )),
            reraise=True,
        )
        def _make_call():
            return self._client.chat.completions.create(**kwargs)

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
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a mock response.

        Args:
            messages: List of message dicts
            system: Optional system prompt
            max_tokens: Maximum tokens (ignored)
            temperature: Temperature (ignored)

        Returns:
            LLMResponse with mock content and token usage

        Raises:
            BudgetExceededError: If budget is exceeded
            IndexError: If no more responses are available
        """
        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Record the call
        self.call_history.append({
            "messages": messages,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

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
            input_tokens += len(system) // 4
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
