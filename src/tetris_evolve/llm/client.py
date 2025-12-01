"""LLM client wrapper for Anthropic Claude API."""

import os
import time
from dataclasses import dataclass
from typing import Optional

import anthropic


@dataclass
class LLMResponse:
    """Structured response from LLM API call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class LLMAuthenticationError(LLMClientError):
    """Authentication failed."""

    pass


class LLMRateLimitError(LLMClientError):
    """Rate limit exceeded."""

    pass


class LLMAPIError(LLMClientError):
    """General API error."""

    pass


class LLMClient:
    """Wrapper for Anthropic Claude API with retry logic."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """Initialize LLM client.

        Args:
            model: Model name to use.
            api_key: API key. Falls back to ANTHROPIC_API_KEY env var.
            max_retries: Maximum retry attempts for transient errors.
            base_delay: Base delay in seconds for exponential backoff.

        Raises:
            LLMAuthenticationError: If no API key provided or found.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay

        if not self.api_key:
            raise LLMAuthenticationError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def send_message(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send message to LLM and return structured response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: Optional system prompt.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with content and token counts.

        Raises:
            LLMAuthenticationError: If authentication fails.
            LLMRateLimitError: If rate limit exceeded after retries.
            LLMAPIError: For other API errors.
        """
        return self._retry_with_backoff(
            self._send_message_impl,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _send_message_impl(
        self,
        messages: list[dict],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Implementation of send_message without retry logic."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)

            # Extract text content from response
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            return LLMResponse(
                content=content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                stop_reason=response.stop_reason,
            )

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
        except anthropic.APIError as e:
            raise LLMAPIError(f"API error: {e}") from e

    def send_single(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simplified interface for single prompt → response.

        Args:
            prompt: User prompt text.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Response content as string.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.send_message(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic of ~4 characters per token.
        For more accurate counts, use Anthropic's token counting API.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Simple heuristic: ~4 chars per token on average
        return len(text) // 4

    def estimate_cost(
        self,
        input_text: str,
        estimated_output_tokens: int,
        cost_config: Optional[dict] = None,
    ) -> float:
        """Estimate cost for a potential API call.

        Args:
            input_text: Input text to send.
            estimated_output_tokens: Expected output token count.
            cost_config: Optional cost config dict.

        Returns:
            Estimated cost in USD.
        """
        if cost_config is None:
            cost_config = {
                "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
                "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
            }

        # Get config for current model or use a default
        model_config = cost_config.get(
            self.model,
            {"input": 0.003, "output": 0.015},  # Default to Sonnet pricing
        )

        input_tokens = self.count_tokens(input_text)
        input_cost = (input_tokens / 1000) * model_config["input"]
        output_cost = (estimated_output_tokens / 1000) * model_config["output"]

        return input_cost + output_cost

    def _retry_with_backoff(self, func, **kwargs):
        """Execute function with exponential backoff retry.

        Args:
            func: Function to execute.
            **kwargs: Arguments to pass to function.

        Returns:
            Function result.

        Raises:
            The last exception if all retries fail.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(**kwargs)
            except LLMRateLimitError as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt)
                    time.sleep(delay)
            except LLMAPIError as e:
                # Retry on server errors (5xx)
                if "500" in str(e) or "502" in str(e) or "503" in str(e):
                    last_exception = e
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2**attempt)
                        time.sleep(delay)
                else:
                    raise
            except LLMAuthenticationError:
                # Don't retry auth errors
                raise

        raise last_exception
