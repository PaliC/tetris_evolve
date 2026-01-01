"""OpenRouter API provider."""

import os
from typing import TYPE_CHECKING, Any

from .openai_compatible import OpenAICompatibleProvider

if TYPE_CHECKING:
    from ...cost_tracker import CostTracker


class OpenRouterProvider(OpenAICompatibleProvider):
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
        self._reasoning_config = reasoning_config
        super().__init__(model, cost_tracker, llm_type, max_retries)

    def _get_api_key(self) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your API key from https://openrouter.ai/keys"
            )
        return api_key

    def _get_base_url(self) -> str | None:
        return "https://openrouter.ai/api/v1"

    def _extract_reasoning(self, message: Any, response: Any) -> tuple[str | None, int]:
        reasoning_content = None
        if hasattr(message, "reasoning") and message.reasoning:
            reasoning_content = message.reasoning

        reasoning_tokens = 0
        if response.usage and hasattr(response.usage, "reasoning_tokens"):
            reasoning_tokens = response.usage.reasoning_tokens or 0

        return reasoning_content, reasoning_tokens

    def _build_reasoning_param(self) -> dict[str, Any] | None:
        """Build the reasoning parameter by filtering out None/False values."""
        if not self._reasoning_config:
            return None

        reasoning = {
            k: v for k, v in self._reasoning_config.items()
            if v is not None and v is not False and v != "none"
        }
        return reasoning or None

    def _build_extra_body(self) -> dict[str, Any] | None:
        reasoning = self._build_reasoning_param()
        if not reasoning:
            return None
        return {"reasoning": reasoning}
