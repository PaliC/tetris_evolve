"""OpenAI API provider."""

import os
from typing import TYPE_CHECKING, Any

from .openai_compatible import OpenAICompatibleProvider

if TYPE_CHECKING:
    from ...cost_tracker import CostTracker

class OpenAIProvider(OpenAICompatibleProvider):
    """
    OpenAI API provider.

    Environment variable: OPENAI_API_KEY
    Optional: OPENAI_BASE_URL for OpenAI-compatible endpoints.
    Reasoning config maps to Chat Completions `reasoning_effort`.
    """

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
        llm_type: str,
        max_retries: int = 3,
        reasoning_config: dict[str, Any] | None = None,
    ):
        self._reasoning_config = reasoning_config
        super().__init__(model, cost_tracker, llm_type, max_retries)

    def _get_api_key(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your API key from https://platform.openai.com/"
            )
        return api_key

    def _get_base_url(self) -> str | None:
        return os.environ.get("OPENAI_BASE_URL")

    def _build_request_overrides(self) -> dict[str, Any] | None:
        if not self._reasoning_config:
            return None

        effort = self._reasoning_config.get("effort")
        enabled = self._reasoning_config.get("enabled", False)

        if effort is None and not enabled:
            return None

        if effort is None and enabled:
            effort = "medium"

        return {"reasoning_effort": effort}
