"""Google Gemini API provider (direct)."""

import os

from .openai_compatible import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    """
    Google Gemini API provider using the OpenAI-compatible endpoint.

    Environment variables:
    - GEMINI_API_KEY or GOOGLE_API_KEY (required)
    - GEMINI_BASE_URL or GOOGLE_GEMINI_BASE_URL (optional override)
    """

    def _get_api_key(self) -> str:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set. "
                "Get your API key from Google AI Studio."
            )
        return api_key

    def _get_base_url(self) -> str | None:
        return (
            os.environ.get("GEMINI_BASE_URL")
            or os.environ.get("GOOGLE_GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
