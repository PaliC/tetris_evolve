"""
Parallel worker for child LLM calls.

This module contains the worker function that runs in a separate process
to make LLM calls and evaluate the results.

Supports multiple providers (Anthropic, OpenRouter, OpenAI, and Gemini).
"""

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from .evaluation.circle_packing import CirclePackingEvaluator
from .utils.code_extraction import extract_python_code, extract_reasoning

_OPENAI_COMPATIBLE_PROVIDERS = {"openrouter", "openai", "gemini"}


@dataclass
class WorkerResult:
    """Result from a parallel worker."""

    prompt: str
    parent_id: str | None
    response_text: str
    code: str
    reasoning: str
    metrics: dict[str, Any]
    success: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    call_id: str
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _write_trial_file(
    trial_id: str,
    generation: int,
    experiment_dir: str,
    code: str | None,
    metrics: dict[str, Any],
    prompt: str,
    response: str,
    reasoning: str | None,
    parent_id: str | None,
    model_config: dict[str, Any] | None = None,
) -> None:
    """Write trial JSON file to disk for real-time progress tracking."""
    gen_dir = Path(experiment_dir) / "generations" / f"gen_{generation}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    trial_data = {
        "trial_id": trial_id,
        "generation": generation,
        "parent_id": parent_id,
        "code": code,
        "metrics": metrics,
        "prompt": prompt,
        "response": response,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
        "cost_data": None,
        "model_config": model_config,
    }

    trial_path = gen_dir / f"{trial_id}.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f, indent=2)


def _make_anthropic_call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    max_retries: int = 3,
) -> anthropic.types.Message:
    """Make an Anthropic API call with retry logic and optional caching.

    Args:
        client: Anthropic client instance
        model: Model name
        prompt: User prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt (will be cached if provided)
        max_retries: Number of retries for transient errors

    Returns:
        API response message
    """

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
            )
        ),
        reraise=True,
    )
    def _call():
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add system prompt with cache_control if provided
        if system_prompt:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return client.messages.create(**kwargs)

    return _call()


def _is_empty_chat_response(response) -> bool:
    """Check if chat response content is empty (triggers retry)."""
    if not response.choices:
        return True
    content = response.choices[0].message.content
    return not content or not content.strip()


def _make_openai_compatible_call_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    max_retries: int = 3,
):
    """Make an OpenAI-compatible API call with retry logic.

    Args:
        client: OpenAI client configured for the provider
        model: Model name
        prompt: User prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt
        max_retries: Number of retries for transient errors

    Returns:
        API response
    """

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_exception_type((OpenAIRateLimitError, OpenAIAPIConnectionError))
            | retry_if_result(_is_empty_chat_response)
        ),
        reraise=True,
    )
    def _call():
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return _call()


def _normalize_provider(provider: str) -> str:
    """Normalize provider aliases to canonical values."""
    provider = provider.lower()
    return "gemini" if provider == "google" else provider


def _get_openai_compatible_credentials(provider: str) -> tuple[str, str | None]:
    """Resolve API key and base URL for OpenAI-compatible providers."""
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return api_key, "https://openrouter.ai/api/v1"

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key, os.environ.get("OPENAI_BASE_URL")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

    base_url = (
        os.environ.get("GEMINI_BASE_URL")
        or os.environ.get("GOOGLE_GEMINI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    return api_key, base_url


def child_worker(args: tuple) -> dict[str, Any]:
    """
    Worker function for parallel child LLM calls.

    This function runs in a separate process and:
    1. Creates an LLM client (Anthropic, OpenRouter, OpenAI, or Gemini based on provider)
    2. Makes the LLM call with optional cached system prompt
    3. Extracts code from the response
    4. Evaluates the code
    5. Writes trial JSON file for real-time progress tracking
    6. Returns all results including cache statistics

    Args:
        args: Tuple of (prompt, parent_id, model, evaluator_kwargs, max_tokens, temperature,
                        trial_id, generation, experiment_dir, system_prompt, provider, model_alias)
              system_prompt, provider, and model_alias are optional for backwards compatibility

    Returns:
        Dictionary with all results needed to record the trial
    """
    # Handle different arg formats for backwards compatibility
    # 9 args: original format (no system_prompt, no provider, no model_alias)
    # 10 args: with system_prompt (no provider, no model_alias)
    # 11 args: with system_prompt and provider (no model_alias)
    # 12 args: with system_prompt, provider, and model_alias
    model_alias: str | None = None

    if len(args) == 9:
        (
            prompt,
            parent_id,
            model,
            evaluator_kwargs,
            max_tokens,
            temperature,
            trial_id,
            generation,
            experiment_dir,
        ) = args
        system_prompt = None
        provider = "anthropic"
    elif len(args) == 10:
        (
            prompt,
            parent_id,
            model,
            evaluator_kwargs,
            max_tokens,
            temperature,
            trial_id,
            generation,
            experiment_dir,
            system_prompt,
        ) = args
        provider = "anthropic"
    elif len(args) == 11:
        (
            prompt,
            parent_id,
            model,
            evaluator_kwargs,
            max_tokens,
            temperature,
            trial_id,
            generation,
            experiment_dir,
            system_prompt,
            provider,
        ) = args
    else:
        # 12 args: full new format with model_alias
        (
            prompt,
            parent_id,
            model,
            evaluator_kwargs,
            max_tokens,
            temperature,
            trial_id,
            generation,
            experiment_dir,
            system_prompt,
            provider,
            model_alias,
        ) = args

    provider = _normalize_provider(provider)

    # Build model config for trial metadata
    model_config = {
        "model": model,
        "temperature": temperature,
    }

    call_id = str(uuid.uuid4())
    response_text = ""
    code = ""
    reasoning = ""
    metrics: dict[str, Any] = {}
    success = False
    error: str | None = None
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0

    try:
        # Make the LLM call based on provider
        if provider in _OPENAI_COMPATIBLE_PROVIDERS:
            # OpenAI-compatible providers
            api_key, base_url = _get_openai_compatible_credentials(provider)
            if base_url:
                client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
            else:
                client = OpenAI(api_key=api_key)

            response = _make_openai_compatible_call_with_retry(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

            # Extract content and token counts (OpenAI format)
            response_text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            # OpenAI-compatible providers don't support Anthropic caching
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0
        else:
            # Anthropic provider (default)
            anthropic_client = anthropic.Anthropic()

            response = _make_anthropic_call_with_retry(
                client=anthropic_client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

            # Extract content and token counts (Anthropic format)
            if response.content:
                response_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Extract cache statistics
            cache_creation_input_tokens = getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0
            cache_read_input_tokens = getattr(
                response.usage, "cache_read_input_tokens", 0
            ) or 0

        # Extract code from response
        code = extract_python_code(response_text)
        reasoning = extract_reasoning(response_text)

        if not code:
            error = "No Python code block found in response"
            # Write trial file for tracking even on failure
            _write_trial_file(
                trial_id=trial_id,
                generation=generation,
                experiment_dir=experiment_dir,
                code="",
                metrics={},
                prompt=prompt,
                response=response_text,
                reasoning=reasoning,
                parent_id=parent_id,
                model_config=model_config,
            )
            return {
                "trial_id": trial_id,
                "prompt": prompt,
                "parent_id": parent_id,
                "response_text": response_text,
                "code": "",
                "reasoning": reasoning,
                "metrics": {},
                "success": False,
                "error": error,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "call_id": call_id,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
                "model_alias": model_alias,
                "model_config": model_config,
            }

        # Create evaluator and evaluate the code
        evaluator = CirclePackingEvaluator(**evaluator_kwargs)
        try:
            metrics = evaluator.evaluate(code)
        except Exception as e:
            metrics = {
                "valid": False,
                "error": f"Evaluation error: {str(e)}",
            }

        # Determine success
        success = bool(metrics.get("valid", False))
        if not success:
            error_value = metrics.get("error")
            error = str(error_value) if error_value is not None else None

    except Exception as e:
        error = f"LLM call failed: {str(e)}"

    # Write trial file for real-time progress tracking
    _write_trial_file(
        trial_id=trial_id,
        generation=generation,
        experiment_dir=experiment_dir,
        code=code,
        metrics=metrics,
        prompt=prompt,
        response=response_text,
        reasoning=reasoning,
        parent_id=parent_id,
        model_config=model_config,
    )

    return {
        "trial_id": trial_id,
        "prompt": prompt,
        "parent_id": parent_id,
        "response_text": response_text,
        "code": code,
        "reasoning": reasoning,
        "metrics": metrics,
        "success": success,
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "call_id": call_id,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "model_alias": model_alias,
        "model_config": model_config,
    }
