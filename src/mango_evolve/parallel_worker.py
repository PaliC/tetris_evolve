"""
Parallel worker for child LLM calls.

This module contains the worker function that runs in a separate process
to make LLM calls and evaluate the results.

Supports multiple providers (Anthropic, OpenRouter, and Google).
"""

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

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


def _parse_worker_args(args: tuple) -> tuple:
    """Normalize worker args to the latest signature."""
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

    return (
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
    )


def query_llm(args: tuple) -> dict[str, Any]:
    """
    Submit an LLM query with no evaluation or file I/O.

    Uses the same inputs/outputs as spawn_child.
    """
    (
        prompt,
        parent_id,
        model,
        _evaluator_kwargs,
        max_tokens,
        temperature,
        trial_id,
        _generation,
        _experiment_dir,
        system_prompt,
        provider,
        model_alias,
    ) = _parse_worker_args(args)

    model_config = {
        "model": model,
        "temperature": temperature,
    }

    call_id = str(uuid.uuid4())
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0
    error: str | None = None
    provider_debug: dict[str, Any] | None = None

    try:
        if provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            response = _make_openrouter_call_with_retry(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

            response_text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
        elif provider == "google":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            response = _make_google_call_with_retry(
                client=client,
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            response_text = _extract_google_text(response)
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason is not None:
                    finish_reason = getattr(finish_reason, "name", str(finish_reason))

            usage_metadata: dict[str, Any] = {}
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0
                usage_metadata = {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "thoughts_token_count": getattr(
                        response.usage_metadata, "thoughts_token_count", 0
                    )
                    or 0,
                }

            provider_debug = {
                "finish_reason": finish_reason,
                "usage_metadata": usage_metadata,
                "response_text_len": len(response_text),
            }
        else:
            anthropic_client = anthropic.Anthropic()

            response = _make_anthropic_call_with_retry(
                client=anthropic_client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

            if response.content:
                response_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cache_creation_input_tokens = (
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            )
            cache_read_input_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    except Exception as e:
        error = f"LLM call failed: {str(e)}"

    success = error is None
    reasoning = response_text if success else ""
    if provider_debug:
        model_config["provider_debug"] = provider_debug

    return {
        "trial_id": trial_id,
        "prompt": prompt,
        "parent_id": parent_id,
        "response_text": response_text,
        "code": "",
        "reasoning": reasoning,
        "metrics": {},
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


def _is_empty_openrouter_response(response) -> bool:
    """Check if OpenRouter response content is empty (triggers retry)."""
    if not response.choices:
        return True
    content = response.choices[0].message.content
    return not content or not content.strip()


def _make_openrouter_call_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    max_retries: int = 3,
):
    """Make an OpenRouter API call with retry logic.

    Args:
        client: OpenAI client configured for OpenRouter
        model: Model name (e.g., "google/gemini-2.0-flash-001")
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
            | retry_if_result(_is_empty_openrouter_response)
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


def _is_empty_google_response(response: Any) -> bool:
    """Check if Google response content is empty (triggers retry)."""
    if not response.candidates:
        return True
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return True
    text = "".join(
        part.text or ""
        for part in candidate.content.parts
        if not getattr(part, "thought", False)
    )
    return not text.strip()


def _make_google_call_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
    max_retries: int = 3,
):
    """Make a Google Gemini API call with retry logic."""

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_exception_type((ConnectionError, TimeoutError))
            | retry_if_result(_is_empty_google_response)
        ),
        reraise=True,
    )
    def _call():
        return client.models.generate_content(
            model=model,
            contents=cast(Any, contents),
            config=config,
        )

    return _call()


def _extract_google_text(response: Any) -> str:
    """Extract non-thinking content from a Gemini response."""
    content = ""
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "thought", False):
                    continue
                content += part.text or ""
    return content


def spawn_child(args: tuple) -> dict[str, Any]:
    """
    Worker function for parallel child LLM calls.

    This function runs in a separate process and:
    1. Creates an LLM client (Anthropic or OpenRouter based on provider)
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
    (
        prompt,
        parent_id,
        _model,
        evaluator_kwargs,
        _max_tokens,
        _temperature,
        trial_id,
        generation,
        experiment_dir,
        _system_prompt,
        _provider,
        _model_alias,
    ) = _parse_worker_args(args)

    result = query_llm(args)

    # If LLM call failed, write trial file and return as-is
    if result["error"]:
        _write_trial_file(
            trial_id=trial_id,
            generation=generation,
            experiment_dir=experiment_dir,
            code="",
            metrics={},
            prompt=prompt,
            response=result["response_text"],
            reasoning=result["reasoning"],
            parent_id=parent_id,
            model_config=result.get("model_config"),
        )
        return result

    response_text = result["response_text"]
    code = extract_python_code(response_text)
    reasoning = extract_reasoning(response_text)

    if not code:
        result.update(
            {
                "code": "",
                "reasoning": reasoning,
                "metrics": {},
                "success": False,
                "error": "No Python code block found in response",
            }
        )
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
            model_config=result.get("model_config"),
        )
        return result

    evaluator = CirclePackingEvaluator(**evaluator_kwargs)
    try:
        metrics = evaluator.evaluate(code)
    except Exception as e:
        metrics = {
            "valid": False,
            "error": f"Evaluation error: {str(e)}",
        }

    success = bool(metrics.get("valid", False))
    error_value = metrics.get("error") if not success else None
    error = str(error_value) if error_value is not None else None

    result.update(
        {
            "code": code,
            "reasoning": reasoning,
            "metrics": metrics,
            "success": success,
            "error": error,
        }
    )

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
        model_config=result.get("model_config"),
    )

    return result


def child_worker(args: tuple) -> dict[str, Any]:
    """Backward compatible alias for spawn_child."""
    return spawn_child(args)
