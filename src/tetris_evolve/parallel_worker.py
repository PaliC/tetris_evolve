"""
Parallel worker for child LLM calls.

This module contains the worker function that runs in a separate process
to make LLM calls and evaluate the results.

Note: This module uses Anthropic directly (not via LLMClient) because:
1. Workers run in separate processes via multiprocessing
2. LLMClient instances can't be pickled across processes
3. Each worker needs its own API client

The provider-specific caching logic is implemented here to match the
provider-agnostic abstraction in the main LLM client.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .evaluation.circle_packing import CirclePackingEvaluator
from .utils.code_extraction import extract_python_code, extract_reasoning

# Supported providers for parallel workers
SUPPORTED_WORKER_PROVIDERS = {"anthropic"}


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
    }

    trial_path = gen_dir / f"{trial_id}.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f, indent=2)


def _make_llm_call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    max_retries: int = 3,
) -> anthropic.types.Message:
    """Make an LLM API call with retry logic and optional caching.

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


def child_worker(args: tuple) -> dict[str, Any]:
    """
    Worker function for parallel child LLM calls.

    This function runs in a separate process and:
    1. Creates its own API client for the specified provider
    2. Makes the LLM call with provider-appropriate caching
    3. Extracts code from the response
    4. Evaluates the code
    5. Writes trial JSON file for real-time progress tracking
    6. Returns all results including cache statistics

    Args:
        args: Tuple of (prompt, parent_id, model, evaluator_kwargs, max_tokens, temperature,
                        trial_id, generation, experiment_dir, system_prompt, provider)
              - system_prompt is optional and defaults to None for backwards compatibility
              - provider is optional and defaults to "anthropic" for backwards compatibility

    Returns:
        Dictionary with all results needed to record the trial
    """
    # Handle various arg formats for backwards compatibility
    # Old format (9 args): no system_prompt, no provider
    # Medium format (10 args): with system_prompt, no provider
    # New format (11 args): with system_prompt and provider
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
        ) = args

    # Validate provider
    if provider not in SUPPORTED_WORKER_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_WORKER_PROVIDERS))
        raise ValueError(
            f"Unsupported provider for parallel worker: {provider}. "
            f"Supported providers: {supported}"
        )

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
        # Create a new Anthropic client for this process
        client = anthropic.Anthropic()

        # Make the LLM call with optional system prompt caching
        response = _make_llm_call_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        # Extract content and token counts
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
    }
