"""
Parallel worker for child LLM calls.

This module contains the worker function that runs in a separate process
to make LLM calls and evaluate the results.
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
    max_retries: int = 3,
) -> anthropic.types.Message:
    """Make an LLM API call with retry logic."""

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
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

    return _call()


def child_worker(args: tuple) -> dict[str, Any]:
    """
    Worker function for parallel child LLM calls.

    This function runs in a separate process and:
    1. Creates its own Anthropic client
    2. Makes the LLM call
    3. Extracts code from the response
    4. Evaluates the code
    5. Writes trial JSON file for real-time progress tracking
    6. Returns all results

    Args:
        args: Tuple of (prompt, parent_id, model, evaluator_kwargs, max_tokens, temperature,
                        trial_id, generation, experiment_dir)

    Returns:
        Dictionary with all results needed to record the trial
    """
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

    call_id = str(uuid.uuid4())
    response_text = ""
    code = ""
    reasoning = ""
    metrics: dict[str, Any] = {}
    success = False
    error: str | None = None
    input_tokens = 0
    output_tokens = 0

    try:
        # Create a new Anthropic client for this process
        client = anthropic.Anthropic()

        # Make the LLM call
        response = _make_llm_call_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract content and token counts
        if response.content:
            response_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

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
    }
