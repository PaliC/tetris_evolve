"""
Prompt substitution utilities for mango_evolve.

Handles substitution of trial code tokens in prompts before sending to child LLMs.
This allows the Root LLM to reference code from previous trials using tokens like
{{CODE_TRIAL_0_3}} which get replaced with the actual code from that trial.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..evolution_api import TrialResult

# Pattern to match {{CODE_TRIAL_X_Y}} tokens
# X = generation number (0+), Y = trial number within generation (0+)
TRIAL_CODE_PATTERN = re.compile(r"\{\{CODE_TRIAL_(\d+)_(\d+)\}\}")


def find_trial_code_tokens(prompt: str) -> list[tuple[str, int, int]]:
    """
    Find all trial code tokens in a prompt.

    Args:
        prompt: The prompt string to search

    Returns:
        List of tuples: (full_token, generation, trial_num)
        e.g., [("{{CODE_TRIAL_0_3}}", 0, 3), ("{{CODE_TRIAL_1_2}}", 1, 2)]
    """
    tokens = []
    for match in TRIAL_CODE_PATTERN.finditer(prompt):
        full_token = match.group(0)
        generation = int(match.group(1))
        trial_num = int(match.group(2))
        tokens.append((full_token, generation, trial_num))
    return tokens


def load_trial_code_from_disk(
    trial_id: str,
    experiment_dir: str | Path,
) -> str | None:
    """
    Load trial code from disk.

    Args:
        trial_id: Trial identifier (e.g., "trial_0_3")
        experiment_dir: Path to experiment directory

    Returns:
        The code string if found, None otherwise
    """
    # Parse generation from trial_id
    match = re.match(r"trial_(\d+)_(\d+)", trial_id)
    if not match:
        return None

    generation = int(match.group(1))

    trial_path = Path(experiment_dir) / "generations" / f"gen_{generation}" / f"{trial_id}.json"

    if not trial_path.exists():
        return None

    try:
        with open(trial_path) as f:
            trial_data = json.load(f)
        return trial_data.get("code")
    except (json.JSONDecodeError, OSError):
        return None


def _get_trial_code_internal(
    trial_id: str,
    all_trials: dict[str, "TrialResult"] | None = None,
    experiment_dir: str | Path | None = None,
) -> str | None:
    """
    Get trial code from memory or disk (internal helper).

    First tries to get code from in-memory trial cache, then falls back to disk.
    """
    # Try in-memory first
    if all_trials and trial_id in all_trials:
        trial = all_trials[trial_id]
        if trial.code:
            return trial.code

    # Fall back to disk
    if experiment_dir:
        return load_trial_code_from_disk(trial_id, experiment_dir)

    return None


def substitute_trial_codes(
    prompt: str,
    all_trials: dict[str, "TrialResult"] | None = None,
    experiment_dir: str | Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Substitute all {{CODE_TRIAL_X_Y}} tokens in a prompt with actual code.

    Args:
        prompt: The prompt string with potential tokens
        all_trials: Dictionary mapping trial_id to TrialResult (optional)
        experiment_dir: Path to experiment directory for disk fallback (optional)

    Returns:
        Tuple of (substituted_prompt, substitution_report)
        substitution_report is a list of dicts with:
            - token: The original token
            - trial_id: The trial ID
            - success: Whether substitution succeeded
            - error: Error message if failed
    """
    tokens = find_trial_code_tokens(prompt)

    if not tokens:
        return prompt, []

    substitution_report: list[dict[str, Any]] = []
    result = prompt

    for token, generation, trial_num in tokens:
        trial_id = f"trial_{generation}_{trial_num}"
        code = _get_trial_code_internal(trial_id, all_trials, experiment_dir)

        if code:
            result = result.replace(token, code)
            substitution_report.append(
                {
                    "token": token,
                    "trial_id": trial_id,
                    "success": True,
                    "error": None,
                }
            )
        else:
            # Leave token as-is if code not found (so it's visible in the prompt)
            # This helps with debugging
            error_marker = f"[CODE NOT FOUND: {trial_id}]"
            result = result.replace(token, error_marker)
            substitution_report.append(
                {
                    "token": token,
                    "trial_id": trial_id,
                    "success": False,
                    "error": f"Trial {trial_id} code not found in memory or on disk",
                }
            )

    return result, substitution_report


def substitute_trial_codes_batch(
    prompts: list[str],
    all_trials: dict[str, "TrialResult"] | None = None,
    experiment_dir: str | Path | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """
    Substitute trial codes in multiple prompts.

    Args:
        prompts: List of prompt strings
        all_trials: Dictionary mapping trial_id to TrialResult (optional)
        experiment_dir: Path to experiment directory for disk fallback (optional)

    Returns:
        List of (substituted_prompt, substitution_report) tuples
    """
    return [substitute_trial_codes(prompt, all_trials, experiment_dir) for prompt in prompts]
