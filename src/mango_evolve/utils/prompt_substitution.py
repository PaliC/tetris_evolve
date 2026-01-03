"""
Prompt substitution utilities for mango_evolve.

Handles substitution of trial code tokens in prompts before sending to child LLMs.
This allows the Root LLM to reference code from previous trials using tokens like
{{CODE_TRIAL_5}} which get replaced with the actual code from that trial.

Supports two formats:
- New sequential format: {{CODE_TRIAL_N}} where N is the trial number (e.g., trial_5)
- Legacy format: {{CODE_TRIAL_X_Y}} where X is generation, Y is trial number (e.g., trial_0_3)
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..evolution_api import TrialResult

# Pattern to match {{CODE_TRIAL_N}} tokens (new sequential format)
TRIAL_CODE_PATTERN_SEQUENTIAL = re.compile(r"\{\{CODE_TRIAL_(\d+)\}\}")

# Pattern to match {{CODE_TRIAL_X_Y}} tokens (legacy format)
# X = generation number (0+), Y = trial number within generation (0+)
TRIAL_CODE_PATTERN_LEGACY = re.compile(r"\{\{CODE_TRIAL_(\d+)_(\d+)\}\}")


def find_trial_code_tokens(prompt: str) -> list[tuple[str, str]]:
    """
    Find all trial code tokens in a prompt.

    Supports both formats:
    - Sequential: {{CODE_TRIAL_5}} -> trial_5
    - Legacy: {{CODE_TRIAL_0_3}} -> trial_0_3

    Args:
        prompt: The prompt string to search

    Returns:
        List of tuples: (full_token, trial_id)
        e.g., [("{{CODE_TRIAL_5}}", "trial_5"), ("{{CODE_TRIAL_0_3}}", "trial_0_3")]
    """
    tokens = []

    # Find legacy format tokens first (more specific pattern)
    for match in TRIAL_CODE_PATTERN_LEGACY.finditer(prompt):
        full_token = match.group(0)
        generation = int(match.group(1))
        trial_num = int(match.group(2))
        trial_id = f"trial_{generation}_{trial_num}"
        tokens.append((full_token, trial_id))

    # Find sequential format tokens
    for match in TRIAL_CODE_PATTERN_SEQUENTIAL.finditer(prompt):
        full_token = match.group(0)
        # Skip if this is part of a legacy token (already matched)
        if any(full_token in legacy_token for legacy_token, _ in tokens):
            continue
        trial_num = int(match.group(1))
        trial_id = f"trial_{trial_num}"
        tokens.append((full_token, trial_id))

    return tokens


def load_trial_code_from_disk(
    trial_id: str,
    experiment_dir: str | Path,
) -> str | None:
    """
    Load trial code from disk.

    Supports both formats:
    - Sequential: trial_5
    - Legacy: trial_0_3

    Args:
        trial_id: Trial identifier (e.g., "trial_5" or "trial_0_3")
        experiment_dir: Path to experiment directory

    Returns:
        The code string if found, None otherwise
    """
    experiment_path = Path(experiment_dir)

    # Try legacy format first (trial_X_Y)
    legacy_match = re.match(r"trial_(\d+)_(\d+)", trial_id)
    if legacy_match:
        generation = int(legacy_match.group(1))
        trial_path = experiment_path / "generations" / f"gen_{generation}" / f"{trial_id}.json"

        if trial_path.exists():
            try:
                with open(trial_path) as f:
                    trial_data = json.load(f)
                return trial_data.get("code")
            except (json.JSONDecodeError, OSError):
                pass

    # Try sequential format (trial_N) - search all generation folders
    sequential_match = re.match(r"trial_(\d+)$", trial_id)
    if sequential_match:
        generations_dir = experiment_path / "generations"
        if generations_dir.exists():
            for gen_dir in sorted(generations_dir.iterdir()):
                if gen_dir.is_dir() and gen_dir.name.startswith("gen_"):
                    trial_path = gen_dir / f"{trial_id}.json"
                    if trial_path.exists():
                        try:
                            with open(trial_path) as f:
                                trial_data = json.load(f)
                            return trial_data.get("code")
                        except (json.JSONDecodeError, OSError):
                            pass

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
    Substitute all trial code tokens in a prompt with actual code.

    Supports both formats:
    - Sequential: {{CODE_TRIAL_5}} -> trial_5
    - Legacy: {{CODE_TRIAL_0_3}} -> trial_0_3

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

    for token, trial_id in tokens:
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
