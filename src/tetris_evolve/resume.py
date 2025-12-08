"""
Experiment resume functionality for tetris_evolve.

Provides state detection and restoration for resuming interrupted experiments.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config, config_from_dict
from .evolution_api import TrialResult


@dataclass
class ResumeInfo:
    """Information about experiment state for resumption."""

    config: Config
    current_generation: int
    trials_in_current_gen: int
    max_children_per_gen: int

    @property
    def can_resume(self) -> bool:
        """Check if we can resume the experiment."""
        return self.trials_in_current_gen > 0 or self.current_generation > 0


def analyze_experiment(experiment_dir: Path | str) -> ResumeInfo:
    """
    Analyze an experiment directory and return resumption info.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        ResumeInfo with experiment state

    Raises:
        FileNotFoundError: If experiment directory or config doesn't exist
    """
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Load config
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)
    config = config_from_dict(config_data)

    # Scan generations directory
    generations_dir = experiment_dir / "generations"
    completed_generations: list[int] = []
    generation_trials: dict[int, int] = {}

    if generations_dir.exists():
        for gen_dir in sorted(generations_dir.iterdir()):
            if not gen_dir.is_dir() or not gen_dir.name.startswith("gen_"):
                continue

            gen_num = int(gen_dir.name.split("_")[1])

            # Check for summary.json (indicates generation is complete)
            if (gen_dir / "summary.json").exists():
                completed_generations.append(gen_num)

            # Count trial files
            trial_count = len(list(gen_dir.glob("trial_*.json")))
            generation_trials[gen_num] = trial_count

    # Determine current generation
    if not generation_trials:
        current_generation = 0
        trials_in_current_gen = 0
    else:
        max_gen_with_trials = max(generation_trials.keys())

        if max_gen_with_trials in completed_generations:
            # Last generation is complete - we're at the next one
            current_generation = max_gen_with_trials + 1
            trials_in_current_gen = generation_trials.get(current_generation, 0)
        else:
            current_generation = max_gen_with_trials
            trials_in_current_gen = generation_trials[current_generation]

    return ResumeInfo(
        config=config,
        current_generation=current_generation,
        trials_in_current_gen=trials_in_current_gen,
        max_children_per_gen=config.evolution.max_children_per_generation,
    )


def load_trials_from_disk(experiment_dir: Path) -> dict[str, TrialResult]:
    """
    Load all trial results from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary mapping trial_id to TrialResult
    """
    all_trials: dict[str, TrialResult] = {}
    generations_dir = experiment_dir / "generations"

    if not generations_dir.exists():
        return all_trials

    for gen_dir in sorted(generations_dir.iterdir()):
        if not gen_dir.is_dir() or not gen_dir.name.startswith("gen_"):
            continue

        for trial_file in gen_dir.glob("trial_*.json"):
            with open(trial_file) as f:
                trial_data = json.load(f)

            metrics = trial_data.get("metrics", {})
            success = bool(metrics.get("valid", False))

            trial = TrialResult(
                trial_id=trial_data["trial_id"],
                code=trial_data.get("code", ""),
                metrics=metrics,
                prompt=trial_data.get("prompt", ""),
                response=trial_data.get("response", ""),
                reasoning=trial_data.get("reasoning", ""),
                success=success,
                parent_id=trial_data.get("parent_id"),
                error=metrics.get("error") if not success else None,
                generation=trial_data.get("generation", 0),
            )
            all_trials[trial.trial_id] = trial

    return all_trials


def prepare_redo(experiment_dir: Path, current_generation: int) -> None:
    """
    Prepare experiment for redo by removing current generation directory.

    Args:
        experiment_dir: Path to experiment directory
        current_generation: The generation to clear
    """
    generations_dir = experiment_dir / "generations"
    current_gen_dir = generations_dir / f"gen_{current_generation}"

    if current_gen_dir.exists():
        shutil.rmtree(current_gen_dir)
