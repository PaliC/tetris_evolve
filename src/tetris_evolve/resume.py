"""
Experiment resume functionality for tetris_evolve.

Provides state detection and restoration for resuming interrupted experiments.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, config_from_dict
from .evolution_api import GenerationSummary, TrialResult, TrialSelection


@dataclass
class ResumeInfo:
    """Information about experiment state for resumption."""

    experiment_dir: Path
    config: Config
    current_generation: int
    total_generations_configured: int
    trials_in_current_gen: int
    max_children_per_gen: int
    generation_complete: bool  # Has summary.json
    total_cost_spent: float
    max_budget: float
    all_trial_ids: list[str] = field(default_factory=list)
    completed_generation_nums: list[int] = field(default_factory=list)

    @property
    def can_resume(self) -> bool:
        """Check if we can resume (redo) the experiment."""
        # Can resume if there are trials in current gen (complete or not)
        # OR if we're at a generation that hasn't started yet but previous ones exist
        return self.trials_in_current_gen > 0 or self.current_generation > 0

    @property
    def remaining_budget(self) -> float:
        """Get remaining budget."""
        return self.max_budget - self.total_cost_spent

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Experiment: {self.experiment_dir.name}",
            f"Current generation: {self.current_generation}/{self.total_generations_configured}",
            f"Trials in current gen: {self.trials_in_current_gen}/{self.max_children_per_gen}",
            f"Generation complete: {self.generation_complete}",
            f"Total trials: {len(self.all_trial_ids)}",
            f"Completed generations: {self.completed_generation_nums}",
            f"Cost spent: ${self.total_cost_spent:.4f} / ${self.max_budget:.2f}",
            f"Can resume: {self.can_resume}",
        ]
        return "\n".join(lines)


def analyze_experiment(experiment_dir: Path | str) -> ResumeInfo:
    """
    Analyze an experiment directory and return resumption info.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        ResumeInfo with experiment state

    Raises:
        FileNotFoundError: If experiment directory or config doesn't exist
        ValueError: If experiment state is inconsistent
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

    # Load cost tracking if exists
    cost_path = experiment_dir / "cost_tracking.json"
    total_cost = 0.0
    if cost_path.exists():
        with open(cost_path) as f:
            cost_data = json.load(f)
        total_cost = cost_data.get("total_cost", 0.0)

    # Scan generations directory
    generations_dir = experiment_dir / "generations"
    all_trial_ids: list[str] = []
    completed_generations: list[int] = []
    generation_trials: dict[int, list[str]] = {}

    if generations_dir.exists():
        for gen_dir in sorted(generations_dir.iterdir()):
            if not gen_dir.is_dir() or not gen_dir.name.startswith("gen_"):
                continue

            gen_num = int(gen_dir.name.split("_")[1])
            generation_trials[gen_num] = []

            # Check for summary.json (indicates generation is complete)
            if (gen_dir / "summary.json").exists():
                completed_generations.append(gen_num)

            # Collect trial files
            for trial_file in gen_dir.glob("trial_*.json"):
                trial_id = trial_file.stem
                all_trial_ids.append(trial_id)
                generation_trials[gen_num].append(trial_id)

    # Determine current generation
    if not generation_trials:
        # No generations started
        current_generation = 0
        trials_in_current_gen = 0
        generation_complete = False
    else:
        # Find highest generation with trials
        max_gen_with_trials = max(generation_trials.keys())

        # Check if this generation is complete
        if max_gen_with_trials in completed_generations:
            # Last generation is complete - we're at the next one
            current_generation = max_gen_with_trials + 1
            trials_in_current_gen = len(generation_trials.get(current_generation, []))
            generation_complete = current_generation in completed_generations
        else:
            # Last generation is incomplete
            current_generation = max_gen_with_trials
            trials_in_current_gen = len(generation_trials[current_generation])
            generation_complete = False

    return ResumeInfo(
        experiment_dir=experiment_dir,
        config=config,
        current_generation=current_generation,
        total_generations_configured=config.evolution.max_generations,
        trials_in_current_gen=trials_in_current_gen,
        max_children_per_gen=config.evolution.max_children_per_generation,
        generation_complete=generation_complete,
        total_cost_spent=total_cost,
        max_budget=config.budget.max_total_cost,
        all_trial_ids=all_trial_ids,
        completed_generation_nums=completed_generations,
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

            # Determine success from metrics
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


def load_generation_summaries(experiment_dir: Path) -> list[GenerationSummary]:
    """
    Load generation summaries from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of GenerationSummary objects
    """
    generations: list[GenerationSummary] = []
    generations_dir = experiment_dir / "generations"
    all_trials = load_trials_from_disk(experiment_dir)

    if not generations_dir.exists():
        return generations

    # Get all generation directories
    gen_dirs = sorted(
        [d for d in generations_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    for gen_dir in gen_dirs:
        gen_num = int(gen_dir.name.split("_")[1])

        # Get trials for this generation
        gen_trials = [t for t in all_trials.values() if t.generation == gen_num]

        # Load summary if exists
        summary_path = gen_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary_data = json.load(f)

            # Load trial selections
            trial_selections = [
                TrialSelection.from_dict(s)
                for s in summary_data.get("trial_selections", [])
            ]

            gen_summary = GenerationSummary(
                generation_num=gen_num,
                trials=gen_trials,
                selected_trial_ids=summary_data.get("selected_trial_ids", []),
                selection_reasoning=summary_data.get("selection_reasoning", ""),
                best_trial_id=summary_data.get("best_trial_id"),
                best_score=summary_data.get("best_sum_radii", 0.0),
                trial_selections=trial_selections,
            )
        else:
            # No summary - generation is incomplete
            best_trial = max(
                (t for t in gen_trials if t.success),
                key=lambda t: t.metrics.get("sum_radii", 0),
                default=None,
            )
            gen_summary = GenerationSummary(
                generation_num=gen_num,
                trials=gen_trials,
                best_trial_id=best_trial.trial_id if best_trial else None,
                best_score=best_trial.metrics.get("sum_radii", 0) if best_trial else 0.0,
            )

        generations.append(gen_summary)

    return generations


def load_cost_data(experiment_dir: Path) -> dict[str, Any]:
    """
    Load cost tracking data from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Cost tracking dictionary or empty dict if not found
    """
    cost_path = experiment_dir / "cost_tracking.json"
    if cost_path.exists():
        with open(cost_path) as f:
            return json.load(f)
    return {}


def prepare_redo(experiment_dir: Path) -> None:
    """
    Prepare experiment for redo mode by removing current generation trials.

    Args:
        experiment_dir: Path to experiment directory
    """
    info = analyze_experiment(experiment_dir)

    if info.current_generation == 0 and info.trials_in_current_gen == 0:
        # Nothing to redo
        return

    generations_dir = experiment_dir / "generations"
    current_gen_dir = generations_dir / f"gen_{info.current_generation}"

    if current_gen_dir.exists():
        # Remove the current generation directory
        shutil.rmtree(current_gen_dir)

    # If cost tracking exists, we need to recalculate costs excluding current gen
    # For simplicity, we'll just keep the existing cost - the budget will be
    # slightly over-counted but that's safer than under-counting
    # A more sophisticated approach would parse the usage log and filter by trial


def build_resume_prompt(
    info: ResumeInfo,
    all_trials: dict[str, TrialResult],
) -> str:
    """
    Build the initial prompt for a resumed experiment.

    Args:
        info: ResumeInfo from analyze_experiment
        all_trials: All loaded trials

    Returns:
        Initial user message for the resumed conversation
    """
    lines = [
        "RESUMING EXPERIMENT",
        "",
        f"Generation {info.current_generation} is being restarted.",
        "",
    ]

    # Show previous generation results if any
    if info.current_generation > 0:
        prev_gen_trials = [
            t for t in all_trials.values()
            if t.generation == info.current_generation - 1
        ]
        if prev_gen_trials:
            lines.append(f"Previous generation {info.current_generation - 1} results:")
            sorted_trials = sorted(
                prev_gen_trials,
                key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
                reverse=True,
            )
            for trial in sorted_trials[:5]:  # Show top 5
                score = trial.metrics.get("sum_radii", 0) if trial.success else 0
                status = "valid" if trial.success else "INVALID"
                lines.append(f"  - {trial.trial_id}: score={score:.4f} [{status}]")
            lines.append("")

    lines.extend([
        f"Spawn up to {info.max_children_per_gen} children for generation {info.current_generation}.",
        "Use insights from previous generations to guide your strategy.",
    ])

    # Add overall best score context
    if all_trials:
        valid_trials = [t for t in all_trials.values() if t.success]
        if valid_trials:
            best = max(valid_trials, key=lambda t: t.metrics.get("sum_radii", 0))
            best_score = best.metrics.get("sum_radii", 0)
            lines.extend([
                "",
                f"Best score so far: {best_score:.4f} ({best.trial_id})",
            ])

    # Add budget info
    lines.extend([
        "",
        f"Budget remaining: ${info.remaining_budget:.2f}",
    ])

    return "\n".join(lines)
