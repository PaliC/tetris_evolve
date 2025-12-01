"""Experiment tracker for managing directory structure and persisting trial data."""

import json
import yaml
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TrialData:
    """Data for a single trial (code generation attempt)."""

    trial_id: str
    generation: int
    parent_id: Optional[str]
    code: str
    prompt: str
    reasoning: str
    llm_response: str
    metrics: Optional[dict]
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "trial_id": self.trial_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "code": self.code,
            "prompt": self.prompt,
            "reasoning": self.reasoning,
            "llm_response": self.llm_response,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrialData":
        """Create from dict."""
        return cls(
            trial_id=data["trial_id"],
            generation=data["generation"],
            parent_id=data.get("parent_id"),
            code=data["code"],
            prompt=data["prompt"],
            reasoning=data["reasoning"],
            llm_response=data["llm_response"],
            metrics=data.get("metrics"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class GenerationStats:
    """Statistics for a completed generation."""

    generation: int
    num_trials: int
    best_score: float
    avg_score: float
    best_trial_id: str
    selected_parent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "generation": self.generation,
            "num_trials": self.num_trials,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
            "best_trial_id": self.best_trial_id,
            "selected_parent_ids": self.selected_parent_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationStats":
        """Create from dict."""
        return cls(
            generation=data["generation"],
            num_trials=data["num_trials"],
            best_score=data["best_score"],
            avg_score=data["avg_score"],
            best_trial_id=data["best_trial_id"],
            selected_parent_ids=data.get("selected_parent_ids", []),
        )


class ExperimentTracker:
    """Manages experiment directory structure and persists trial/generation data."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize experiment tracker.

        Args:
            base_dir: Base directory for experiments.
        """
        self.base_dir = Path(base_dir)
        self.experiment_dir: Optional[Path] = None
        self.experiment_id: Optional[str] = None
        self.current_generation: int = 0
        self._trial_counter: int = 0

    def create_experiment(self, config: dict) -> str:
        """Create a new experiment directory with config.

        Args:
            config: Experiment configuration dict.

        Returns:
            Experiment ID (timestamped string).
        """
        # Generate timestamped experiment ID
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.experiment_id = f"exp_{timestamp}"

        # Create experiment directory
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save frozen config
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create generations directory
        (self.experiment_dir / "generations").mkdir(exist_ok=True)

        # Initialize experiment stats
        self.update_experiment_stats({
            "experiment_id": self.experiment_id,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "total_trials": 0,
            "total_generations": 0,
            "best_score": None,
            "best_trial_id": None,
        })

        return self.experiment_id

    def start_generation(self, generation: int) -> Path:
        """Start a new generation.

        Args:
            generation: Generation number (1-indexed).

        Returns:
            Path to generation directory.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created. Call create_experiment first.")

        self.current_generation = generation

        # Create generation directory
        gen_dir = self.experiment_dir / "generations" / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Create trials subdirectory
        (gen_dir / "trials").mkdir(exist_ok=True)

        return gen_dir

    def complete_generation(
        self,
        generation: int,
        selected_ids: list[str],
        root_reasoning: str,
        stats: GenerationStats,
    ) -> None:
        """Complete a generation and save its data.

        Args:
            generation: Generation number.
            selected_ids: Trial IDs selected as parents for next generation.
            root_reasoning: Root LLM's reasoning for selections.
            stats: Generation statistics.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        gen_dir = self.experiment_dir / "generations" / f"gen_{generation:03d}"

        # Save generation stats
        stats_path = gen_dir / "generation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        # Save root LLM reasoning
        reasoning_path = gen_dir / "root_llm_reasoning.md"
        with open(reasoning_path, "w") as f:
            f.write(f"# Generation {generation} - Root LLM Reasoning\n\n")
            f.write(root_reasoning)

        # Save selected parents
        selected_path = gen_dir / "selected_parents.json"
        with open(selected_path, "w") as f:
            json.dump({"selected_trial_ids": selected_ids}, f, indent=2)

    def _generate_trial_id(self) -> str:
        """Generate a unique trial ID.

        Returns:
            Trial ID in format trial_NNN.
        """
        self._trial_counter += 1
        return f"trial_{self._trial_counter:03d}"

    def save_trial(self, trial: TrialData) -> Path:
        """Save trial data to disk.

        Args:
            trial: Trial data to save.

        Returns:
            Path to trial directory.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        # Get trial directory
        gen_dir = self.experiment_dir / "generations" / f"gen_{trial.generation:03d}"
        trial_dir = gen_dir / "trials" / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save code
        with open(trial_dir / "code.py", "w") as f:
            f.write(trial.code)

        # Save prompt
        with open(trial_dir / "prompt.txt", "w") as f:
            f.write(trial.prompt)

        # Save reasoning
        with open(trial_dir / "reasoning.md", "w") as f:
            f.write(f"# Trial {trial.trial_id} Reasoning\n\n")
            f.write(trial.reasoning)

        # Save LLM response
        with open(trial_dir / "llm_response.txt", "w") as f:
            f.write(trial.llm_response)

        # Save metrics if present
        if trial.metrics is not None:
            with open(trial_dir / "metrics.json", "w") as f:
                json.dump(trial.metrics, f, indent=2)

        # Save parent ID if present
        if trial.parent_id is not None:
            with open(trial_dir / "parent_id.txt", "w") as f:
                f.write(trial.parent_id)

        # Save trial metadata
        with open(trial_dir / "trial.json", "w") as f:
            json.dump(trial.to_dict(), f, indent=2)

        # Update trial counter to ensure uniqueness
        trial_num = int(trial.trial_id.split("_")[1])
        if trial_num >= self._trial_counter:
            self._trial_counter = trial_num

        return trial_dir

    def get_trial(self, trial_id: str) -> TrialData:
        """Load trial data from disk.

        Args:
            trial_id: Trial ID to load.

        Returns:
            TrialData for the trial.

        Raises:
            FileNotFoundError: If trial not found.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        # Search for trial in all generations
        gens_dir = self.experiment_dir / "generations"
        for gen_dir in sorted(gens_dir.glob("gen_*")):
            trial_dir = gen_dir / "trials" / trial_id
            if trial_dir.exists():
                return self._load_trial_from_dir(trial_dir)

        raise FileNotFoundError(f"Trial {trial_id} not found")

    def _load_trial_from_dir(self, trial_dir: Path) -> TrialData:
        """Load trial data from a directory.

        Args:
            trial_dir: Path to trial directory.

        Returns:
            TrialData for the trial.
        """
        # Try to load from trial.json first (fastest)
        trial_json = trial_dir / "trial.json"
        if trial_json.exists():
            with open(trial_json, "r") as f:
                return TrialData.from_dict(json.load(f))

        # Fall back to loading individual files
        trial_id = trial_dir.name

        # Extract generation from path
        gen_str = trial_dir.parent.parent.name  # e.g., "gen_001"
        generation = int(gen_str.split("_")[1])

        with open(trial_dir / "code.py", "r") as f:
            code = f.read()

        with open(trial_dir / "prompt.txt", "r") as f:
            prompt = f.read()

        with open(trial_dir / "reasoning.md", "r") as f:
            reasoning = f.read()
            # Strip header if present
            if reasoning.startswith("#"):
                lines = reasoning.split("\n", 2)
                reasoning = lines[2] if len(lines) > 2 else ""

        with open(trial_dir / "llm_response.txt", "r") as f:
            llm_response = f.read()

        # Load optional files
        metrics = None
        metrics_path = trial_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        parent_id = None
        parent_path = trial_dir / "parent_id.txt"
        if parent_path.exists():
            with open(parent_path, "r") as f:
                parent_id = f.read().strip()

        # Get timestamp from directory modification time
        timestamp = datetime.fromtimestamp(trial_dir.stat().st_mtime)

        return TrialData(
            trial_id=trial_id,
            generation=generation,
            parent_id=parent_id,
            code=code,
            prompt=prompt,
            reasoning=reasoning,
            llm_response=llm_response,
            metrics=metrics,
            timestamp=timestamp,
        )

    def get_generation_trials(self, generation: int) -> list[TrialData]:
        """Get all trials for a generation.

        Args:
            generation: Generation number.

        Returns:
            List of TrialData for the generation.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        gen_dir = self.experiment_dir / "generations" / f"gen_{generation:03d}"
        trials_dir = gen_dir / "trials"

        if not trials_dir.exists():
            return []

        trials = []
        for trial_dir in sorted(trials_dir.glob("trial_*")):
            try:
                trials.append(self._load_trial_from_dir(trial_dir))
            except Exception:
                continue

        return trials

    def get_all_trials(self) -> list[TrialData]:
        """Get all trials across all generations.

        Returns:
            List of all TrialData.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        all_trials = []
        gens_dir = self.experiment_dir / "generations"

        for gen_dir in sorted(gens_dir.glob("gen_*")):
            gen_num = int(gen_dir.name.split("_")[1])
            trials = self.get_generation_trials(gen_num)
            all_trials.extend(trials)

        return all_trials

    def get_best_trial(self) -> Optional[TrialData]:
        """Get the trial with the highest score.

        Returns:
            TrialData with highest score, or None if no trials.
        """
        all_trials = self.get_all_trials()

        if not all_trials:
            return None

        # Filter trials with metrics
        trials_with_scores = [
            t for t in all_trials
            if t.metrics is not None and t.metrics.get("avg_score") is not None
        ]

        if not trials_with_scores:
            return None

        return max(trials_with_scores, key=lambda t: t.metrics["avg_score"])

    def update_experiment_stats(self, stats: dict) -> None:
        """Update experiment-level statistics.

        Args:
            stats: Statistics dict to save.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        stats_path = self.experiment_dir / "experiment_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def get_experiment_stats(self) -> dict:
        """Get experiment-level statistics.

        Returns:
            Statistics dict.
        """
        if self.experiment_dir is None:
            raise RuntimeError("No experiment created.")

        stats_path = self.experiment_dir / "experiment_stats.json"
        if not stats_path.exists():
            return {}

        with open(stats_path, "r") as f:
            return json.load(f)

    @classmethod
    def load_experiment(cls, experiment_dir: Path) -> "ExperimentTracker":
        """Load an existing experiment for resumption.

        Args:
            experiment_dir: Path to experiment directory.

        Returns:
            ExperimentTracker configured for the experiment.

        Raises:
            FileNotFoundError: If experiment directory doesn't exist.
        """
        experiment_dir = Path(experiment_dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        # Create tracker
        tracker = cls(experiment_dir.parent)
        tracker.experiment_dir = experiment_dir
        tracker.experiment_id = experiment_dir.name

        # Find highest generation and trial numbers
        gens_dir = experiment_dir / "generations"
        max_gen = 0
        max_trial = 0

        if gens_dir.exists():
            for gen_dir in gens_dir.glob("gen_*"):
                gen_num = int(gen_dir.name.split("_")[1])
                max_gen = max(max_gen, gen_num)

                trials_dir = gen_dir / "trials"
                if trials_dir.exists():
                    for trial_dir in trials_dir.glob("trial_*"):
                        trial_num = int(trial_dir.name.split("_")[1])
                        max_trial = max(max_trial, trial_num)

        tracker.current_generation = max_gen
        tracker._trial_counter = max_trial

        return tracker
