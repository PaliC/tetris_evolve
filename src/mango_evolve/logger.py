"""
Experiment logging system for mango_evolve.

Handles structured logging for experiments, generations, and trials.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config


class ExperimentLogger:
    """
    Structured logger for evolution experiments.

    Creates a directory structure:
        {output_dir}/{experiment_name}/
        ├── experiment.json          # Full experiment state
        ├── config.json              # Configuration snapshot
        ├── root_llm_log.jsonl       # Root LLM conversation turns
        ├── cost_tracking.json       # Cost data
        └── generations/
            ├── gen_0/
            │   ├── trial_0_0.json
            │   └── trial_0_1.json
            └── gen_1/
                └── ...
    """

    def __init__(self, config: Config, run_id: str | None = None):
        """
        Initialize the experiment logger.

        Args:
            config: Experiment configuration
            run_id: Optional unique run identifier (defaults to timestamp)
        """
        self.config = config
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment directory
        self.base_dir = (
            Path(config.experiment.output_dir) / f"{config.experiment.name}_{self.run_id}"
        )
        self.generations_dir = self.base_dir / "generations"

        self._experiment_data: dict[str, Any] = {
            "experiment_id": f"{config.experiment.name}_{self.run_id}",
            "config": config.to_dict(),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "termination_reason": None,
            "num_generations": 0,
            "total_trials": 0,
            "best_trial": None,
            "scratchpad": "",
        }

        self._generation_data: list[dict[str, Any]] = []
        self._initialized = False

    def create_experiment_directory(self) -> Path:
        """
        Create the experiment directory structure.

        Returns:
            Path to the experiment directory
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(exist_ok=True)

        # Save initial config
        config_path = self.base_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self._initialized = True
        return self.base_dir

    def _ensure_initialized(self) -> None:
        """Ensure experiment directory is created."""
        if not self._initialized:
            self.create_experiment_directory()

    def _get_generation_dir(self, generation: int) -> Path:
        """Get or create directory for a generation."""
        gen_dir = self.generations_dir / f"gen_{generation}"
        gen_dir.mkdir(exist_ok=True)
        return gen_dir

    def log_trial(
        self,
        trial_id: str,
        generation: int,
        code: str,
        metrics: dict[str, Any],
        prompt: str,
        response: str,
        reasoning: str,
        parent_id: str | None = None,
        cost_data: dict[str, Any] | None = None,
    ) -> Path:
        """
        Log a trial result.

        Args:
            trial_id: Unique trial identifier
            generation: Generation number
            code: The program code
            metrics: Evaluation metrics
            prompt: The prompt given to the LLM
            response: The full LLM response
            reasoning: Extracted reasoning from response
            parent_id: Optional parent trial ID (for mutations)
            cost_data: Optional cost/token data for this trial

        Returns:
            Path to the trial JSON file
        """
        self._ensure_initialized()

        trial_data = {
            "trial_id": trial_id,
            "generation": generation,
            "parent_id": parent_id,
            "metrics": metrics,
            "code": code,
            "prompt": prompt,
            "response": response,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "cost_data": cost_data,
        }

        gen_dir = self._get_generation_dir(generation)
        trial_path = gen_dir / f"{trial_id}.json"

        with open(trial_path, "w") as f:
            json.dump(trial_data, f, indent=2)

        self._experiment_data["total_trials"] += 1

        return trial_path

    def save_scratchpad(
        self,
        generation: int,
        scratchpad: str | None = None,
        lineage_map: str | None = None,
    ) -> Path | None:
        """
        Save scratchpad content to the generation folder.

        This is called immediately when the scratchpad is updated, and also
        when a generation is finalized (with updated lineage map).

        Args:
            generation: Generation number
            scratchpad: Scratchpad content
            lineage_map: Optional lineage map at time of save

        Returns:
            Path to the scratchpad file, or None if nothing to save
        """
        if scratchpad is None and lineage_map is None:
            return None

        self._ensure_initialized()
        gen_dir = self._get_generation_dir(generation)
        scratchpad_path = gen_dir / "scratchpad.txt"

        with open(scratchpad_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"SCRATCHPAD OUTPUT FOR GENERATION {generation}\n")
            f.write("=" * 60 + "\n\n")
            f.write("This is the scratchpad content that was available to the\n")
            f.write("Root LLM when producing this generation.\n\n")
            f.write("-" * 60 + "\n")
            f.write("LINEAGE MAP\n")
            f.write("-" * 60 + "\n\n")
            f.write(lineage_map if lineage_map else "(No lineage yet)\n")
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("SCRATCHPAD\n")
            f.write("-" * 60 + "\n\n")
            f.write(scratchpad if scratchpad else "(Empty)\n")

        return scratchpad_path

    def log_generation(
        self,
        generation: int,
        trials: list[dict[str, Any]],
        selected_trial_ids: list[str],
        selection_reasoning: str,
        best_trial_id: str | None = None,
        best_score: float = 0.0,
        trial_selections: list[dict[str, Any]] | None = None,
        scratchpad: str | None = None,
        lineage_map: str | None = None,
    ) -> Path:
        """
        Log generation summary.

        Args:
            generation: Generation number
            trials: List of trial data dictionaries
            selected_trial_ids: IDs of trials selected for next generation
            selection_reasoning: Reasoning for selection
            best_trial_id: ID of best trial this generation
            best_score: Best score achieved (sum of radii)
            trial_selections: List of detailed trial selection data with reasoning
            scratchpad: Optional scratchpad content at time of generation
            lineage_map: Optional lineage map at time of generation

        Returns:
            Path to the generation summary file
        """
        self._ensure_initialized()

        gen_data = {
            "generation_num": generation,
            "num_trials": len(trials),
            "num_successful_trials": sum(
                1 for t in trials if t.get("metrics", {}).get("valid", False)
            ),
            "best_trial_id": best_trial_id,
            "best_score": best_score,
            "selected_trial_ids": selected_trial_ids,
            "selection_reasoning": selection_reasoning,
            "trial_selections": trial_selections or [],
            "timestamp": datetime.now().isoformat(),
        }

        self._generation_data.append(gen_data)
        self._experiment_data["num_generations"] = generation + 1

        # Update best trial if this is the best overall
        if self._experiment_data["best_trial"] is None or best_score > self._experiment_data[
            "best_trial"
        ].get("score", 0):
            self._experiment_data["best_trial"] = {
                "trial_id": best_trial_id,
                "score": best_score,
                "generation": generation,
            }

        gen_dir = self._get_generation_dir(generation)
        summary_path = gen_dir / "summary.json"

        with open(summary_path, "w") as f:
            json.dump(gen_data, f, indent=2)

        # Save scratchpad (with final lineage map for this generation)
        self.save_scratchpad(generation, scratchpad, lineage_map)

        # Save experiment state after each generation for incremental progress tracking
        self._save_experiment_state()

        return summary_path

    def _save_experiment_state(self) -> Path:
        """
        Save the current experiment state to experiment.json.

        This is called after each generation to provide incremental progress
        tracking and crash recovery capability.

        Returns:
            Path to the experiment JSON file
        """
        self._experiment_data["generations"] = self._generation_data

        experiment_path = self.base_dir / "experiment.json"
        with open(experiment_path, "w") as f:
            json.dump(self._experiment_data, f, indent=2)

        return experiment_path

    def log_root_turn(
        self,
        turn_number: int,
        role: str,
        content: str,
        code_executed: str | None = None,
        execution_result: str | None = None,
    ) -> None:
        """
        Append a root LLM conversation turn to the log.

        Args:
            turn_number: Sequential turn number
            role: "user", "assistant", or "system"
            content: The message content
            code_executed: Optional code that was executed
            execution_result: Optional result of code execution
        """
        self._ensure_initialized()

        turn_data = {
            "turn": turn_number,
            "role": role,
            "content": content,
            "code_executed": code_executed,
            "execution_result": execution_result,
            "timestamp": datetime.now().isoformat(),
        }

        log_path = self.base_dir / "root_llm_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(turn_data) + "\n")

    def log_cost_tracking(self, cost_data: dict[str, Any]) -> Path:
        """
        Save cost tracking data.

        Args:
            cost_data: Cost tracker data dictionary

        Returns:
            Path to the cost tracking file
        """
        self._ensure_initialized()

        cost_path = self.base_dir / "cost_tracking.json"
        with open(cost_path, "w") as f:
            json.dump(cost_data, f, indent=2)

        return cost_path

    def save_experiment(
        self, termination_reason: str | None = None, scratchpad: str = ""
    ) -> Path:
        """
        Save the full experiment state with finalization data.

        This is called at the end of an experiment to save the final state
        including termination reason and end time.

        Args:
            termination_reason: Why the experiment ended
            scratchpad: The Root LLM's scratchpad content

        Returns:
            Path to the experiment JSON file
        """
        self._ensure_initialized()

        self._experiment_data["end_time"] = datetime.now().isoformat()
        self._experiment_data["termination_reason"] = termination_reason
        self._experiment_data["scratchpad"] = scratchpad

        return self._save_experiment_state()

    def load_experiment(self) -> dict[str, Any]:
        """
        Load experiment state from disk.

        Returns:
            Experiment data dictionary

        Raises:
            FileNotFoundError: If experiment file doesn't exist
        """
        experiment_path = self.base_dir / "experiment.json"
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

        with open(experiment_path) as f:
            data = json.load(f)

        self._experiment_data = data
        self._generation_data = data.get("generations", [])
        self._initialized = True

        return data

    @classmethod
    def from_directory(cls, directory: Path) -> "ExperimentLogger":
        """
        Create a logger from an existing experiment directory.

        Args:
            directory: Path to experiment directory

        Returns:
            ExperimentLogger instance with loaded state
        """
        config_path = directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data = json.load(f)

        from .config import config_from_dict

        config = config_from_dict(config_data)

        # Extract run_id from directory name
        dir_name = directory.name
        exp_name = config.experiment.name
        if dir_name.startswith(exp_name + "_"):
            run_id = dir_name[len(exp_name) + 1 :]
        else:
            run_id = dir_name

        logger = cls(config, run_id=run_id)
        logger.base_dir = directory
        logger.generations_dir = directory / "generations"
        logger.load_experiment()

        return logger
