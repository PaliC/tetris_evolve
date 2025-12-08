"""
Experiment logging system for tetris_evolve.

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
            "code": code,
            "metrics": metrics,
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

    def log_generation(
        self,
        generation: int,
        trials: list[dict[str, Any]],
        selected_trial_ids: list[str],
        selection_reasoning: str,
        best_trial_id: str | None = None,
        best_sum_radii: float = 0.0,
        trial_selections: list[dict[str, Any]] | None = None,
    ) -> Path:
        """
        Log generation summary.

        Args:
            generation: Generation number
            trials: List of trial data dictionaries
            selected_trial_ids: IDs of trials selected for next generation
            selection_reasoning: Reasoning for selection
            best_trial_id: ID of best trial this generation
            best_sum_radii: Best sum_radii achieved
            trial_selections: List of detailed trial selection data with reasoning

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
            "best_sum_radii": best_sum_radii,
            "selected_trial_ids": selected_trial_ids,
            "selection_reasoning": selection_reasoning,
            "trial_selections": trial_selections or [],
            "timestamp": datetime.now().isoformat(),
        }

        self._generation_data.append(gen_data)
        self._experiment_data["num_generations"] = generation + 1

        # Update best trial if this is the best overall
        if self._experiment_data["best_trial"] is None or best_sum_radii > self._experiment_data[
            "best_trial"
        ].get("sum_radii", 0):
            self._experiment_data["best_trial"] = {
                "trial_id": best_trial_id,
                "sum_radii": best_sum_radii,
                "generation": generation,
            }

        gen_dir = self._get_generation_dir(generation)
        summary_path = gen_dir / "summary.json"

        with open(summary_path, "w") as f:
            json.dump(gen_data, f, indent=2)

        return summary_path

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

    def save_experiment(self, termination_reason: str | None = None) -> Path:
        """
        Save the full experiment state.

        Args:
            termination_reason: Why the experiment ended

        Returns:
            Path to the experiment JSON file
        """
        self._ensure_initialized()

        self._experiment_data["end_time"] = datetime.now().isoformat()
        self._experiment_data["termination_reason"] = termination_reason
        self._experiment_data["generations"] = self._generation_data

        experiment_path = self.base_dir / "experiment.json"
        with open(experiment_path, "w") as f:
            json.dump(self._experiment_data, f, indent=2)

        return experiment_path

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
