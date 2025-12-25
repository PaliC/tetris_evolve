"""
Evolution API for tetris_evolve.

Provides the core API exposed to the Root LLM for controlling the evolution process.
"""

import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from tqdm import tqdm

from .cost_tracker import CostTracker
from .exceptions import ChildrenLimitError, GenerationLimitError
from .llm.prompts import CHILD_LLM_SYSTEM_PROMPT
from .logger import ExperimentLogger
from .parallel_worker import child_worker
from .utils.code_extraction import extract_python_code, extract_reasoning
from .utils.prompt_substitution import substitute_trial_codes


class Evaluator(Protocol):
    """Protocol for evaluators."""

    def evaluate(self, code: str) -> dict[str, Any]:
        """Evaluate code and return metrics."""
        ...


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Any:
        """Generate a response."""
        ...


@dataclass
class TrialResult:
    """Result of a single trial (child LLM spawn + evaluation)."""

    trial_id: str
    code: str
    metrics: dict[str, Any]
    prompt: str
    response: str
    reasoning: str
    success: bool
    parent_id: str | None = None
    error: str | None = None
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "metrics": self.metrics,
            "code": self.code,
            "prompt": self.prompt,
            "response": self.response,
            "reasoning": self.reasoning,
            "success": self.success,
            "parent_id": self.parent_id,
            "error": self.error,
            "generation": self.generation,
        }


@dataclass
class TrialSelection:
    """Selection of a trial for advancement to next generation with reasoning."""

    trial_id: str
    reasoning: str
    category: str  # "performance" | "diversity" | "potential"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "reasoning": self.reasoning,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialSelection":
        """Create TrialSelection from dictionary."""
        return cls(
            trial_id=data["trial_id"],
            reasoning=data.get("reasoning", ""),
            category=data.get("category", "performance"),
        )


@dataclass
class GenerationSummary:
    """Summary of a generation."""

    generation_num: int
    trials: list[TrialResult]
    selected_trial_ids: list[str] = field(default_factory=list)
    selection_reasoning: str = ""
    best_trial_id: str | None = None
    best_score: float = 0.0
    trial_selections: list[TrialSelection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation_num": self.generation_num,
            "num_trials": len(self.trials),
            "selected_trial_ids": self.selected_trial_ids,
            "selection_reasoning": self.selection_reasoning,
            "best_trial_id": self.best_trial_id,
            "best_score": self.best_score,
            "trial_selections": [s.to_dict() for s in self.trial_selections],
        }


class EvolutionAPI:
    """
    Core API for evolution control.

    This class provides functions that are injected into the Root LLM's
    REPL environment for controlling the evolution process.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        child_llm: LLMClientProtocol,
        cost_tracker: CostTracker,
        logger: ExperimentLogger,
        max_generations: int = 10,
        max_children_per_generation: int = 10,
        child_llm_model: str | None = None,
        child_llm_provider: str = "anthropic",
        evaluator_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the Evolution API.

        Args:
            evaluator: Evaluator instance for scoring programs
            child_llm: LLM client for spawning child LLMs
            cost_tracker: CostTracker for budget management
            logger: ExperimentLogger for logging
            max_generations: Maximum number of generations
            max_children_per_generation: Maximum children per generation
            child_llm_model: Model name for child LLM (for parallel spawning)
            child_llm_provider: Provider name for child LLM ("anthropic" or "openrouter")
            evaluator_kwargs: Kwargs for creating evaluator in worker processes
        """
        self.evaluator = evaluator
        self.child_llm = child_llm
        self.cost_tracker = cost_tracker
        self.logger = logger
        self._max_generations = max_generations  # Read-only after init
        self._max_children_per_generation = max_children_per_generation  # Read-only after init
        self.child_llm_model = child_llm_model
        self.child_llm_provider = child_llm_provider
        self.evaluator_kwargs = evaluator_kwargs or {}

        self.current_generation = 0
        self.generations: list[GenerationSummary] = [GenerationSummary(generation_num=0, trials=[])]
        self.all_trials: dict[str, TrialResult] = {}
        self._terminated = False
        self._termination_reason: str | None = None

    @property
    def max_generations(self) -> int:
        """Maximum number of generations (read-only after initialization)."""
        return self._max_generations

    @property
    def max_children_per_generation(self) -> int:
        """Maximum children per generation (read-only after initialization)."""
        return self._max_children_per_generation

    @property
    def is_terminated(self) -> bool:
        """Check if evolution has been terminated."""
        return self._terminated

    def spawn_child_llm(
        self,
        prompt: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Spawn a child LLM to generate a program.

        The prompt should be the complete prompt to send to the child LLM.
        The Root LLM is responsible for crafting the full prompt including
        problem specification, constraints, and parent code if mutating.

        Args:
            prompt: Complete prompt to send to the child LLM
            parent_id: Optional trial ID of parent (for mutation tracking)

        Returns:
            Dictionary with trial result:
            {
                'trial_id': str,
                'code': str,
                'metrics': dict,
                'reasoning': str,
                'success': bool,
                'error': Optional[str]
            }

        Raises:
            ChildrenLimitError: If max_children_per_generation limit is reached
        """
        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Check children limit
        trial_num = len(self.generations[self.current_generation].trials)
        if trial_num >= self.max_children_per_generation:
            raise ChildrenLimitError(
                f"Cannot spawn more children in generation {self.current_generation}. "
                f"Limit of {self.max_children_per_generation} children reached."
            )

        # Generate trial ID
        trial_id = f"trial_{self.current_generation}_{trial_num}"

        # Show progress for child LLM spawn
        tqdm.write(
            f"  └─ Spawning child LLM: gen {self.current_generation}, "
            f"trial {trial_num + 1}/{self.max_children_per_generation}"
            + (f" (parent: {parent_id})" if parent_id else "")
        )

        # Substitute {{CODE_TRIAL_X_Y}} tokens with actual code
        substituted_prompt, substitution_report = substitute_trial_codes(
            prompt,
            all_trials=self.all_trials,
            experiment_dir=str(self.logger.base_dir),
        )
        if substitution_report:
            for sub in substitution_report:
                if sub["success"]:
                    tqdm.write(
                        f"       → Substituted {sub['token']} with code from {sub['trial_id']}"
                    )
                else:
                    tqdm.write(f"       ⚠ Failed to substitute {sub['token']}: {sub['error']}")

        # Call child LLM
        try:
            response = self.child_llm.generate(
                messages=[{"role": "user", "content": substituted_prompt}],
                max_tokens=4096,
                temperature=0.7,
            )
            response_text = response.content
        except Exception as e:
            # LLM call failed
            trial = TrialResult(
                trial_id=trial_id,
                code="",
                metrics={},
                prompt=substituted_prompt,
                response="",
                reasoning="",
                success=False,
                parent_id=parent_id,
                error=f"LLM call failed: {str(e)}",
                generation=self.current_generation,
            )
            self._record_trial(trial)
            return trial.to_dict()

        # Extract code from response
        code = extract_python_code(response_text)
        reasoning = extract_reasoning(response_text)

        if not code:
            # No code found in response
            trial = TrialResult(
                trial_id=trial_id,
                code="",
                metrics={},
                prompt=substituted_prompt,
                response=response_text,
                reasoning=reasoning,
                success=False,
                parent_id=parent_id,
                error="No Python code block found in response",
                generation=self.current_generation,
            )
            self._record_trial(trial)
            return trial.to_dict()

        # Evaluate the code
        try:
            metrics = self.evaluator.evaluate(code)
        except Exception as e:
            metrics = {
                "valid": False,
                "error": f"Evaluation error: {str(e)}",
            }

        # Determine success
        success = bool(metrics.get("valid", False))
        error_value = metrics.get("error") if not success else None
        error_msg = str(error_value) if error_value is not None else None

        trial = TrialResult(
            trial_id=trial_id,
            code=code,
            metrics=metrics,
            prompt=substituted_prompt,
            response=response_text,
            reasoning=reasoning,
            success=success,
            parent_id=parent_id,
            error=error_msg,
            generation=self.current_generation,
        )

        self._record_trial(trial)
        return trial.to_dict()

    def _record_trial(self, trial: TrialResult, skip_file_write: bool = False) -> None:
        """Record a trial in the current generation and log it.

        Args:
            trial: The trial result to record
            skip_file_write: If True, skip writing the trial file (used when
                            file was already written by parallel worker)
        """
        self.generations[self.current_generation].trials.append(trial)
        self.all_trials[trial.trial_id] = trial

        # Update best trial for generation
        gen = self.generations[self.current_generation]
        score = trial.metrics.get("sum_radii", 0) if trial.success else 0
        if score > gen.best_score:
            gen.best_score = score
            gen.best_trial_id = trial.trial_id

        # Show trial result
        if trial.success:
            tqdm.write(f"       ✓ {trial.trial_id}: score={score:.4f}")
        else:
            error_short = (trial.error or "unknown error")[:50]
            tqdm.write(f"       ✗ {trial.trial_id}: {error_short}")

        # Log the trial (write file) unless already written by worker
        if not skip_file_write:
            self.logger.log_trial(
                trial_id=trial.trial_id,
                generation=trial.generation,
                code=trial.code,
                metrics=trial.metrics,
                prompt=trial.prompt,
                response=trial.response,
                reasoning=trial.reasoning,
                parent_id=trial.parent_id,
            )

    def spawn_children_parallel(
        self,
        children: list[dict[str, Any]],
        num_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Spawn multiple child LLMs in parallel using multiprocessing.

        Each child LLM call and its evaluation runs in a separate process.
        This significantly speeds up evolution when spawning multiple children.

        Args:
            children: List of dicts with keys:
                - 'prompt': Complete prompt to send to the child LLM
                - 'parent_id': Optional trial ID of parent (for mutation tracking)
            num_workers: Number of parallel workers (defaults to number of children)

        Returns:
            List of dictionaries with trial results (same format as spawn_child_llm)

        Raises:
            ChildrenLimitError: If spawning would exceed max_children_per_generation
            ValueError: If child_llm_model is not set (required for parallel spawning)
        """
        if not self.child_llm_model:
            raise ValueError(
                "child_llm_model must be set for parallel spawning. "
                "Ensure the EvolutionAPI is initialized with child_llm_model."
            )

        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Check children limit
        current_count = len(self.generations[self.current_generation].trials)
        if current_count + len(children) > self.max_children_per_generation:
            raise ChildrenLimitError(
                f"Cannot spawn {len(children)} children in generation {self.current_generation}. "
                f"Current count: {current_count}, limit: {self.max_children_per_generation}."
            )

        # Show progress
        tqdm.write(
            f"  └─ Spawning {len(children)} child LLMs in parallel: gen {self.current_generation}"
        )

        # Sort children by parent_id for better prompt cache hits
        # Children with the same parent share code prefix, so processing them
        # together improves cache utilization
        indexed_children = list(enumerate(children))
        indexed_children.sort(key=lambda x: x[1].get("parent_id") or "")

        # Pre-assign trial IDs for each child (based on original order for consistency)
        starting_trial_num = len(self.generations[self.current_generation].trials)
        trial_ids = [
            f"trial_{self.current_generation}_{starting_trial_num + i}"
            for i in range(len(children))
        ]

        # Get experiment directory for workers to write trial files
        experiment_dir = str(self.logger.base_dir)

        # Substitute {{CODE_TRIAL_X_Y}} tokens in all prompts before passing to workers
        # We do this in the main process where we have access to all_trials
        substituted_prompts: dict[int, str] = {}
        for orig_idx, child in indexed_children:
            prompt = child.get("prompt", "")
            substituted_prompt, substitution_report = substitute_trial_codes(
                prompt,
                all_trials=self.all_trials,
                experiment_dir=experiment_dir,
            )
            substituted_prompts[orig_idx] = substituted_prompt
            if substitution_report:
                for sub in substitution_report:
                    if sub["success"]:
                        tqdm.write(
                            f"       → Substituted {sub['token']} with code from {sub['trial_id']}"
                        )
                    else:
                        tqdm.write(f"       ⚠ Failed to substitute {sub['token']}: {sub['error']}")

        # Prepare worker arguments in sorted order (by parent_id) for cache optimization
        # Each worker gets: (prompt, parent_id, model, evaluator_kwargs, max_tokens, temperature,
        #                    trial_id, generation, experiment_dir, system_prompt, provider)
        worker_args = []
        result_order = []  # Track original indices for result ordering
        for orig_idx, child in indexed_children:
            parent_id = child.get("parent_id")
            worker_args.append((
                substituted_prompts[orig_idx],  # Use substituted prompt
                parent_id,
                self.child_llm_model,
                self.evaluator_kwargs,
                4096,  # max_tokens
                0.7,   # temperature
                trial_ids[orig_idx],
                self.current_generation,
                experiment_dir,
                CHILD_LLM_SYSTEM_PROMPT,  # Cacheable system prompt
                self.child_llm_provider,  # Provider for child LLM
            ))
            result_order.append(orig_idx)

        # Determine number of workers
        if num_workers is None:
            # get num cores
            num_cores = multiprocessing.cpu_count()
            num_workers = min(len(children), num_cores)

        # Run workers in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            worker_results = pool.map(child_worker, worker_args)

        # Process results and record trials
        # Results come back in sorted order, we'll return them in original order
        results_by_trial_id: dict[str, dict[str, Any]] = {}

        for worker_result in worker_results:
            # Use pre-assigned trial ID from worker
            trial_id = worker_result["trial_id"]

            # Record token usage in cost tracker (including cache stats)
            if worker_result["input_tokens"] > 0 or worker_result["output_tokens"] > 0:
                self.cost_tracker.record_usage(
                    input_tokens=worker_result["input_tokens"],
                    output_tokens=worker_result["output_tokens"],
                    llm_type="child",
                    call_id=worker_result["call_id"],
                    cache_creation_input_tokens=worker_result.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=worker_result.get("cache_read_input_tokens", 0),
                )

            # Create trial result
            trial = TrialResult(
                trial_id=trial_id,
                code=worker_result["code"],
                metrics=worker_result["metrics"],
                prompt=worker_result["prompt"],
                response=worker_result["response_text"],
                reasoning=worker_result["reasoning"],
                success=worker_result["success"],
                parent_id=worker_result["parent_id"],
                error=worker_result["error"],
                generation=self.current_generation,
            )

            self._record_trial(trial, skip_file_write=True)
            results_by_trial_id[trial_id] = trial.to_dict()

        # Return results in original order (by trial_id which preserves input order)
        results = [results_by_trial_id[trial_ids[i]] for i in range(len(children))]
        return results

    def evaluate_program(self, code: str) -> dict[str, Any]:
        """
        Evaluate a program directly without spawning a child LLM.

        Useful for testing code modifications or evaluating manually
        constructed programs.

        Args:
            code: Python code to evaluate

        Returns:
            Evaluation metrics dictionary
        """
        try:
            return self.evaluator.evaluate(code)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Evaluation error: {str(e)}",
            }

    def _advance_generation(
        self,
        selections: list[dict[str, str]] | None = None,
        selection_summary: str | None = None,
    ) -> int:
        """
        Finalize current generation and advance to the next generation.

        This is an internal method called by the orchestrator after children
        are spawned for a generation. It is not exposed to the Root LLM.

        The method first finalizes the current generation (selection, logging),
        then advances to the next generation if possible.

        Args:
            selections: Optional list of LLM-selected trials. Each dict should have:
                - trial_id: The trial ID to carry forward
                - reasoning: Why this trial was selected
                - category: "performance" | "diversity" | "potential"
                If None or empty, falls back to auto-selection.
            selection_summary: Optional overall summary of selection reasoning.

        Returns:
            The new generation number (or current if at max)

        Raises:
            GenerationLimitError: If this was the last generation (after finalizing it)
        """
        # Update current generation summary
        gen = self.generations[self.current_generation]

        # Process selections: use LLM selections if provided, else auto-select
        if selections:
            # Validate and filter selections to only include existing trials
            valid_selections: list[TrialSelection] = []
            valid_trial_ids: list[str] = []
            current_gen_trial_ids = {t.trial_id for t in gen.trials}

            for sel in selections:
                trial_id = sel.get("trial_id", "")
                # Allow selection of trials from current generation
                if trial_id in current_gen_trial_ids:
                    valid_selections.append(TrialSelection.from_dict(sel))
                    valid_trial_ids.append(trial_id)

            if valid_selections:
                # Use LLM selections
                gen.trial_selections = valid_selections
                gen.selected_trial_ids = valid_trial_ids
                gen.selection_reasoning = selection_summary or "LLM-selected trials"
            else:
                # No valid selections, fall back to auto-select
                self._auto_select_trials(gen)
        else:
            # No selections provided, auto-select
            self._auto_select_trials(gen)

        # Log generation
        self.logger.log_generation(
            generation=self.current_generation,
            trials=[t.to_dict() for t in gen.trials],
            selected_trial_ids=gen.selected_trial_ids,
            selection_reasoning=gen.selection_reasoning,
            best_trial_id=gen.best_trial_id,
            best_sum_radii=gen.best_score,
            trial_selections=[s.to_dict() for s in gen.trial_selections],
        )

        # Show generation progress
        num_trials = len(gen.trials)
        num_success = sum(1 for t in gen.trials if t.success)
        tqdm.write(
            f"\n  ═══ Generation {self.current_generation} complete: "
            f"{num_success}/{num_trials} successful, best={gen.best_score:.4f} ═══"
        )
        if gen.trial_selections:
            tqdm.write(f"  Selected trials: {', '.join(gen.selected_trial_ids)}")

        # Check if we can advance to next generation (current_generation is 0-indexed,
        # so if we're at generation max_generations-1, we can't advance)
        if self.current_generation >= self.max_generations - 1:
            tqdm.write(f"  ═══ All {self.max_generations} generations complete ═══\n")
            raise GenerationLimitError(
                f"Cannot advance beyond generation {self.current_generation}. "
                f"Maximum of {self.max_generations} generations reached."
            )

        tqdm.write(
            f"  → Advancing to generation {self.current_generation + 1}/{self.max_generations}\n"
        )

        # Move to next generation
        self.current_generation += 1
        self.generations.append(
            GenerationSummary(generation_num=self.current_generation, trials=[])
        )

        return self.current_generation

    def _auto_select_trials(self, gen: GenerationSummary) -> None:
        """Auto-select best trials when no LLM selection provided."""
        best_trials = self._get_best_trials(n=min(3, len(gen.trials)))
        gen.selected_trial_ids = [t["trial_id"] for t in best_trials]
        gen.selection_reasoning = "Auto-selected top performing trials"
        gen.trial_selections = []

    def can_advance_generation(self) -> bool:
        """Check if we can advance to the next generation."""
        return self.current_generation < self.max_generations - 1

    def has_children_in_current_generation(self) -> bool:
        """Check if any children have been spawned in the current generation."""
        return len(self.generations[self.current_generation].trials) > 0

    def terminate_evolution(self, reason: str, best_program: str | None = None) -> dict[str, Any]:
        """
        End the evolution process and return final results.

        Args:
            reason: Explanation for termination
            best_program: The best program code to save as the final result

        Returns:
            Final results dictionary with best program and statistics
        """
        self._terminated = True
        self._termination_reason = reason

        # Compute statistics
        total_trials = len(self.all_trials)
        successful_trials = sum(1 for t in self.all_trials.values() if t.success)

        # Get best trials for reference
        best_trials = self._get_best_trials(n=5)

        # If no best_program provided, use the best trial's code
        if best_program is None and best_trials:
            best_program = best_trials[0].get("code", "")

        best_score = best_trials[0].get("metrics", {}).get("sum_radii", 0) if best_trials else 0

        # Show termination message
        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"  EVOLUTION TERMINATED: {reason}")
        tqdm.write(f"  Total trials: {total_trials} ({successful_trials} successful)")
        tqdm.write(f"  Generations: {self.current_generation + 1}")
        tqdm.write(f"  Best score: {best_score:.4f}")
        cost_summary = self.cost_tracker.get_summary()
        tqdm.write(f"  Total cost: ${cost_summary.total_cost:.4f}")
        tqdm.write(f"{'=' * 60}\n")

        # Save experiment
        self.logger.log_cost_tracking(self.cost_tracker.to_dict())
        self.logger.save_experiment(termination_reason=reason)

        return {
            "terminated": True,
            "reason": reason,
            "best_program": best_program,
            "num_generations": self.current_generation + 1,
            "total_trials": total_trials,
            "successful_trials": successful_trials,
            "cost_summary": self.cost_tracker.get_summary().__dict__,
        }

    def _get_best_trials(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Get the top n trials by score (internal method).

        Args:
            n: Number of trials to return

        Returns:
            List of trial dictionaries sorted by score (descending)
        """
        # Sort by sum_radii (only valid trials)
        valid_trials = [
            t for t in self.all_trials.values() if t.success and t.metrics.get("sum_radii", 0) > 0
        ]

        sorted_trials = sorted(
            valid_trials,
            key=lambda t: t.metrics.get("sum_radii", 0),
            reverse=True,
        )

        return [t.to_dict() for t in sorted_trials[:n]]

    def _get_generation_history(self) -> list[dict[str, Any]]:
        """
        Get history of all generations (internal method).

        Returns:
            List of generation summary dictionaries
        """
        return [gen.to_dict() for gen in self.generations]

    def _get_cost_remaining(self) -> float:
        """
        Get remaining budget in USD (internal method).

        Returns:
            Remaining budget
        """
        return self.cost_tracker.get_remaining_budget()

    def _get_trial(self, trial_id: str) -> dict[str, Any] | None:
        """
        Get a specific trial by ID (internal method).

        Args:
            trial_id: The trial ID to look up

        Returns:
            Trial dictionary or None if not found
        """
        trial = self.all_trials.get(trial_id)
        return trial.to_dict() if trial else None

    def get_trial_code(self, trial_ids: list[str]) -> dict[str, str | None]:
        """
        Retrieve the code for specific trials by their IDs.

        This function allows the Root LLM to fetch code from previous trials
        on demand, rather than having all code included in feedback messages.
        Use this when you need to inspect or analyze specific trial code.

        For including code in child LLM prompts, prefer using the {{CODE_TRIAL_X_Y}}
        token syntax which automatically substitutes the code.

        Args:
            trial_ids: List of trial IDs to retrieve code for.
                      Example: ["trial_0_3", "trial_1_2", "trial_2_5"]

        Returns:
            Dictionary mapping trial_id to code string (or None if not found).
            Example: {
                "trial_0_3": "import numpy as np\\n...",
                "trial_1_2": None,  # Not found
                "trial_2_5": "import numpy as np\\n..."
            }
        """
        result: dict[str, str | None] = {}
        for trial_id in trial_ids:
            trial = self.all_trials.get(trial_id)
            if trial:
                result[trial_id] = trial.code
            else:
                result[trial_id] = None
        return result

    def _get_current_generation(self) -> int:
        """Get the current generation number (internal method)."""
        return self.current_generation

    def get_api_functions(self) -> dict[str, Callable]:
        """
        Get a dictionary of API functions to inject into the REPL.

        The 5 core evolution functions are exposed:
        - spawn_child_llm: Generate new programs via child LLM (sequential)
        - spawn_children_parallel: Generate multiple programs in parallel
        - evaluate_program: Evaluate code directly
        - terminate_evolution: End the evolution process
        - get_trial_code: Retrieve code from specific trials on demand

        Note: advance_generation is no longer exposed - it happens automatically.

        Returns:
            Dictionary mapping function names to callables
        """
        return {
            "spawn_child_llm": self.spawn_child_llm,
            "spawn_children_parallel": self.spawn_children_parallel,
            "evaluate_program": self.evaluate_program,
            "terminate_evolution": self.terminate_evolution,
            "get_trial_code": self.get_trial_code,
        }
