"""
Evolution API for tetris_evolve.

Provides the core API exposed to the Root LLM for controlling the evolution process.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from tqdm import tqdm

from .cost_tracker import CostTracker
from .exceptions import ChildrenLimitError, GenerationLimitError
from .logger import ExperimentLogger
from .utils.code_extraction import extract_python_code, extract_reasoning


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
            "code": self.code,
            "metrics": self.metrics,
            "prompt": self.prompt,
            "response": self.response,
            "reasoning": self.reasoning,
            "success": self.success,
            "parent_id": self.parent_id,
            "error": self.error,
            "generation": self.generation,
        }


@dataclass
class GenerationSummary:
    """Summary of a generation."""

    generation_num: int
    trials: list[TrialResult]
    selected_trial_ids: list[str] = field(default_factory=list)
    selection_reasoning: str = ""
    best_trial_id: str | None = None
    best_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation_num": self.generation_num,
            "num_trials": len(self.trials),
            "selected_trial_ids": self.selected_trial_ids,
            "selection_reasoning": self.selection_reasoning,
            "best_trial_id": self.best_trial_id,
            "best_score": self.best_score,
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
        """
        self.evaluator = evaluator
        self.child_llm = child_llm
        self.cost_tracker = cost_tracker
        self.logger = logger
        self.max_generations = max_generations
        self.max_children_per_generation = max_children_per_generation

        self.current_generation = 0
        self.generations: list[GenerationSummary] = [
            GenerationSummary(generation_num=0, trials=[])
        ]
        self.all_trials: dict[str, TrialResult] = {}
        self._terminated = False
        self._termination_reason: str | None = None

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
                f"Limit of {self.max_children_per_generation} children reached. "
                f"Call advance_generation() to move to the next generation."
            )

        # Generate trial ID
        trial_id = f"trial_{self.current_generation}_{trial_num}"

        # Show progress for child LLM spawn
        tqdm.write(
            f"  └─ Spawning child LLM: gen {self.current_generation}, "
            f"trial {trial_num + 1}/{self.max_children_per_generation}"
            + (f" (parent: {parent_id})" if parent_id else "")
        )

        # Call child LLM
        try:
            response = self.child_llm.generate(
                messages=[{"role": "user", "content": prompt}],
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
                prompt=prompt,
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
                prompt=prompt,
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
            prompt=prompt,
            response=response_text,
            reasoning=reasoning,
            success=success,
            parent_id=parent_id,
            error=error_msg,
            generation=self.current_generation,
        )

        self._record_trial(trial)
        return trial.to_dict()

    def _record_trial(self, trial: TrialResult) -> None:
        """Record a trial in the current generation and log it."""
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
            tqdm.write(
                f"       ✓ {trial.trial_id}: score={score:.4f}"
            )
        else:
            error_short = (trial.error or "unknown error")[:50]
            tqdm.write(f"       ✗ {trial.trial_id}: {error_short}")

        # Log the trial
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

    def advance_generation(
        self,
        selected_trial_ids: list[str],
        reasoning: str,
    ) -> int:
        """
        Advance to the next generation with selected trials as parents.

        Args:
            selected_trial_ids: IDs of trials to use as parents
            reasoning: Explanation for selection

        Returns:
            The new generation number

        Raises:
            GenerationLimitError: If max_generations limit is reached
        """
        # Check generation limit (current_generation is 0-indexed,
        # so if we're at generation max_generations-1, we can't advance)
        if self.current_generation >= self.max_generations - 1:
            raise GenerationLimitError(
                f"Cannot advance beyond generation {self.current_generation}. "
                f"Maximum of {self.max_generations} generations reached. "
                f"Call terminate_evolution() to end the evolution process."
            )

        # Update current generation summary
        gen = self.generations[self.current_generation]
        gen.selected_trial_ids = selected_trial_ids
        gen.selection_reasoning = reasoning

        # Log generation
        self.logger.log_generation(
            generation=self.current_generation,
            trials=[t.to_dict() for t in gen.trials],
            selected_trial_ids=selected_trial_ids,
            selection_reasoning=reasoning,
            best_trial_id=gen.best_trial_id,
            best_sum_radii=gen.best_score,
        )

        # Show generation progress
        num_trials = len(gen.trials)
        num_success = sum(1 for t in gen.trials if t.success)
        tqdm.write(
            f"\n  ═══ Generation {self.current_generation} complete: "
            f"{num_success}/{num_trials} successful, best={gen.best_score:.4f} ═══"
        )
        tqdm.write(f"  → Advancing to generation {self.current_generation + 1}/{self.max_generations}\n")

        # Move to next generation
        self.current_generation += 1
        self.generations.append(
            GenerationSummary(generation_num=self.current_generation, trials=[])
        )

        return self.current_generation

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
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  EVOLUTION TERMINATED: {reason}")
        tqdm.write(f"  Total trials: {total_trials} ({successful_trials} successful)")
        tqdm.write(f"  Generations: {self.current_generation + 1}")
        tqdm.write(f"  Best score: {best_score:.4f}")
        cost_summary = self.cost_tracker.get_summary()
        tqdm.write(f"  Total cost: ${cost_summary.total_cost:.4f}")
        tqdm.write(f"{'='*60}\n")

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
            t for t in self.all_trials.values()
            if t.success and t.metrics.get("sum_radii", 0) > 0
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

    def _get_current_generation(self) -> int:
        """Get the current generation number (internal method)."""
        return self.current_generation

    def get_api_functions(self) -> dict[str, Callable]:
        """
        Get a dictionary of API functions to inject into the REPL.

        Only the 4 core evolution functions are exposed:
        - spawn_child_llm: Generate new programs via child LLM
        - evaluate_program: Evaluate code directly
        - advance_generation: Move to next generation
        - terminate_evolution: End the evolution process

        Returns:
            Dictionary mapping function names to callables
        """
        return {
            "spawn_child_llm": self.spawn_child_llm,
            "evaluate_program": self.evaluate_program,
            "advance_generation": self.advance_generation,
            "terminate_evolution": self.terminate_evolution,
        }
