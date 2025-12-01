"""Root LLM interface providing REPL functions for controlling evolution."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..evaluation.evaluator import ProgramEvaluator, EvaluationResult
from ..tracking.cost_tracker import CostTracker
from ..tracking.experiment_tracker import ExperimentTracker, TrialData, GenerationStats
from .child_llm import ChildLLMExecutor, ChildResult


@dataclass
class PopulationMember:
    """Summary of a trial in the population."""

    trial_id: str
    generation: int
    parent_id: Optional[str]
    score: float
    lines_cleared: float
    survival_steps: float
    code_preview: str  # First 500 chars


@dataclass
class GenerationSummary:
    """Summary statistics for a generation."""

    generation: int
    best_score: float
    avg_score: float
    num_trials: int
    best_trial_id: str


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""

    pass


class RootLLMInterface:
    """REPL functions available to the root LLM for controlling evolution."""

    def __init__(
        self,
        child_executor: ChildLLMExecutor,
        evaluator: ProgramEvaluator,
        experiment_tracker: ExperimentTracker,
        cost_tracker: CostTracker,
        config: dict,
    ) -> None:
        """Initialize root LLM interface.

        Args:
            child_executor: Executor for spawning child LLMs.
            evaluator: Program evaluator for testing code.
            experiment_tracker: Tracker for saving trials/generations.
            cost_tracker: Tracker for LLM API costs.
            config: Configuration dict with limits.
        """
        self.child_executor = child_executor
        self.evaluator = evaluator
        self.exp_tracker = experiment_tracker
        self.cost_tracker = cost_tracker
        self.config = config

        self._current_generation = 1
        self._children_this_gen = 0
        self._terminated = False
        self._termination_reason: Optional[str] = None

        # Limits from config
        self._max_generations = config.get("max_generations", 50)
        self._max_children_per_gen = config.get("max_children_per_generation", 20)

    def spawn_child_llm(
        self,
        prompt: str,
        parent_id: Optional[str] = None,
    ) -> dict:
        """Spawn child LLM to generate code.

        Args:
            prompt: Task description for the child LLM.
            parent_id: Optional trial ID of parent code to improve upon.

        Returns:
            Dict with trial_id, code, metrics, reasoning.

        Raises:
            ResourceLimitError: If children limit exceeded for this generation.
        """
        # Check children limit
        if self._children_this_gen >= self._max_children_per_gen:
            raise ResourceLimitError(
                f"Children limit reached: {self._children_this_gen}/{self._max_children_per_gen}"
            )

        # Get parent code if specified
        parent_code = None
        if parent_id:
            try:
                parent_trial = self.exp_tracker.get_trial(parent_id)
                parent_code = parent_trial.code
            except FileNotFoundError:
                pass  # No parent code available

        # Generate code with child LLM
        child_result = self.child_executor.generate_and_validate(
            prompt=prompt,
            parent_code=parent_code,
            evaluator=self.evaluator,
            max_retries=2,
        )

        # Record LLM cost
        self.cost_tracker.record_call(
            model=self.child_executor.llm_client.model,
            role="child",
            generation=self._current_generation,
            input_tokens=child_result.input_tokens,
            output_tokens=child_result.output_tokens,
            trial_id=None,  # Will be set after trial created
        )

        # Generate trial ID
        trial_id = self.exp_tracker._generate_trial_id()

        # Evaluate code if generation was successful
        metrics = None
        eval_result = None
        if child_result.success:
            eval_result = self.evaluator.evaluate(child_result.code, trial_id)
            if eval_result.success:
                metrics = eval_result.to_dict()

        # Save trial
        trial = TrialData(
            trial_id=trial_id,
            generation=self._current_generation,
            parent_id=parent_id,
            code=child_result.code if child_result.success else "",
            prompt=prompt,
            reasoning=child_result.reasoning,
            llm_response=child_result.raw_response,
            metrics=metrics,
            timestamp=datetime.now(),
        )
        self.exp_tracker.save_trial(trial)

        # Increment counter
        self._children_this_gen += 1

        # Build result dict
        result = {
            "trial_id": trial_id,
            "code": child_result.code if child_result.success else "",
            "metrics": metrics,
            "reasoning": child_result.reasoning,
            "success": child_result.success and (eval_result is None or eval_result.success),
            "error": child_result.error if not child_result.success else (
                eval_result.error if eval_result and not eval_result.success else None
            ),
        }

        return result

    def evaluate_program(self, code: str, num_games: int = 10) -> dict:
        """Evaluate code without creating a trial.

        Args:
            code: Python code to evaluate.
            num_games: Number of games to run.

        Returns:
            Dict with evaluation metrics.
        """
        # Create temporary evaluator with specified game count
        temp_evaluator = ProgramEvaluator(
            num_games=num_games,
            max_steps=self.evaluator.max_steps,
            timeout_seconds=self.evaluator.timeout_seconds,
        )

        result = temp_evaluator.evaluate(code, "temp_eval")
        return result.to_dict()

    def get_population(self) -> list[PopulationMember]:
        """Get current generation's programs.

        Returns:
            List of PopulationMember summaries.
        """
        trials = self.exp_tracker.get_generation_trials(self._current_generation)

        population = []
        for trial in trials:
            if trial.metrics:
                member = PopulationMember(
                    trial_id=trial.trial_id,
                    generation=trial.generation,
                    parent_id=trial.parent_id,
                    score=trial.metrics.get("avg_score", 0.0),
                    lines_cleared=trial.metrics.get("avg_lines_cleared", 0.0),
                    survival_steps=trial.metrics.get("avg_survival_steps", 0.0),
                    code_preview=trial.code[:500] if trial.code else "",
                )
                population.append(member)

        return population

    def get_trial_code(self, trial_id: str) -> str:
        """Get full code for a trial.

        Args:
            trial_id: Trial ID to look up.

        Returns:
            Full code string.
        """
        trial = self.exp_tracker.get_trial(trial_id)
        return trial.code

    def get_generation_history(self) -> list[GenerationSummary]:
        """Get stats from all previous generations.

        Returns:
            List of GenerationSummary objects.
        """
        history = []

        for gen in range(1, self._current_generation):
            trials = self.exp_tracker.get_generation_trials(gen)

            if not trials:
                continue

            # Calculate stats
            scores = [
                t.metrics.get("avg_score", 0.0)
                for t in trials
                if t.metrics
            ]

            if scores:
                best_score = max(scores)
                avg_score = sum(scores) / len(scores)
                best_idx = scores.index(best_score)
                best_trial_id = [t for t in trials if t.metrics][best_idx].trial_id
            else:
                best_score = 0.0
                avg_score = 0.0
                best_trial_id = ""

            summary = GenerationSummary(
                generation=gen,
                best_score=best_score,
                avg_score=avg_score,
                num_trials=len(trials),
                best_trial_id=best_trial_id,
            )
            history.append(summary)

        return history

    def get_improvement_rate(self, window: int = 3) -> float:
        """Calculate improvement rate over last N generations.

        Args:
            window: Number of generations to consider.

        Returns:
            Average improvement rate (0.0 to 1.0+).
        """
        history = self.get_generation_history()

        if len(history) < 2:
            return 0.0

        # Get last N generations
        recent = history[-window:] if len(history) >= window else history

        if len(recent) < 2:
            return 0.0

        # Calculate average improvement
        improvements = []
        for i in range(1, len(recent)):
            prev_score = recent[i - 1].best_score
            curr_score = recent[i].best_score
            if prev_score > 0:
                improvement = (curr_score - prev_score) / prev_score
                improvements.append(improvement)

        if not improvements:
            return 0.0

        return sum(improvements) / len(improvements)

    def advance_generation(
        self,
        selected_trial_ids: list[str],
        reasoning: str,
    ) -> int:
        """Advance to next generation.

        Args:
            selected_trial_ids: Trial IDs selected as parents.
            reasoning: Root LLM's reasoning for selections.

        Returns:
            New generation number.

        Raises:
            ResourceLimitError: If generation limit exceeded.
        """
        # Check generation limit
        if self._current_generation >= self._max_generations:
            raise ResourceLimitError(
                f"Generation limit reached: {self._current_generation}/{self._max_generations}"
            )

        # Calculate stats for current generation
        trials = self.exp_tracker.get_generation_trials(self._current_generation)
        scores = [
            t.metrics.get("avg_score", 0.0)
            for t in trials
            if t.metrics
        ]

        if scores:
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            best_idx = scores.index(best_score)
            best_trial_id = [t for t in trials if t.metrics][best_idx].trial_id
        else:
            best_score = 0.0
            avg_score = 0.0
            best_trial_id = ""

        stats = GenerationStats(
            generation=self._current_generation,
            num_trials=len(trials),
            best_score=best_score,
            avg_score=avg_score,
            best_trial_id=best_trial_id,
            selected_parent_ids=selected_trial_ids,
        )

        # Complete current generation
        self.exp_tracker.complete_generation(
            generation=self._current_generation,
            selected_ids=selected_trial_ids,
            root_reasoning=reasoning,
            stats=stats,
        )

        # Start new generation
        self._current_generation += 1
        self._children_this_gen = 0
        self.exp_tracker.start_generation(self._current_generation)

        return self._current_generation

    def terminate_evolution(self, reason: str) -> dict:
        """End evolution and return summary.

        Args:
            reason: Reason for termination.

        Returns:
            Summary dict with best trial info.
        """
        self._terminated = True
        self._termination_reason = reason

        # Get best trial
        best_trial = self.exp_tracker.get_best_trial()

        # Update experiment stats
        self.exp_tracker.update_experiment_stats({
            "status": "completed",
            "termination_reason": reason,
            "total_generations": self._current_generation,
            "total_cost_usd": self.cost_tracker.get_total_cost(),
            "best_trial_id": best_trial.trial_id if best_trial else None,
            "best_score": best_trial.metrics.get("avg_score") if best_trial and best_trial.metrics else None,
        })

        return {
            "terminated": True,
            "reason": reason,
            "generations_completed": self._current_generation,
            "total_cost_usd": self.cost_tracker.get_total_cost(),
            "best_trial_id": best_trial.trial_id if best_trial else None,
            "best_score": best_trial.metrics.get("avg_score") if best_trial and best_trial.metrics else None,
            "best_code": best_trial.code if best_trial else None,
        }

    def get_cost_remaining(self) -> float:
        """Get remaining budget in USD.

        Returns:
            Remaining budget.
        """
        return self.cost_tracker.get_remaining_budget()

    def get_limits(self) -> dict:
        """Get all resource limits and current state.

        Returns:
            Dict with limits and current values.
        """
        return {
            "max_generations": self._max_generations,
            "max_children_per_gen": self._max_children_per_gen,
            "max_cost_usd": self.cost_tracker.max_cost_usd,
            "current_gen": self._current_generation,
            "children_this_gen": self._children_this_gen,
            "total_cost_usd": self.cost_tracker.get_total_cost(),
            "cost_remaining_usd": self.get_cost_remaining(),
        }

    @property
    def is_terminated(self) -> bool:
        """Check if evolution has been terminated."""
        return self._terminated

    @property
    def current_generation(self) -> int:
        """Get current generation number."""
        return self._current_generation
