"""
Evolution API for mango_evolve.

Provides the core API exposed to the Root LLM for controlling the evolution process.
"""

import multiprocessing
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from typing_extensions import Self

from tqdm import tqdm

from .config import ChildLLMConfig
from .cost_tracker import CostTracker
from .exceptions import CalibrationBudgetError, ChildrenLimitError, GenerationLimitError
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
        max_tokens: int = 65536,
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
    model_alias: str | None = None  # Which child LLM was used
    model_config: dict[str, Any] | None = None  # Model config (model, temperature, etc.)

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
            "model_alias": self.model_alias,
            "model_config": self.model_config,
        }


@dataclass
class TrialView:
    """Read-only view of a trial for REPL access.

    This provides a convenient interface for the Root LLM to query and analyze
    trials in the REPL environment.
    """

    trial_id: str
    code: str
    score: float  # Convenience: metrics.get("score", 0) if valid else 0
    success: bool
    generation: int
    parent_id: str | None
    reasoning: str
    error: str | None
    model_alias: str | None
    metrics: dict[str, Any]  # Full metrics dict

    @classmethod
    def from_trial_result(cls, trial: "TrialResult") -> "Self":
        """Create from internal TrialResult."""
        score = trial.metrics.get("score", 0) if trial.success else 0
        return cls(
            trial_id=trial.trial_id,
            code=trial.code,
            score=score,
            success=trial.success,
            generation=trial.generation,
            parent_id=trial.parent_id,
            reasoning=trial.reasoning,
            error=trial.error,
            model_alias=trial.model_alias,
            metrics=trial.metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"<Trial {self.trial_id} [{status}] score={self.score:.6f}>"


class TrialsProxy:
    """Read-only queryable view of all trials, injected into REPL.

    Provides convenient access to trials with filtering capabilities.
    """

    def __init__(self, api: "EvolutionAPI"):
        self._api = api

    def __len__(self) -> int:
        return len(self._api.all_trials)

    def __iter__(self):
        for trial in self._api.all_trials.values():
            yield TrialView.from_trial_result(trial)

    def __getitem__(self, trial_id: str) -> TrialView:
        trial = self._api.all_trials.get(trial_id)
        if trial is None:
            raise KeyError(f"Trial {trial_id} not found")
        return TrialView.from_trial_result(trial)

    def __contains__(self, trial_id: str) -> bool:
        return trial_id in self._api.all_trials

    def filter(
        self,
        *,
        # Direct attribute filters
        success: bool | None = None,
        generation: int | None = None,
        parent_id: str | None = None,
        model_alias: str | None = None,
        # Lineage filters
        ancestor_of: str | None = None,  # Find trials that are ancestors of this trial
        descendant_of: str | None = None,  # Find trials that descend from this trial
        # Custom predicate (lambda)
        predicate: Callable[[TrialView], bool] | None = None,
        # Sorting and limiting
        sort_by: str | None = None,  # "score", "-score" (descending), "generation", etc.
        limit: int | None = None,
    ) -> list[TrialView]:
        """
        Filter trials by various criteria.

        Examples:
            # Top 5 by score
            trials.filter(success=True, sort_by="-score", limit=5)

            # All from generation 2
            trials.filter(generation=2)

            # Custom predicate
            trials.filter(predicate=lambda t: t.score > 2.4 and "grid" in t.reasoning)

            # All descendants of a trial
            trials.filter(descendant_of="trial_0_3")

            # Combined filters
            trials.filter(success=True, generation=1, sort_by="-score", limit=3)

        Args:
            success: Filter by success/failure
            generation: Filter by generation number
            parent_id: Filter by direct parent
            model_alias: Filter by child LLM model
            descendant_of: Find all trials descending from this trial
            ancestor_of: Find all ancestors of this trial
            predicate: Custom filter function
            sort_by: Sort field ("-score" for descending)
            limit: Max results to return

        Returns:
            List of TrialView objects matching the criteria
        """
        # Build set of descendant/ancestor trial IDs if needed
        descendant_ids: set[str] | None = None
        if descendant_of is not None:
            descendant_ids = self._get_all_descendant_ids(descendant_of)

        ancestor_ids: set[str] | None = None
        if ancestor_of is not None:
            ancestor_ids = self._get_all_ancestor_ids(ancestor_of)

        results = []
        for trial in self._api.all_trials.values():
            # Apply direct attribute filters
            if success is not None and trial.success != success:
                continue
            if generation is not None and trial.generation != generation:
                continue
            if parent_id is not None and trial.parent_id != parent_id:
                continue
            if model_alias is not None and trial.model_alias != model_alias:
                continue

            # Apply lineage filters
            if descendant_ids is not None and trial.trial_id not in descendant_ids:
                continue
            if ancestor_ids is not None and trial.trial_id not in ancestor_ids:
                continue

            view = TrialView.from_trial_result(trial)

            # Apply custom predicate
            if predicate is not None and not predicate(view):
                continue

            results.append(view)

        # Apply sorting
        if sort_by is not None:
            descending = sort_by.startswith("-")
            field_name = sort_by.lstrip("-")
            results.sort(key=lambda t: getattr(t, field_name, 0), reverse=descending)

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return results

    def _get_all_descendant_ids(self, trial_id: str) -> set[str]:
        """Get all trial IDs that descend from the given trial (recursive)."""
        descendants: set[str] = set()
        for trial in self._api.all_trials.values():
            if trial.parent_id == trial_id:
                descendants.add(trial.trial_id)
                descendants.update(self._get_all_descendant_ids(trial.trial_id))
        return descendants

    def _get_all_ancestor_ids(self, trial_id: str) -> set[str]:
        """Get all trial IDs that are ancestors of the given trial."""
        ancestors: set[str] = set()
        trial = self._api.all_trials.get(trial_id)
        while trial and trial.parent_id:
            ancestors.add(trial.parent_id)
            trial = self._api.all_trials.get(trial.parent_id)
        return ancestors

    def __repr__(self) -> str:
        total = len(self)
        success = sum(1 for t in self._api.all_trials.values() if t.success)
        return f"<Trials: {total} total, {success} successful>"


@dataclass
class TrialSelection:
    """Selection of a trial for advancement to next generation with reasoning."""

    trial_id: str
    reasoning: str
    category: str  # "performance" | "diversity" | "potential"
    source_generation: int = -1  # Which generation this trial came from (-1 = unknown)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "reasoning": self.reasoning,
            "category": self.category,
            "source_generation": self.source_generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_generation: int = -1) -> "TrialSelection":
        """Create TrialSelection from dictionary."""
        return cls(
            trial_id=data["trial_id"],
            reasoning=data.get("reasoning", ""),
            category=data.get("category", "performance"),
            source_generation=data.get("source_generation", source_generation),
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
    parents_used: list[str] = field(default_factory=list)
    parents_used_counts: dict[str, int] = field(default_factory=dict)
    parents_not_selected_prev_gen: list[str] = field(default_factory=list)

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
            "parents_used": self.parents_used,
            "parents_used_counts": self.parents_used_counts,
            "parents_not_selected_prev_gen": self.parents_not_selected_prev_gen,
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
        child_llm_configs: dict[str, ChildLLMConfig],
        cost_tracker: CostTracker,
        logger: ExperimentLogger,
        max_generations: int = 10,
        max_children_per_generation: int = 10,
        default_child_llm_alias: str | None = None,
        evaluator_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the Evolution API.

        Args:
            evaluator: Evaluator instance for scoring programs
            child_llm_configs: Dict mapping effective_alias -> ChildLLMConfig
            cost_tracker: CostTracker for budget management
            logger: ExperimentLogger for logging
            max_generations: Maximum number of generations
            max_children_per_generation: Maximum children per generation
            default_child_llm_alias: Default model alias to use if none specified
            evaluator_kwargs: Kwargs for creating evaluator in worker processes
        """
        self.evaluator = evaluator
        self.child_llm_configs = child_llm_configs
        self.cost_tracker = cost_tracker
        self.logger = logger
        self._max_generations = max_generations  # Read-only after init
        self._max_children_per_generation = max_children_per_generation  # Read-only after init
        self.default_child_llm_alias = default_child_llm_alias
        self.evaluator_kwargs = evaluator_kwargs or {}

        # Child LLM clients - created lazily per model
        self.child_llm_clients: dict[str, LLMClientProtocol] = {}

        # Calibration phase tracking
        self.in_calibration_phase: bool = True
        self.calibration_calls_remaining: dict[str, int] = {
            alias: cfg.calibration_calls for alias, cfg in child_llm_configs.items()
        }

        self.current_generation = 0
        self.generations: list[GenerationSummary] = [GenerationSummary(generation_num=0, trials=[])]
        self.all_trials: dict[str, TrialResult] = {}
        self._terminated = False
        self._termination_reason: str | None = None

        # Scratchpad: persistent notes controlled by Root LLM
        self.scratchpad: str = ""

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

    def _resolve_model_alias(self, model: str | None) -> str:
        """Resolve a model alias to an effective alias, validating it exists."""
        if model is None:
            if self.default_child_llm_alias is None:
                raise ValueError(
                    "No model specified and no default_child_llm_alias configured. "
                    f"Available models: {list(self.child_llm_configs.keys())}"
                )
            model = self.default_child_llm_alias

        if model not in self.child_llm_configs:
            raise ValueError(
                f"Unknown model alias '{model}'. Available: {list(self.child_llm_configs.keys())}"
            )
        return model

    def _get_or_create_child_llm_client(self, alias: str) -> LLMClientProtocol:
        """Get or create an LLM client for the specified model alias."""
        if alias not in self.child_llm_clients:
            # Import here to avoid circular imports
            from .llm.client import create_llm_client

            config = self.child_llm_configs[alias]
            self.child_llm_clients[alias] = create_llm_client(
                provider=config.provider,
                model=config.model,
                cost_tracker=self.cost_tracker,
                llm_type=f"child:{alias}",
            )
        return self.child_llm_clients[alias]

    def _record_trial(self, trial: TrialResult, skip_file_write: bool = False) -> None:
        """Record a trial in the current generation and log it.

        Args:
            trial: The trial result to record
            skip_file_write: If True, skip writing the trial file (used when
                            file was already written by parallel worker)
        """
        # Record in all_trials
        self.all_trials[trial.trial_id] = trial

        # Only add to generations list if not a calibration trial
        if trial.generation >= 0:
            self.generations[self.current_generation].trials.append(trial)

            # Update best trial for generation
            gen = self.generations[self.current_generation]
            score = trial.metrics.get("score", 0) if trial.success else 0
            if score > gen.best_score:
                gen.best_score = score
                gen.best_trial_id = trial.trial_id

        # Show trial result
        score = trial.metrics.get("score", 0) if trial.success else 0
        if trial.success:
            tqdm.write(f"       ✓ {trial.trial_id}: score={score:.16f}")
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
                model_config=trial.model_config,
            )

    def spawn_children(
        self,
        children: list[dict[str, Any]],
        num_workers: int | None = None,
    ) -> list[TrialView]:
        """
        Spawn child LLMs in parallel using multiprocessing.

        Each child LLM call and its evaluation runs in a separate process.

        Args:
            children: List of dicts with keys:
                - 'prompt': Complete prompt to send to the child LLM
                - 'parent_id': Optional trial ID of parent (for mutation tracking)
                - 'model': Optional model alias (defaults to default_child_llm_alias)
                - 'temperature': Optional temperature (defaults to 0.7)
            num_workers: Number of parallel workers (defaults to number of children)

        Returns:
            List of TrialView objects with trial results.
            Use .to_dict() on each for backward compatibility if dict format is needed.

        Raises:
            ChildrenLimitError: If spawning would exceed max_children_per_generation
        """
        # Check budget
        self.cost_tracker.raise_if_over_budget()

        # Check children limit (only during evolution, not calibration)
        if not self.in_calibration_phase:
            current_count = len(self.generations[self.current_generation].trials)
            if current_count + len(children) > self.max_children_per_generation:
                raise ChildrenLimitError(
                    f"Cannot spawn {len(children)} children in generation {self.current_generation}. "
                    f"Current count: {current_count}, limit: {self.max_children_per_generation}."
                )

        # Resolve model aliases for all children and check calibration budget
        resolved_models: dict[int, tuple[str, ChildLLMConfig]] = {}
        for i, child in enumerate(children):
            model_alias = self._resolve_model_alias(child.get("model"))
            config = self.child_llm_configs[model_alias]
            resolved_models[i] = (model_alias, config)

            # Check calibration budget if in calibration phase
            if self.in_calibration_phase:
                if self.calibration_calls_remaining[model_alias] <= 0:
                    raise CalibrationBudgetError(
                        f"Calibration budget exhausted for model '{model_alias}'. "
                        f"Remaining calls: {self.calibration_calls_remaining}"
                    )
                self.calibration_calls_remaining[model_alias] -= 1

        # Determine trial generation and system prompt
        trial_generation = -1 if self.in_calibration_phase else self.current_generation
        system_prompt = None if self.in_calibration_phase else CHILD_LLM_SYSTEM_PROMPT

        # Show progress
        phase_str = "calibration" if self.in_calibration_phase else f"gen {self.current_generation}"
        tqdm.write(f"  └─ Spawning {len(children)} child LLMs in parallel: {phase_str}")

        # Sort children by parent_id for better prompt cache hits
        indexed_children = list(enumerate(children))
        indexed_children.sort(key=lambda x: x[1].get("parent_id") or "")

        # Pre-assign trial IDs for each child
        if self.in_calibration_phase:
            starting_trial_num = len([t for t in self.all_trials.values() if t.generation == -1])
        else:
            starting_trial_num = len(self.generations[self.current_generation].trials)
        trial_ids = [
            f"trial_{trial_generation}_{starting_trial_num + i}" for i in range(len(children))
        ]

        # Get experiment directory for workers to write trial files
        experiment_dir = str(self.logger.base_dir)

        # Substitute {{CODE_TRIAL_X_Y}} tokens in all prompts before passing to workers
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
        #                    trial_id, generation, experiment_dir, system_prompt, provider, model_alias)
        worker_args = []
        result_order = []  # Track original indices for result ordering
        for orig_idx, child in indexed_children:
            parent_id = child.get("parent_id")
            temperature = child.get("temperature", 0.7)
            model_alias, config = resolved_models[orig_idx]

            # Serialize model config for worker
            worker_args.append(
                (
                    substituted_prompts[orig_idx],  # Use substituted prompt
                    parent_id,
                    config.model,  # Actual model name
                    self.evaluator_kwargs,
                    4096,  # max_tokens
                    temperature,
                    trial_ids[orig_idx],
                    trial_generation,
                    experiment_dir,
                    system_prompt,  # None during calibration, else cacheable system prompt
                    config.provider,  # Provider for child LLM
                    model_alias,  # Model alias for tracking
                )
            )
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
            model_alias = worker_result.get("model_alias")

            # Record token usage in cost tracker (including cache stats)
            if worker_result["input_tokens"] > 0 or worker_result["output_tokens"] > 0:
                llm_type = f"child:{model_alias}" if model_alias else "child"
                self.cost_tracker.record_usage(
                    input_tokens=worker_result["input_tokens"],
                    output_tokens=worker_result["output_tokens"],
                    llm_type=llm_type,
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
                generation=trial_generation,
                model_alias=model_alias,
                model_config=worker_result.get("model_config"),
            )

            self._record_trial(trial, skip_file_write=True)
            results_by_trial_id[trial_id] = TrialView.from_trial_result(trial)

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
            all_trial_ids = set(self.all_trials.keys())

            for sel in selections:
                trial_id = sel.get("trial_id", "")
                # Allow selection of any existing trial (current or historical)
                if trial_id in all_trial_ids:
                    source_generation = self.all_trials[trial_id].generation
                    valid_selections.append(
                        TrialSelection.from_dict(sel, source_generation=source_generation)
                    )
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

        # Track which parents were actually used to create this generation
        parents_used_counts: dict[str, int] = {}
        for trial in gen.trials:
            if trial.parent_id:
                parents_used_counts[trial.parent_id] = (
                    parents_used_counts.get(trial.parent_id, 0) + 1
                )
        parents_used = sorted(parents_used_counts.keys())

        if self.current_generation > 0:
            prev_selected = set(self.generations[self.current_generation - 1].selected_trial_ids)
            parents_not_selected_prev_gen = sorted(
                [pid for pid in parents_used if pid not in prev_selected]
            )
        else:
            parents_not_selected_prev_gen = []

        gen.parents_used = parents_used
        gen.parents_used_counts = parents_used_counts
        gen.parents_not_selected_prev_gen = parents_not_selected_prev_gen

        # Log generation
        self.logger.log_generation(
            generation=self.current_generation,
            trials=[t.to_dict() for t in gen.trials],
            selected_trial_ids=gen.selected_trial_ids,
            selection_reasoning=gen.selection_reasoning,
            best_trial_id=gen.best_trial_id,
            best_score=gen.best_score,
            trial_selections=[s.to_dict() for s in gen.trial_selections],
            parents_used=parents_used,
            parents_used_counts=parents_used_counts,
            parents_not_selected_prev_gen=parents_not_selected_prev_gen,
        )

        # Show generation progress
        num_trials = len(gen.trials)
        num_success = sum(1 for t in gen.trials if t.success)
        tqdm.write(
            f"\n  ═══ Generation {self.current_generation} complete: "
            f"{num_success}/{num_trials} successful, best={gen.best_score:.16f} ═══"
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

        # Write scratchpad to new generation directory immediately
        self.logger.save_scratchpad(
            generation=self.current_generation,
            scratchpad=self.scratchpad,
            lineage_map=self._build_lineage_map(),
        )

        return self.current_generation

    def _auto_select_trials(self, gen: GenerationSummary) -> None:
        """Auto-select best trials when no LLM selection provided."""
        # Get best trials from current generation only
        current_gen_trials = [t for t in gen.trials if t.success and t.metrics.get("score", 0) > 0]
        sorted_trials = sorted(
            current_gen_trials,
            key=lambda t: t.metrics.get("score", 0),
            reverse=True,
        )[:3]

        gen.selected_trial_ids = [t.trial_id for t in sorted_trials]
        gen.selection_reasoning = "Auto-selected top performing trials"
        gen.trial_selections = [
            TrialSelection(
                trial_id=t.trial_id,
                reasoning="Auto-selected for high score",
                category="performance",
                source_generation=gen.generation_num,
            )
            for t in sorted_trials
        ]

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

        best_score = best_trials[0].get("metrics", {}).get("score", 0) if best_trials else 0

        # Show termination message
        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"  EVOLUTION TERMINATED: {reason}")
        tqdm.write(f"  Total trials: {total_trials} ({successful_trials} successful)")
        tqdm.write(f"  Generations: {self.current_generation + 1}")
        tqdm.write(f"  Best score: {best_score:.16f}")
        cost_summary = self.cost_tracker.get_summary()
        tqdm.write(f"  Total cost: ${cost_summary.total_cost:.4f}")
        tqdm.write(f"{'=' * 60}\n")

        # Save experiment
        self.logger.log_cost_tracking(self.cost_tracker.to_dict())
        self.logger.save_experiment(termination_reason=reason, scratchpad=self.scratchpad)

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
        # Sort by score (only valid trials)
        valid_trials = [
            t for t in self.all_trials.values() if t.success and t.metrics.get("score", 0) > 0
        ]

        sorted_trials = sorted(
            valid_trials,
            key=lambda t: t.metrics.get("score", 0),
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

    def get_top_trials(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve a compact summary of the top-scoring trials across all generations.

        Args:
            n: Number of top trials to return

        Returns:
            List of trial summary dictionaries sorted by score (descending)
        """
        top_trials = self._get_best_trials(n=n)
        summaries: list[dict[str, Any]] = []
        for trial in top_trials:
            metrics = trial.get("metrics", {})
            summaries.append(
                {
                    "trial_id": trial.get("trial_id"),
                    "generation": trial.get("generation"),
                    "score": metrics.get("score", 0),
                    "reasoning": (trial.get("reasoning") or "")[:200],
                    "parent_id": trial.get("parent_id"),
                    "model_alias": trial.get("model_alias"),
                }
            )
        return summaries

    def _get_current_generation(self) -> int:
        """Get the current generation number (internal method)."""
        return self.current_generation

    def update_scratchpad(self, content: str) -> dict[str, Any]:
        """
        Update the scratchpad with new content.

        The scratchpad persists across generations and is shown to you at the
        start of each generation. Use it to track insights, promising approaches,
        and notes for future iterations.

        Suggested structure:
        - **Active Approaches**: What lineages are you developing?
        - **Key Insights**: What have you learned works/doesn't work?
        - **To Try Next**: Ideas for future generations

        Args:
            content: New scratchpad content (replaces existing content).
                     Max recommended length: 4000 characters.

        Returns:
            Dictionary with success status and content length.
        """
        max_length = 8000  # Hard limit to prevent context bloat
        if len(content) > max_length:
            tqdm.write(f"  ⚠️ Scratchpad truncated from {len(content)} to {max_length} chars")
            content = content[:max_length]

        self.scratchpad = content

        # Materialize scratchpad immediately to the current generation folder
        self.logger.save_scratchpad(
            generation=self.current_generation,
            scratchpad=content,
            lineage_map=self._build_lineage_map(),
        )

        tqdm.write(f"  📝 Scratchpad updated ({len(content)} chars)")
        return {"success": True, "length": len(content)}

    def end_calibration_phase(self) -> dict[str, Any]:
        """
        End the calibration phase and transition to main evolution.

        Call this after making test calls to learn about different models.
        After calling this, calibration call limits no longer apply and
        regular evolution begins.

        Returns:
            Summary of calibration phase including calls made per model.
        """
        if not self.in_calibration_phase:
            return {"already_ended": True, "message": "Calibration phase already ended."}

        self.in_calibration_phase = False

        # Calculate calls made per model
        calls_made = {
            alias: config.calibration_calls - self.calibration_calls_remaining[alias]
            for alias, config in self.child_llm_configs.items()
        }

        calibration_trials = [t for t in self.all_trials.values() if t.generation == -1]

        tqdm.write("\n  ══════════════════════════════════════")
        tqdm.write("  CALIBRATION PHASE COMPLETE")
        tqdm.write(f"  Trials: {len(calibration_trials)}")
        tqdm.write(f"  Calls per model: {calls_made}")
        tqdm.write("  ══════════════════════════════════════\n")

        return {
            "calibration_ended": True,
            "calls_made": calls_made,
            "total_calibration_trials": len(calibration_trials),
            "message": "Calibration complete. Evolution will now begin.",
        }

    def get_calibration_status(self) -> dict[str, Any]:
        """
        Get current calibration phase status.

        Use this to check how many calibration calls remain for each model
        and whether you're still in the calibration phase.

        Returns:
            Dictionary with calibration status and remaining calls per model.
        """
        calibration_trials = [t for t in self.all_trials.values() if t.generation == -1]
        return {
            "in_calibration_phase": self.in_calibration_phase,
            "calls_remaining": dict(self.calibration_calls_remaining),
            "total_calibration_trials": len(calibration_trials),
            "available_models": list(self.child_llm_configs.keys()),
        }

    def _build_lineage_map(self) -> str:
        """
        Build a visual lineage map from parent_id relationships.

        Returns a formatted string showing how trials evolved from parents,
        with scores and annotations. Includes an All-Time Top 5 summary for
        quick reference to the best candidates across all generations.
        """
        if not self.all_trials:
            return "(No trials yet)"

        # Build parent -> children mapping
        children_map: dict[str | None, list[str]] = {}
        for trial_id, trial in self.all_trials.items():
            parent = trial.parent_id
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(trial_id)

        # Find root trials (no parent or parent doesn't exist)
        roots = list(children_map.get(None, []))
        for parent_id in children_map:
            if parent_id is not None and parent_id not in self.all_trials:
                roots.extend(children_map[parent_id])

        # Find the global best trial
        best_trial_id = None
        best_score = 0.0
        for trial in self.all_trials.values():
            if trial.success:
                score = trial.metrics.get("score", 0)
                if score > best_score:
                    best_score = score
                    best_trial_id = trial.trial_id

        # Build tree representation
        lines: list[str] = []

        # Add All-Time Top 5 summary at the top
        top_trials = sorted(
            [t for t in self.all_trials.values() if t.success and t.metrics.get("score", 0) > 0],
            key=lambda t: t.metrics.get("score", 0),
            reverse=True,
        )[:5]

        if top_trials:
            lines.append("### All-Time Top 5 (cross-generation selection candidates)")
            for i, trial in enumerate(top_trials, 1):
                score = trial.metrics.get("score", 0)
                gen = trial.generation
                approach = (trial.reasoning or "")[:50].replace("\n", " ").strip()
                lines.append(f"  {i}. {trial.trial_id} [Gen {gen}] = {score:.10f}")
                if approach:
                    lines.append(f"     {approach}...")
            lines.append("")
            lines.append("### Full Lineage Tree")
            lines.append("")

        def format_trial(trial_id: str, indent: str = "", is_last: bool = True) -> None:
            """Recursively format a trial and its descendants."""
            trial = self.all_trials.get(trial_id)
            if not trial:
                return

            # Format score
            if trial.success:
                score = trial.metrics.get("score", 0)
                score_str = f"{score:.16f}"
            else:
                score_str = "INVALID"

            # Mark best trial
            best_marker = " ← best" if trial_id == best_trial_id else ""

            # Get reasoning snippet (first 40 chars)
            reasoning_snippet = ""
            if trial.reasoning:
                snippet = trial.reasoning[:40].replace("\n", " ").strip()
                if snippet:
                    reasoning_snippet = f" [{snippet}...]"

            # Build the line
            prefix = "└── " if indent else ""
            lines.append(
                f"{indent}{prefix}{trial_id} ({score_str}){reasoning_snippet}{best_marker}"
            )

            # Process children
            trial_children = children_map.get(trial_id, [])
            # Sort children by generation and trial number
            trial_children.sort()

            for i, child_id in enumerate(trial_children):
                is_last_child = i == len(trial_children) - 1
                child_indent = indent + ("    " if is_last else "│   ")
                format_trial(child_id, child_indent, is_last_child)

        # Sort roots by score (best first)
        def get_score(trial_id: str) -> float:
            trial = self.all_trials.get(trial_id)
            if trial and trial.success:
                return trial.metrics.get("score", 0)
            return 0.0

        roots.sort(key=get_score, reverse=True)

        # Format each root lineage
        for root_id in roots:
            format_trial(root_id)
            lines.append("")  # Empty line between lineages

        return "\n".join(lines).rstrip()

    def get_api_functions(self) -> dict[str, Callable]:
        """
        Get a dictionary of API functions to inject into the REPL.

        Core evolution functions exposed:
        - spawn_children: Generate multiple programs in parallel
        - evaluate_program: Evaluate code directly
        - terminate_evolution: End the evolution process
        - get_top_trials: Retrieve top trials across history (summary)
        - update_scratchpad: Update persistent notes across generations
        - end_calibration_phase: End calibration and start evolution
        - get_calibration_status: Check calibration phase status

        Note: advance_generation is no longer exposed - it happens automatically.

        Returns:
            Dictionary mapping function names to callables
        """
        return {
            "spawn_children": self.spawn_children,
            "evaluate_program": self.evaluate_program,
            "terminate_evolution": self.terminate_evolution,
            "get_top_trials": self.get_top_trials,
            "update_scratchpad": self.update_scratchpad,
            "end_calibration_phase": self.end_calibration_phase,
            "get_calibration_status": self.get_calibration_status,
        }

    def get_repl_namespace(self) -> dict[str, Any]:
        """
        Get functions AND variables to inject into the REPL namespace.

        This extends get_api_functions() to also include the `trials` variable
        for direct access to trial data in the REPL.

        Returns:
            Dictionary mapping names to callables or values to inject
        """
        namespace = self.get_api_functions()
        # Add the trials variable (TrialsProxy)
        namespace["trials"] = TrialsProxy(self)
        return namespace
