"""Evolution controller - main orchestrator for the evolution loop."""

import signal
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..evaluation.evaluator import ProgramEvaluator
from ..tracking.cost_tracker import CostTracker
from ..tracking.experiment_tracker import ExperimentTracker
from ..llm.client import LLMClient
from ..llm.child_llm import ChildLLMExecutor
from ..llm.root_llm import RootLLMInterface, ResourceLimitError


@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""

    max_generations: int = 50
    max_cost_usd: float = 100.0
    max_time_minutes: int = 60
    max_children_per_generation: int = 20
    initial_population_size: int = 5
    games_per_evaluation: int = 10
    root_model: str = "claude-sonnet-4-20250514"
    child_model: str = "claude-haiku-3-5-20241022"
    root_temperature: float = 0.5
    child_temperature: float = 0.8
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    @classmethod
    def from_yaml(cls, path: Path) -> "EvolutionConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            EvolutionConfig instance.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert output_dir to Path
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If config values are invalid.
        """
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        if self.max_time_minutes <= 0:
            raise ValueError("max_time_minutes must be positive")
        if self.max_children_per_generation < 1:
            raise ValueError("max_children_per_generation must be at least 1")
        if self.initial_population_size < 1:
            raise ValueError("initial_population_size must be at least 1")
        if self.games_per_evaluation < 1:
            raise ValueError("games_per_evaluation must be at least 1")

    def to_dict(self) -> dict:
        """Convert to dict for saving."""
        return {
            "max_generations": self.max_generations,
            "max_cost_usd": self.max_cost_usd,
            "max_time_minutes": self.max_time_minutes,
            "max_children_per_generation": self.max_children_per_generation,
            "initial_population_size": self.initial_population_size,
            "games_per_evaluation": self.games_per_evaluation,
            "root_model": self.root_model,
            "child_model": self.child_model,
            "root_temperature": self.root_temperature,
            "child_temperature": self.child_temperature,
            "output_dir": str(self.output_dir),
        }


@dataclass
class EvolutionResult:
    """Result of an evolution run."""

    experiment_id: str
    generations_completed: int
    total_cost_usd: float
    total_time_minutes: float
    best_trial_id: Optional[str]
    best_score: Optional[float]
    best_code: Optional[str]
    termination_reason: str


# Initial prompts for population diversity
INITIAL_PROMPTS = [
    "Create a Tetris player that minimizes holes in the board by trying to fill gaps before they form.",
    "Create a Tetris player that clears lines as quickly as possible by prioritizing completed rows.",
    "Create a Tetris player that keeps the board as flat as possible by avoiding height differences.",
    "Create a Tetris player that uses piece lookahead to plan better placements.",
    "Create a simple baseline Tetris player that uses random moves with some basic heuristics.",
]

# System prompt for root LLM
ROOT_SYSTEM_PROMPT = """You are a researcher running an evolutionary algorithm to evolve Tetris-playing programs.

Your goal is to evolve the best possible Tetris player by:
1. Creating initial population with diverse strategies
2. Evaluating each program's performance
3. Selecting the best performers as parents
4. Spawning new children that improve on parents
5. Repeating until convergence or limits reached

## Available Functions

You have access to these functions:

```python
# Spawn a new child LLM to generate code
spawn_child_llm(prompt: str, parent_id: Optional[str] = None) -> dict
# Returns: {trial_id, code, metrics, reasoning, success, error}

# Evaluate code without creating trial
evaluate_program(code: str, num_games: int = 10) -> dict

# Get current population
get_population() -> list[PopulationMember]
# PopulationMember has: trial_id, generation, parent_id, score, lines_cleared, survival_steps, code_preview

# Get full code for a trial
get_trial_code(trial_id: str) -> str

# Get history of all generations
get_generation_history() -> list[GenerationSummary]
# GenerationSummary has: generation, best_score, avg_score, num_trials, best_trial_id

# Get improvement rate (avg improvement over last N generations)
get_improvement_rate(window: int = 3) -> float

# Advance to next generation
advance_generation(selected_trial_ids: list[str], reasoning: str) -> int

# End evolution (call when converged or satisfied)
terminate_evolution(reason: str) -> dict

# Check remaining budget
get_cost_remaining() -> float

# Get all resource limits
get_limits() -> dict
```

## Your Task

Write Python code that will be executed to run ONE iteration of your evolution strategy.
Your code should:
1. Check the current state (population, limits, history)
2. Decide what to do (spawn more children, advance generation, terminate)
3. Execute those actions using the available functions

## Output Format

Provide your reasoning first, then Python code in a code block:

```python
# Your evolution logic here
```

Important:
- Check limits before spawning children
- Use get_population() to see current trials
- Use get_generation_history() to track progress
- Call terminate_evolution() when converged or done
- Call advance_generation() to move to next generation
"""


class EvolutionController:
    """Main controller for the evolution loop."""

    def __init__(self, config: EvolutionConfig) -> None:
        """Initialize evolution controller.

        Args:
            config: Evolution configuration.
        """
        self.config = config
        config.validate()

        self.start_time: Optional[datetime] = None
        self._interrupted = False

        # Will be initialized in run()
        self.cost_tracker: Optional[CostTracker] = None
        self.exp_tracker: Optional[ExperimentTracker] = None
        self.evaluator: Optional[ProgramEvaluator] = None
        self.root_client: Optional[LLMClient] = None
        self.child_client: Optional[LLMClient] = None
        self.child_executor: Optional[ChildLLMExecutor] = None
        self.root_interface: Optional[RootLLMInterface] = None

    def _initialize_components(self) -> None:
        """Initialize all components for a new run."""
        # Initialize trackers
        self.cost_tracker = CostTracker(self.config.max_cost_usd)
        self.exp_tracker = ExperimentTracker(self.config.output_dir)

        # Initialize evaluator
        self.evaluator = ProgramEvaluator(
            num_games=self.config.games_per_evaluation,
            max_steps=10000,
            timeout_seconds=60,
        )

        # Initialize LLM clients
        self.root_client = LLMClient(model=self.config.root_model)
        self.child_client = LLMClient(model=self.config.child_model)

        # Initialize child executor
        self.child_executor = ChildLLMExecutor(
            llm_client=self.child_client,
            temperature=self.config.child_temperature,
        )

        # Initialize root interface
        self.root_interface = RootLLMInterface(
            child_executor=self.child_executor,
            evaluator=self.evaluator,
            experiment_tracker=self.exp_tracker,
            cost_tracker=self.cost_tracker,
            config=self.config.to_dict(),
        )

    def run(self) -> EvolutionResult:
        """Run the evolution loop.

        Returns:
            EvolutionResult with final statistics.
        """
        self.start_time = datetime.now()
        self._interrupted = False

        # Set up signal handlers for graceful shutdown
        old_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        old_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Initialize components
            self._initialize_components()

            # Create experiment
            exp_id = self.exp_tracker.create_experiment(self.config.to_dict())

            # Start first generation
            self.exp_tracker.start_generation(1)

            # Create initial population
            self._create_initial_population()

            # Main evolution loop
            termination_reason = None
            while not self.root_interface.is_terminated:
                # Check hard limits
                limit_reason = self._check_all_limits()
                if limit_reason:
                    termination_reason = limit_reason
                    break

                # Check if interrupted
                if self._interrupted:
                    termination_reason = "interrupted"
                    break

                # Run one root LLM turn
                try:
                    should_continue = self._run_root_llm_turn()
                    if not should_continue:
                        termination_reason = self.root_interface._termination_reason or "root_terminated"
                        break
                except Exception as e:
                    # Log error but continue
                    print(f"Root LLM turn error: {e}")
                    continue

            # Build result
            best_trial = self.exp_tracker.get_best_trial()
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60

            return EvolutionResult(
                experiment_id=exp_id,
                generations_completed=self.root_interface.current_generation,
                total_cost_usd=self.cost_tracker.get_total_cost(),
                total_time_minutes=elapsed,
                best_trial_id=best_trial.trial_id if best_trial else None,
                best_score=best_trial.metrics.get("avg_score") if best_trial and best_trial.metrics else None,
                best_code=best_trial.code if best_trial else None,
                termination_reason=termination_reason or "completed",
            )

        finally:
            # Restore signal handlers
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

            # Save final state
            if self.cost_tracker and self.exp_tracker:
                self.cost_tracker.save(self.exp_tracker.experiment_dir / "cost_history.json")

    def resume(self, experiment_dir: Path) -> EvolutionResult:
        """Resume from an existing experiment.

        Args:
            experiment_dir: Path to experiment directory.

        Returns:
            EvolutionResult with final statistics.
        """
        self.start_time = datetime.now()

        # Load existing experiment
        self.exp_tracker = ExperimentTracker.load_experiment(experiment_dir)

        # Load cost tracker
        cost_path = experiment_dir / "cost_history.json"
        if cost_path.exists():
            self.cost_tracker = CostTracker.load(cost_path)
        else:
            self.cost_tracker = CostTracker(self.config.max_cost_usd)

        # Initialize other components
        self.evaluator = ProgramEvaluator(
            num_games=self.config.games_per_evaluation,
            max_steps=10000,
            timeout_seconds=60,
        )
        self.root_client = LLMClient(model=self.config.root_model)
        self.child_client = LLMClient(model=self.config.child_model)
        self.child_executor = ChildLLMExecutor(
            llm_client=self.child_client,
            temperature=self.config.child_temperature,
        )
        self.root_interface = RootLLMInterface(
            child_executor=self.child_executor,
            evaluator=self.evaluator,
            experiment_tracker=self.exp_tracker,
            cost_tracker=self.cost_tracker,
            config=self.config.to_dict(),
        )

        # Set generation from tracker
        self.root_interface._current_generation = self.exp_tracker.current_generation
        if self.root_interface._current_generation == 0:
            self.root_interface._current_generation = 1
            self.exp_tracker.start_generation(1)

        # Continue the run loop
        return self._continue_run()

    def _continue_run(self) -> EvolutionResult:
        """Continue evolution loop after initialization."""
        old_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        old_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            termination_reason = None
            while not self.root_interface.is_terminated:
                limit_reason = self._check_all_limits()
                if limit_reason:
                    termination_reason = limit_reason
                    break

                if self._interrupted:
                    termination_reason = "interrupted"
                    break

                try:
                    should_continue = self._run_root_llm_turn()
                    if not should_continue:
                        termination_reason = self.root_interface._termination_reason or "root_terminated"
                        break
                except Exception as e:
                    print(f"Root LLM turn error: {e}")
                    continue

            best_trial = self.exp_tracker.get_best_trial()
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60

            return EvolutionResult(
                experiment_id=self.exp_tracker.experiment_id,
                generations_completed=self.root_interface.current_generation,
                total_cost_usd=self.cost_tracker.get_total_cost(),
                total_time_minutes=elapsed,
                best_trial_id=best_trial.trial_id if best_trial else None,
                best_score=best_trial.metrics.get("avg_score") if best_trial and best_trial.metrics else None,
                best_code=best_trial.code if best_trial else None,
                termination_reason=termination_reason or "completed",
            )

        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if self.cost_tracker and self.exp_tracker:
                self.cost_tracker.save(self.exp_tracker.experiment_dir / "cost_history.json")

    def _create_initial_population(self) -> None:
        """Create initial population with diverse strategies."""
        prompts = INITIAL_PROMPTS[: self.config.initial_population_size]

        # Pad with generic prompts if needed
        while len(prompts) < self.config.initial_population_size:
            prompts.append(f"Create an innovative Tetris player with a unique strategy (attempt {len(prompts) + 1})")

        for prompt in prompts:
            try:
                self.root_interface.spawn_child_llm(prompt)
            except ResourceLimitError:
                break
            except Exception as e:
                print(f"Error creating initial population: {e}")
                continue

    def _check_generation_limit(self) -> Optional[str]:
        """Check if generation limit exceeded.

        Returns:
            Termination reason if exceeded, None otherwise.
        """
        if self.root_interface.current_generation > self.config.max_generations:
            return f"generation_limit ({self.config.max_generations})"
        return None

    def _check_cost_limit(self) -> Optional[str]:
        """Check if cost limit exceeded.

        Returns:
            Termination reason if exceeded, None otherwise.
        """
        if self.cost_tracker.get_total_cost() >= self.config.max_cost_usd:
            return f"cost_limit (${self.config.max_cost_usd})"
        return None

    def _check_time_limit(self) -> Optional[str]:
        """Check if time limit exceeded.

        Returns:
            Termination reason if exceeded, None otherwise.
        """
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            if elapsed >= self.config.max_time_minutes:
                return f"time_limit ({self.config.max_time_minutes} min)"
        return None

    def _check_all_limits(self) -> Optional[str]:
        """Check all limits.

        Returns:
            First exceeded limit reason, or None.
        """
        return (
            self._check_generation_limit()
            or self._check_cost_limit()
            or self._check_time_limit()
        )

    def _build_root_prompt(self) -> str:
        """Build the prompt for root LLM with current state.

        Returns:
            Complete prompt string.
        """
        limits = self.root_interface.get_limits()
        population = self.root_interface.get_population()
        history = self.root_interface.get_generation_history()

        # Format current state
        state_lines = [
            "## Current State",
            f"- Generation: {limits['current_gen']}/{limits['max_generations']}",
            f"- Children this gen: {limits['children_this_gen']}/{limits['max_children_per_gen']}",
            f"- Budget: ${limits['cost_remaining_usd']:.2f} remaining of ${limits['max_cost_usd']:.2f}",
            "",
        ]

        # Format population
        if population:
            state_lines.append("## Current Population")
            for p in sorted(population, key=lambda x: x.score, reverse=True):
                parent_info = f" (parent: {p.parent_id})" if p.parent_id else ""
                state_lines.append(
                    f"- {p.trial_id}: score={p.score:.1f}, lines={p.lines_cleared:.1f}{parent_info}"
                )
            state_lines.append("")

        # Format history
        if history:
            state_lines.append("## Generation History")
            for h in history:
                state_lines.append(
                    f"- Gen {h.generation}: best={h.best_score:.1f}, avg={h.avg_score:.1f}, trials={h.num_trials}"
                )
            improvement = self.root_interface.get_improvement_rate()
            state_lines.append(f"- Improvement rate: {improvement:.1%}")
            state_lines.append("")

        state_lines.append("## Instructions")
        state_lines.append("Decide what to do next based on the current state.")
        state_lines.append("Write Python code to execute your decision.")

        return "\n".join(state_lines)

    def _execute_root_code(self, code: str) -> None:
        """Execute Python code in context with root interface functions.

        Args:
            code: Python code to execute.
        """
        # Build execution context with available functions
        context = {
            "spawn_child_llm": self.root_interface.spawn_child_llm,
            "evaluate_program": self.root_interface.evaluate_program,
            "get_population": self.root_interface.get_population,
            "get_trial_code": self.root_interface.get_trial_code,
            "get_generation_history": self.root_interface.get_generation_history,
            "get_improvement_rate": self.root_interface.get_improvement_rate,
            "advance_generation": self.root_interface.advance_generation,
            "terminate_evolution": self.root_interface.terminate_evolution,
            "get_cost_remaining": self.root_interface.get_cost_remaining,
            "get_limits": self.root_interface.get_limits,
            # Standard functions
            "print": print,
            "len": len,
            "max": max,
            "min": min,
            "sorted": sorted,
            "sum": sum,
            "range": range,
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "True": True,
            "False": False,
            "None": None,
        }

        exec(code, context)

    def _run_root_llm_turn(self) -> bool:
        """Run one root LLM turn.

        Returns:
            True to continue, False if terminated.
        """
        # Build prompt with current state
        user_prompt = self._build_root_prompt()

        # Call root LLM
        response = self.root_client.send_message(
            messages=[{"role": "user", "content": user_prompt}],
            system=ROOT_SYSTEM_PROMPT,
            temperature=self.config.root_temperature,
            max_tokens=4096,
        )

        # Record cost
        self.cost_tracker.record_call(
            model=self.root_client.model,
            role="root",
            generation=self.root_interface.current_generation,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        # Extract code from response
        code = self._extract_code(response.content)
        if not code:
            print("Warning: No code found in root LLM response")
            return True  # Continue anyway

        # Execute the code
        try:
            self._execute_root_code(code)
        except ResourceLimitError as e:
            print(f"Resource limit: {e}")
        except Exception as e:
            print(f"Root code execution error: {e}")

        return not self.root_interface.is_terminated

    def _extract_code(self, response: str) -> str:
        """Extract Python code from response.

        Args:
            response: LLM response text.

        Returns:
            Extracted code, or empty string.
        """
        import re

        # Try markdown code blocks
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        print("\nInterrupt received, shutting down gracefully...")
        self._interrupted = True
