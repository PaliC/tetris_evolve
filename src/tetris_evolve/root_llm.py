"""
Root LLM Orchestrator for tetris_evolve.

Main orchestrator that runs the Root LLM evolution loop, executing REPL
code blocks and managing the conversation with the Root LLM.
"""

from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .config import Config, load_evaluator
from .cost_tracker import CostTracker
from .evolution_api import EvolutionAPI
from .exceptions import BudgetExceededError, GenerationLimitError
from .llm.client import LLMClient, MockLLMClient
from .llm.prompts import get_root_system_prompt
from .logger import ExperimentLogger
from .repl import REPLEnvironment
from .utils.code_extraction import extract_repl_blocks, extract_selection_block


@dataclass
class OrchestratorResult:
    """Result of running the orchestrator."""

    terminated: bool
    reason: str
    num_generations: int
    best_program: str | None = None
    best_score: float = 0.0
    total_trials: int = 0
    successful_trials: int = 0
    cost_summary: dict[str, Any] | None = None


class RootLLMOrchestrator:
    """
    Main orchestrator for the Root LLM evolution loop.

    The orchestrator runs a generation-driven loop:
    1. Initializes all components (LLM clients, REPL, Evolution API, etc.)
    2. For each generation:
       - Call Root LLM with current generation info
       - Execute REPL code to spawn children
       - Automatically advance to next generation
    3. Returns final results
    """

    def __init__(
        self,
        config: Config,
        root_llm: LLMClient | MockLLMClient | None = None,
        child_llm: LLMClient | MockLLMClient | None = None,
        logger: ExperimentLogger | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Experiment configuration
            root_llm: Optional pre-configured root LLM client (for testing)
            child_llm: Optional pre-configured child LLM client (for testing)
            logger: Optional pre-configured logger (for testing)
        """
        self.config = config
        self.max_generations = config.evolution.max_generations
        self.max_children_per_generation = config.evolution.max_children_per_generation

        # Initialize cost tracker
        self.cost_tracker = CostTracker(config)

        # Initialize logger
        self.logger = logger or ExperimentLogger(config)
        self.logger.create_experiment_directory()

        # Initialize LLM clients
        if root_llm is not None:
            self.root_llm = root_llm
        else:
            self.root_llm = LLMClient(
                model=config.root_llm.model,
                cost_tracker=self.cost_tracker,
                llm_type="root",
            )

        if child_llm is not None:
            self.child_llm = child_llm
        else:
            self.child_llm = LLMClient(
                model=config.child_llm.model,
                cost_tracker=self.cost_tracker,
                llm_type="child",
            )

        # Load evaluator
        self.evaluator = load_evaluator(config.evaluation)

        # Get evaluator kwargs for parallel workers
        evaluator_kwargs = config.evaluation.evaluator_kwargs or {}

        # Initialize Evolution API
        self.evolution_api = EvolutionAPI(
            evaluator=self.evaluator,
            child_llm=self.child_llm,
            cost_tracker=self.cost_tracker,
            logger=self.logger,
            max_generations=config.evolution.max_generations,
            max_children_per_generation=config.evolution.max_children_per_generation,
            child_llm_model=config.child_llm.model,
            evaluator_kwargs=evaluator_kwargs,
        )

        # Initialize REPL with Evolution API functions
        self.repl = REPLEnvironment(api_functions=self.evolution_api.get_api_functions())

        # Conversation state
        self.messages: list[dict[str, str]] = []
        self.turn_number = 0

    def _get_system_prompt(self) -> str:
        """Get the system prompt with current generation info."""
        return get_root_system_prompt(
            max_children_per_generation=self.max_children_per_generation,
            max_generations=self.max_generations,
            current_generation=self.evolution_api.current_generation,
        )

    def build_initial_messages(self) -> list[dict[str, str]]:
        """
        Build the initial messages for the conversation.

        Returns:
            List of message dicts to start the conversation
        """
        # Start with a user message prompting the LLM to begin
        self.messages = [
            {
                "role": "user",
                "content": (
                    f"Begin generation 0. Spawn up to {self.max_children_per_generation} children "
                    "exploring different circle packing strategies."
                ),
            }
        ]
        return self.messages

    def extract_code_blocks(self, response: str) -> list[str]:
        """
        Extract REPL code blocks from the LLM response.

        Args:
            response: The LLM response text

        Returns:
            List of code strings from ```repl``` blocks
        """
        return extract_repl_blocks(response)

    def execute_code_in_repl(self, code: str) -> str:
        """
        Execute code in the REPL and format the result.

        Args:
            code: Python code to execute

        Returns:
            Formatted result string
        """
        result = self.repl.execute(code)

        # Build result message
        parts = []

        if result.stdout:
            parts.append(f"Output:\n{result.stdout}")

        if result.stderr and not result.success:
            parts.append(f"Error:\n{result.stderr}")

        if result.return_value is not None and result.success:
            parts.append(f"Return value: {result.return_value}")

        if not parts:
            if result.success:
                parts.append("(code executed successfully, no output)")
            else:
                parts.append(f"Error: {result.error}")

        return "\n".join(parts)

    def check_termination(self, _response: str) -> bool:
        """
        Check if the evolution has been terminated.

        Args:
            _response: The LLM response text (unused, for interface compatibility)

        Returns:
            True if evolution was terminated
        """
        return self.evolution_api.is_terminated

    def _build_generation_feedback_message(self) -> str:
        """Build a feedback message with results from the previous generation."""
        prev_gen = self.evolution_api.current_generation - 1
        if prev_gen < 0:
            return ""

        gen_summary = self.evolution_api.generations[prev_gen]
        trials = gen_summary.trials

        # Build feedback message
        lines = [
            f"Generation {prev_gen} complete. {len(trials)} children spawned.",
            "",
            "Results:",
        ]

        # Sort by score
        sorted_trials = sorted(
            trials,
            key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
            reverse=True,
        )

        for trial in sorted_trials:
            lines.append(f"  - {trial.trial_id}:")
            lines.append(f"    Code: {trial.code}")
            lines.append(f"    Metrics: {trial.metrics}")
            lines.append(f"    Reasoning: {trial.reasoning}")

        return "\n".join(lines)

    def _build_selection_request_message(self) -> str:
        """Build a message asking Root LLM to select trials for next generation."""
        current_gen = self.evolution_api.current_generation
        gen_summary = self.evolution_api.generations[current_gen]
        trials = gen_summary.trials

        lines = [
            f"Generation {current_gen} spawning complete. {len(trials)} trials evaluated.",
            "",
            "## Trial Results",
            "",
        ]

        # Sort by score but show all
        sorted_trials = sorted(
            trials,
            key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
            reverse=True,
        )

        for i, trial in enumerate(sorted_trials, 1):
            score = trial.metrics.get("sum_radii", 0) if trial.success else 0
            valid_str = "valid" if trial.success else "INVALID"
            lines.append(f"{i}. **{trial.trial_id}** [{valid_str}]")
            lines.append(f"   Score: {score:.4f}")
            reasoning_short = (trial.reasoning or "No reasoning")[:200]
            lines.append(f"   Reasoning: {reasoning_short}...")
            if not trial.success:
                lines.append(f"   Error: {trial.error}")
            lines.append("")

        lines.extend(
            [
                "## Your Task",
                "",
                "Select 2-5 trials to carry forward to the next generation. Consider:",
                "- **Performance**: Which trials achieved the best scores?",
                "- **Diversity**: Which trials use different approaches worth exploring?",
                "- **Potential**: Which trials might improve with refinement, even if current scores are lower?",
                "",
                "Respond with a ```selection``` block containing JSON:",
                "```selection",
                "{",
                '  "selections": [',
                '    {"trial_id": "trial_X_Y", "reasoning": "Why selected", "category": "performance|diversity|potential"},',
                "    ...",
                "  ],",
                '  "summary": "Overall selection reasoning"',
                "}",
                "```",
            ]
        )

        return "\n".join(lines)

    def _parse_selection_response(
        self, response: str
    ) -> tuple[list[dict[str, str]] | None, str | None]:
        """
        Parse selection JSON from LLM response.

        Args:
            response: The LLM response text

        Returns:
            Tuple of (selections list, summary string) or (None, None) if no valid selection
        """
        data = extract_selection_block(response)
        if data is None:
            return None, None

        selections = data.get("selections", [])
        summary = data.get("summary", "")

        if not selections:
            return None, None

        return selections, summary

    def run(self) -> OrchestratorResult:
        """
        Run the Root LLM evolution loop.

        The loop is generation-driven:
        1. Root LLM is called once per generation
        2. Children are spawned via REPL execution
        3. Generation advances automatically after spawning
        4. Results are fed back for next generation

        Returns:
            OrchestratorResult with final statistics
        """
        # Build initial messages
        self.build_initial_messages()

        termination_reason = "max_generations_reached"
        generations_completed = 0

        # Create outer progress bar for generations
        gen_pbar = tqdm(
            total=self.max_generations,
            desc="Generations",
            unit="gen",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        # Create inner progress bar for children in current generation
        children_pbar = tqdm(
            total=self.max_children_per_generation,
            desc="  Children",
            unit="child",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        )

        def update_pbar_postfix() -> None:
            """Update progress bars with current stats."""
            cost_summary = self.cost_tracker.get_summary()
            trials = len(self.evolution_api.all_trials)
            successes = sum(1 for t in self.evolution_api.all_trials.values() if t.success)
            best_score = max(
                (
                    t.metrics.get("sum_radii", 0)
                    for t in self.evolution_api.all_trials.values()
                    if t.success
                ),
                default=0,
            )

            # Update children progress bar
            current_gen = self.evolution_api.current_generation
            current_gen_trials = len(self.evolution_api.generations[current_gen].trials)
            children_pbar.n = current_gen_trials
            children_pbar.refresh()

            gen_pbar.set_postfix(
                trials=trials,
                ok=successes,
                best=f"{best_score:.3f}" if best_score else "N/A",
                cost=f"${cost_summary.total_cost:.2f}",
            )

        try:
            for generation in range(self.max_generations):
                current_gen = self.evolution_api.current_generation
                children_pbar.reset()
                children_pbar.set_description(f"  Gen {current_gen} children")
                update_pbar_postfix()

                # Check budget before LLM call
                try:
                    self.cost_tracker.raise_if_over_budget()
                except BudgetExceededError as e:
                    termination_reason = f"budget_exceeded: {str(e)}"
                    break

                # Get system prompt with current generation info
                system_prompt = self._get_system_prompt()

                # Call Root LLM
                response = self.root_llm.generate(
                    messages=self.messages,
                    system=system_prompt,
                    max_tokens=4096,
                    temperature=0.7,
                )

                assistant_message = response.content
                self.messages.append({"role": "assistant", "content": assistant_message})

                # Log the assistant turn
                self.logger.log_root_turn(
                    turn_number=self.turn_number,
                    role="assistant",
                    content=assistant_message,
                )
                self.turn_number += 1

                # Extract and execute code blocks
                code_blocks = self.extract_code_blocks(assistant_message)
                execution_results = []

                for code in code_blocks:
                    result = self.execute_code_in_repl(code)
                    execution_results.append(f"```\n{code}\n```\n\nResult:\n{result}")

                    # Log the execution
                    self.logger.log_root_turn(
                        turn_number=self.turn_number,
                        role="system",
                        content="REPL execution result",
                        code_executed=code,
                        execution_result=result,
                    )
                    self.turn_number += 1

                    # Update progress after each code block
                    update_pbar_postfix()

                # Check if evolution was explicitly terminated by Root LLM
                if self.check_termination(assistant_message):
                    termination_reason = (
                        self.evolution_api._termination_reason or "evolution_terminated"
                    )
                    generations_completed = generation + 1
                    break

                # Automatically advance generation if children were spawned
                if self.evolution_api.has_children_in_current_generation():
                    # Request selection from Root LLM
                    selection_request = self._build_selection_request_message()
                    self.messages.append({"role": "user", "content": selection_request})

                    # Log the selection request
                    self.logger.log_root_turn(
                        turn_number=self.turn_number,
                        role="user",
                        content=selection_request,
                    )
                    self.turn_number += 1

                    # Get selection response from Root LLM
                    tqdm.write("  └─ Requesting trial selection from Root LLM...")
                    selection_response = self.root_llm.generate(
                        messages=self.messages,
                        system=system_prompt,
                        max_tokens=2048,
                        temperature=0.5,  # Lower temp for more deterministic selection
                    )

                    selection_message = selection_response.content
                    self.messages.append({"role": "assistant", "content": selection_message})

                    # Log the selection response
                    self.logger.log_root_turn(
                        turn_number=self.turn_number,
                        role="assistant",
                        content=selection_message,
                    )
                    self.turn_number += 1

                    # Parse selections from response
                    selections, selection_summary = self._parse_selection_response(
                        selection_message
                    )

                    if selections:
                        tqdm.write(f"  └─ LLM selected {len(selections)} trials")
                    else:
                        tqdm.write("  └─ No valid selection, using auto-selection")

                    try:
                        self.evolution_api._advance_generation(
                            selections=selections,
                            selection_summary=selection_summary,
                        )
                        gen_pbar.update(1)
                        generations_completed = generation + 1
                    except GenerationLimitError:
                        # Reached max generations - generation was finalized before exception
                        gen_pbar.update(1)
                        termination_reason = "max_generations_reached"
                        generations_completed = self.max_generations
                        break

                    # Build feedback message for next generation
                    feedback = self._build_generation_feedback_message()
                    self.messages.append({"role": "user", "content": feedback})

                    # Log the user turn
                    self.logger.log_root_turn(
                        turn_number=self.turn_number,
                        role="user",
                        content=feedback,
                    )
                    self.turn_number += 1
                else:
                    # No children spawned - add execution results and continue
                    if execution_results:
                        user_message = "Execution results:\n\n" + "\n\n---\n\n".join(
                            execution_results
                        )
                    else:
                        user_message = (
                            "No children were spawned. Please spawn children using "
                            "`spawn_children_parallel()` in a ```repl``` code block."
                        )

                    self.messages.append({"role": "user", "content": user_message})
                    self.logger.log_root_turn(
                        turn_number=self.turn_number,
                        role="user",
                        content=user_message,
                    )
                    self.turn_number += 1

        except BudgetExceededError as e:
            termination_reason = f"budget_exceeded: {str(e)}"
        finally:
            children_pbar.close()
            gen_pbar.close()

        # Save experiment
        self.logger.log_cost_tracking(self.cost_tracker.to_dict())
        self.logger.save_experiment(termination_reason=termination_reason)

        # Get best trial info
        best_trials = self.evolution_api._get_best_trials(n=1)
        best_program = best_trials[0]["code"] if best_trials else None
        best_score = best_trials[0]["metrics"].get("sum_radii", 0) if best_trials else 0

        # Compute statistics
        all_trials = self.evolution_api.all_trials
        total_trials = len(all_trials)
        successful_trials = sum(1 for t in all_trials.values() if t.success)

        return OrchestratorResult(
            terminated=True,
            reason=termination_reason,
            num_generations=generations_completed,
            best_program=best_program,
            best_score=best_score,
            total_trials=total_trials,
            successful_trials=successful_trials,
            cost_summary=self.cost_tracker.get_summary().__dict__,
        )
