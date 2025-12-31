"""
Root LLM Orchestrator for mango_evolve.

Main orchestrator that runs the Root LLM evolution loop, executing REPL
code blocks and managing the conversation with the Root LLM.
"""

import traceback
from dataclasses import asdict, dataclass
from typing import Any

from tqdm import tqdm

from .config import ChildLLMConfig, Config, load_evaluator, save_calibration_notes, load_calibration_notes
from .cost_tracker import CostTracker
from .evolution_api import EvolutionAPI
from .exceptions import BudgetExceededError, GenerationLimitError
from .llm.client import LLMClient, MockLLMClient, create_llm_client
from .llm.prompts import get_root_system_prompt_parts_with_models, get_calibration_system_prompt_parts
from .logger import ExperimentLogger
from .repl import REPLEnvironment
from .utils.code_extraction import extract_python_blocks, extract_selection_block


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


@dataclass
class SelectionResult:
    """Result of trial selection and generation advancement."""

    should_break: bool = False
    termination_reason: str | None = None
    generations_completed: int | None = None


@dataclass
class GenerationResult:
    """Result of executing a single generation."""

    should_break: bool = False
    termination_reason: str | None = None
    generations_completed: int | None = None
    errors: list[tuple[int, str]] | None = None


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
        child_llm_clients: dict[str, LLMClient | MockLLMClient] | None = None,
        logger: ExperimentLogger | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Experiment configuration
            root_llm: Optional pre-configured root LLM client (for testing)
            child_llm_clients: Optional pre-configured child LLM clients by alias (for testing)
            logger: Optional pre-configured logger (for testing)
        """
        self.config = config
        # Note: max_generations and max_children_per_generation are accessed via
        # self.evolution_api to ensure a single source of truth

        # Initialize cost tracker
        self.cost_tracker = CostTracker(config)

        # Initialize logger
        self.logger = logger or ExperimentLogger(config)
        self.logger.create_experiment_directory()

        # Initialize Root LLM client
        if root_llm is not None:
            self.root_llm = root_llm
        else:
            self.root_llm = create_llm_client(
                provider=config.root_llm.provider,
                model=config.root_llm.model,
                cost_tracker=self.cost_tracker,
                llm_type="root",
                reasoning_config=asdict(config.root_llm.reasoning) if config.root_llm.reasoning else None,
            )

        # Build child LLM configs dict keyed by effective_alias
        self.child_llm_configs: dict[str, ChildLLMConfig] = {
            cfg.effective_alias: cfg for cfg in config.child_llms
        }

        # Store pre-configured child clients (for testing) or empty dict (lazy init)
        self.child_llm_clients = child_llm_clients or {}

        # Load evaluator
        self.evaluator = load_evaluator(config.evaluation)

        # Get evaluator kwargs for parallel workers
        evaluator_kwargs = config.evaluation.evaluator_kwargs or {}

        # Initialize Evolution API with multi-model support
        self.evolution_api = EvolutionAPI(
            evaluator=self.evaluator,
            child_llm_configs=self.child_llm_configs,
            cost_tracker=self.cost_tracker,
            logger=self.logger,
            max_generations=config.evolution.max_generations,
            max_children_per_generation=config.evolution.max_children_per_generation,
            default_child_llm_alias=config.default_child_llm_alias,
            evaluator_kwargs=evaluator_kwargs,
        )

        # Initialize REPL with Evolution API functions
        self.repl = REPLEnvironment(api_functions=self.evolution_api.get_api_functions())

        # Conversation state
        self.messages: list[dict[str, str]] = []
        self.turn_number = 0

        # Calibration state
        self._calibration_messages: list[dict[str, str]] = []
        self._calibration_turn_number = 0

    def _has_calibration_budget(self) -> bool:
        """Check if any model has calibration calls remaining."""
        return any(
            cfg.calibration_calls > 0
            for cfg in self.child_llm_configs.values()
        )

    def _build_calibration_prompt(self) -> str:
        """Build the calibration phase prompt with model info."""
        lines = [
            "# Calibration Phase",
            "",
            "Before evolution begins, you can test the available child LLMs to understand their "
            "capabilities and find optimal settings. Each model has a limited calibration budget.",
            "",
            "## Available Child LLMs",
            "",
        ]

        for alias, cfg in self.child_llm_configs.items():
            lines.extend([
                f"### {alias}",
                f"- **Model**: {cfg.model}",
                f"- **Provider**: {cfg.provider}",
                f"- **Calibration calls**: {cfg.calibration_calls}",
                f"- **Cost**: ${cfg.cost_per_million_input_tokens:.2f}/M input, "
                f"${cfg.cost_per_million_output_tokens:.2f}/M output",
                "",
            ])

        lines.extend([
            "## Your Task",
            "",
            "Use the available calibration calls to:",
            "1. Test each model with simple circle packing prompts",
            "2. Experiment with different temperature settings (0.0-1.0)",
            "3. Observe output quality, code style, and reasoning depth",
            "4. Update your scratchpad with observations about each model",
            "",
            "Use `spawn_child_llm(prompt, model=alias, temperature=T)` to test models.",
            "Use `update_scratchpad(content)` to record your observations.",
            "Use `get_calibration_status()` to check remaining calibration calls.",
            "",
            "When done calibrating, call `end_calibration_phase()` to begin evolution.",
            "",
        ])

        return "\n".join(lines)

    def _get_calibration_system_prompt(self) -> list[dict]:
        """Get the calibration system prompt as structured content blocks."""
        return get_calibration_system_prompt_parts(
            child_llm_configs=self.child_llm_configs,
        )

    def _run_calibration_phase(self) -> None:
        """Run the calibration phase before evolution begins."""
        tqdm.write("📊 Starting calibration phase...")

        # Build initial calibration message
        calibration_prompt = self._build_calibration_prompt()
        self._calibration_messages = [
            {"role": "user", "content": calibration_prompt}
        ]

        # Get system prompt for calibration
        system_prompt = self._get_calibration_system_prompt()

        max_calibration_turns = 20  # Prevent infinite loops

        for turn in range(max_calibration_turns):
            # Check if calibration is complete
            if not self.evolution_api.in_calibration_phase:
                break

            # Check budget
            try:
                self.cost_tracker.raise_if_over_budget()
            except BudgetExceededError as e:
                tqdm.write(f"⚠️ Budget exceeded during calibration: {e}")
                break

            # Call Root LLM
            response = self.root_llm.generate(
                messages=self._calibration_messages,
                system=system_prompt,
                max_tokens=4096,
                temperature=0.7,
                enable_caching=False,
            )

            assistant_message = response.content
            self._calibration_messages.append({"role": "assistant", "content": assistant_message})

            # Log the calibration turn
            self.logger.log_root_turn(
                turn_number=self._calibration_turn_number,
                role="assistant",
                content=f"[CALIBRATION] {assistant_message}",
            )
            self._calibration_turn_number += 1

            # Extract and execute code blocks
            code_blocks = self.extract_code_blocks(assistant_message)
            execution_results = []

            for code in code_blocks:
                result = self.execute_code_in_repl(code)
                execution_results.append(f"```\n{code}\n```\n\nResult:\n{result}")

                # Log execution
                self.logger.log_root_turn(
                    turn_number=self._calibration_turn_number,
                    role="system",
                    content="[CALIBRATION] REPL execution",
                    code_executed=code,
                    execution_result=result,
                )
                self._calibration_turn_number += 1

            # Check if end_calibration_phase was called
            if not self.evolution_api.in_calibration_phase:
                tqdm.write("✅ Calibration phase complete.")
                break

            # Build user message with execution results and status
            status = self.evolution_api.get_calibration_status()
            remaining = status.get("remaining_calls", {})
            remaining_str = ", ".join(f"{k}: {v}" for k, v in remaining.items())

            if execution_results:
                user_content = (
                    "Execution results:\n\n"
                    + "\n\n---\n\n".join(execution_results)
                    + f"\n\nRemaining calibration calls: {remaining_str}"
                )
            else:
                user_content = (
                    f"No code executed. Remaining calibration calls: {remaining_str}\n\n"
                    "Use `spawn_child_llm(prompt, model=alias, temperature=T)` to test models, "
                    "or call `end_calibration_phase()` when done."
                )

            self._calibration_messages.append({"role": "user", "content": user_content})

            self.logger.log_root_turn(
                turn_number=self._calibration_turn_number,
                role="user",
                content=f"[CALIBRATION] {user_content}",
            )
            self._calibration_turn_number += 1

        # Ensure calibration is ended if we hit max turns
        if self.evolution_api.in_calibration_phase:
            tqdm.write("⚠️ Max calibration turns reached, ending calibration.")
            self.evolution_api.end_calibration_phase()

    def _save_calibration_notes(self) -> str:
        """Save calibration notes to a file and return the path."""
        notes_path = self.logger.experiment_dir / "calibration_notes.yaml"

        save_calibration_notes(
            path=str(notes_path),
            notes=self.evolution_api.scratchpad,
            child_llm_configs=list(self.child_llm_configs.values()),
            experiment_name=self.config.experiment.name,
        )

        tqdm.write(f"📝 Calibration notes saved to: {notes_path}")
        return str(notes_path)

    def _load_calibration_notes(self, path: str) -> None:
        """Load calibration notes from a file and inject into scratchpad."""
        tqdm.write(f"📖 Loading calibration notes from: {path}")

        cal_notes = load_calibration_notes(path)

        # Inject notes into scratchpad (not metadata)
        self.evolution_api.update_scratchpad(cal_notes.notes)

        # Skip calibration phase since we have pre-existing notes
        self.evolution_api.end_calibration_phase()

        tqdm.write("✅ Calibration notes loaded, skipping calibration phase.")

    def _get_system_prompt(self) -> list[dict]:
        """Get the system prompt as structured content blocks for caching.

        Returns:
            List of content blocks with cache_control on the static portion.
        """
        # Get timeout from evaluator kwargs
        timeout_seconds = self.evolution_api.evaluator_kwargs.get("timeout_seconds")

        return get_root_system_prompt_parts_with_models(
            child_llm_configs=self.child_llm_configs,
            default_child_llm_alias=self.config.default_child_llm_alias,
            max_children_per_generation=self.evolution_api.max_children_per_generation,
            max_generations=self.evolution_api.max_generations,
            current_generation=self.evolution_api.current_generation,
            timeout_seconds=timeout_seconds,
        )

    def build_initial_messages(self) -> list[dict[str, str]]:
        """
        Build the initial messages for the conversation.

        Returns:
            List of message dicts to start the conversation
        """
        # Build evolution memory (empty for generation 0, but shows the structure)
        evolution_memory = self._build_evolution_memory()

        # Start with a user message prompting the LLM to begin
        self.messages = [
            {
                "role": "user",
                "content": (
                    f"Begin generation 0. Spawn up to {self.evolution_api.max_children_per_generation} children "
                    "exploring different circle packing strategies.\n\n"
                    f"{evolution_memory}"
                ),
            }
        ]
        return self.messages

    def _prepare_messages_with_caching(
        self, messages: list[dict[str, str]]
    ) -> list[dict]:
        """
        Prepare messages with cache_control for optimal prompt caching.

        Strategy: Cache at the END of a stable prefix that won't change.
        We cache the first user message (stable "Begin generation 0..." prompt)
        since it remains constant throughout the conversation.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            List of message dicts with cache_control added where appropriate.
            Content may be converted to list format for messages with cache_control.
        """
        if len(messages) < 1:
            return messages

        result = []
        for i, msg in enumerate(messages):
            if i == 0:
                # Cache the first user message (stable "Begin generation 0..." prompt)
                result.append({
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                })
            else:
                result.append(msg)

        return result

    def _prune_message_history(self, keep_recent_generations: int = 2) -> None:
        """
        Prune old messages to prevent unbounded context growth.

        Keeps:
        - Initial user message (problem setup)
        - Messages from the last N generations
        - Replaces older generation messages with summaries

        Args:
            keep_recent_generations: Number of recent generations to keep in full detail
        """
        if self.evolution_api.current_generation <= keep_recent_generations:
            return  # Not enough history to prune

        cutoff_generation = self.evolution_api.current_generation - keep_recent_generations

        # Build summary of old generations
        old_gen_summaries = []
        for gen_num in range(cutoff_generation):
            if gen_num < len(self.evolution_api.generations):
                gen = self.evolution_api.generations[gen_num]
                best_score = gen.best_score
                best_trial = gen.best_trial_id or "N/A"
                selected = ", ".join(gen.selected_trial_ids) if gen.selected_trial_ids else "N/A"
                summary = (
                    f"Gen {gen_num}: {len(gen.trials)} trials, "
                    f"best={best_score:.16f} ({best_trial}), "
                    f"selected=[{selected}]"
                )
                old_gen_summaries.append(summary)

        if not old_gen_summaries:
            return

        # Create consolidated history message
        history_summary = {
            "role": "user",
            "content": (
                f"[Historical Summary - Generations 0 to {cutoff_generation - 1}]\n"
                + "\n".join(old_gen_summaries)
                + f"\n\n[Detailed history continues from generation {cutoff_generation}]"
            ),
        }

        # Find where recent generations start in message history
        # Each generation adds ~5 messages, so estimate the cutoff point
        messages_per_gen = 5
        keep_from_index = max(1, len(self.messages) - (keep_recent_generations * messages_per_gen))

        # Rebuild messages: initial message + summary + recent messages
        self.messages = (
            [self.messages[0]]  # Keep initial "Begin generation 0" message
            + [history_summary]
            + self.messages[keep_from_index:]
        )

        tqdm.write(
            f"  📦 Pruned message history: keeping generations {cutoff_generation}-"
            f"{self.evolution_api.current_generation}, summarized earlier generations"
        )

    def _get_context_size_estimate(self) -> dict:
        """
        Estimate current context size.

        Returns:
            Dictionary with context statistics and health indicators.
        """
        total_chars = sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str)
            else sum(len(c.get("text", "")) for c in m.get("content", []) if isinstance(c, dict))
            for m in self.messages
        )
        estimated_tokens = total_chars // 4

        return {
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "message_count": len(self.messages),
            "warning": estimated_tokens > 150_000,
            "critical": estimated_tokens > 250_000,
        }

    def _check_context_health(self) -> None:
        """Check context size and warn if growing too large."""
        stats = self._get_context_size_estimate()

        if stats["critical"]:
            tqdm.write(
                f"  ⚠️  CRITICAL: Context size ~{stats['estimated_tokens']:,} tokens. "
                "Risk of API failure. Consider pruning history."
            )
        elif stats["warning"]:
            tqdm.write(
                f"  ⚠️  WARNING: Context size ~{stats['estimated_tokens']:,} tokens. "
                "Approaching limits."
            )

    def extract_code_blocks(self, response: str) -> list[str]:
        """
        Extract Python code blocks from the LLM response.

        Args:
            response: The LLM response text

        Returns:
            List of code strings from ```python``` blocks
        """
        return extract_python_blocks(response)

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

    def _build_evolution_memory(self) -> str:
        """
        Build the evolution memory block containing lineage map and scratchpad.

        This is shown to the Root LLM at the start of each generation to provide
        context about solution lineages and persistent notes.

        Returns:
            Formatted evolution memory string.
        """
        lineage_map = self.evolution_api._build_lineage_map()
        scratchpad = self.evolution_api.scratchpad

        lines = [
            "─" * 60,
            "## Evolution Memory",
            "",
            "### Lineage Map (auto-generated from parent_id tracking)",
            "",
            lineage_map if lineage_map else "(No trials yet)",
            "",
            "### Scratchpad (update with `update_scratchpad(content)`)",
            "",
            scratchpad if scratchpad else "(Empty - use update_scratchpad() to add persistent notes)",
            "",
            "─" * 60,
        ]
        return "\n".join(lines)

    def _build_generation_feedback_message(self) -> str:
        """Build a compact feedback message with results from the previous generation.

        Note: Full code is NOT included to reduce context size. The Root LLM can
        use get_trial_code(trial_ids) to retrieve specific code when needed, or
        use {{CODE_TRIAL_X_Y}} tokens in child prompts.
        """
        prev_gen = self.evolution_api.current_generation - 1
        if prev_gen < 0:
            return ""

        gen_summary = self.evolution_api.generations[prev_gen]
        trials = gen_summary.trials

        # Build compact feedback message
        lines = [
            f"Generation {prev_gen} complete. {len(trials)} children spawned.",
            "",
            "Results (use `get_trial_code([trial_ids])` to retrieve code, "
            "or `{{CODE_TRIAL_X_Y}}` tokens in child prompts):",
            "",
        ]

        # Sort by score
        sorted_trials = sorted(
            trials,
            key=lambda t: t.metrics.get("score", 0) if t.success else 0,
            reverse=True,
        )

        for trial in sorted_trials:
            score = trial.metrics.get("score", 0) if trial.success else 0
            valid = "valid" if trial.success else "INVALID"

            # Extract generation and trial number for code reference token
            parts = trial.trial_id.split("_")
            gen_num, trial_num = parts[1], parts[2]
            code_ref = f"{{{{CODE_TRIAL_{gen_num}_{trial_num}}}}}"

            lines.append(f"  - **{trial.trial_id}** [{valid}] score={score:.16f}")
            lines.append(f"    Code ref: {code_ref}")
            if trial.reasoning:
                # Truncate reasoning to first 150 chars
                reasoning_short = trial.reasoning[:150].replace("\n", " ")
                lines.append(f"    Approach: {reasoning_short}...")
            if not trial.success and trial.error:
                error_short = str(trial.error)[:100]
                lines.append(f"    Error: {error_short}")
            lines.append("")

        # Add evolution memory at the end
        lines.append("")
        lines.append(self._build_evolution_memory())
        lines.append("")
        lines.append(
            f"Continue to generation {self.evolution_api.current_generation}. "
            "Consider updating your scratchpad with insights from the previous generation."
        )

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
            key=lambda t: t.metrics.get("score", 0) if t.success else 0,
            reverse=True,
        )

        for i, trial in enumerate(sorted_trials, 1):
            score = trial.metrics.get("score", 0) if trial.success else 0
            valid_str = "valid" if trial.success else "INVALID"
            lines.append(f"{i}. **{trial.trial_id}** [{valid_str}]")
            lines.append(f"   Score: {score:.16f}")
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

    def _handle_trial_selection(
        self,
        system_prompt: list[dict],
        gen_pbar: Any,  # tqdm progress bar
        loop_iteration: int,
    ) -> SelectionResult:
        """
        Handle trial selection and generation advancement.

        Requests selection from Root LLM, advances generation, and
        sends feedback message.

        Args:
            system_prompt: The system prompt for LLM calls
            gen_pbar: The generation progress bar
            loop_iteration: The current loop iteration number

        Returns:
            SelectionResult indicating whether to break and why
        """
        # Request selection from Root LLM
        selection_request = self._build_selection_request_message()
        self._append_and_log("user", selection_request)

        # Get selection response from Root LLM
        tqdm.write("  └─ Requesting trial selection from Root LLM...")
        cached_messages = self._prepare_messages_with_caching(self.messages)
        selection_response = self.root_llm.generate(
            messages=cached_messages,
            system=system_prompt,
            max_tokens=2048,
            temperature=0.5,  # Lower temp for more deterministic selection
            enable_caching=False,
        )

        selection_message = selection_response.content
        self._append_and_log("assistant", selection_message)

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
            generations_completed = self.evolution_api.current_generation

            # Prune message history to prevent unbounded context growth
            self._prune_message_history(keep_recent_generations=2)
        except GenerationLimitError:
            # Reached max generations - generation was finalized before exception
            gen_pbar.update(1)
            return SelectionResult(
                should_break=True,
                termination_reason="max_generations_reached",
                generations_completed=self.evolution_api.max_generations,
            )

        # Build feedback message for next generation
        feedback = self._build_generation_feedback_message()
        self._append_and_log("user", feedback)

        return SelectionResult(
            should_break=False,
            generations_completed=generations_completed,
        )

    def _handle_no_children(
        self,
        execution_results: list[str],
        current_gen: int,
    ) -> list[tuple[int, str]]:
        """
        Handle case when no children were spawned in a generation.

        Checks for errors in execution results, logs warnings, and
        adds feedback to the conversation.

        Args:
            execution_results: List of REPL execution result strings
            current_gen: The current generation number

        Returns:
            List of (generation_number, error_message) tuples for any errors found
        """
        errors: list[tuple[int, str]] = []

        # Check for errors in execution results
        error_in_results = any(
            "Error" in r or "Exception" in r for r in execution_results
        )
        if error_in_results:
            # Extract error snippets for debugging
            error_snippets = [
                r[:200] for r in execution_results
                if "Error" in r or "Exception" in r
            ]
            error_msg = "; ".join(error_snippets)[:500]
            errors.append((self.evolution_api.current_generation, error_msg))
            tqdm.write(
                f"  [WARNING] Generation {current_gen} had errors in REPL: {error_msg[:100]}..."
            )

        # Build and send user message
        if execution_results:
            user_message = "Execution results:\n\n" + "\n\n---\n\n".join(
                execution_results
            )
        else:
            user_message = (
                "No children were spawned. Please spawn children using "
                "`spawn_children_parallel()` in a ```python``` code block."
            )

        self._append_and_log("user", user_message)
        return errors

    def _handle_generation_error(
        self,
        error: Exception,
        current_gen: int,
    ) -> tuple[int, str]:
        """
        Handle an error that occurred during generation execution.

        Logs the error, adds feedback to the conversation, and returns
        error info for tracking.

        Args:
            error: The exception that occurred
            current_gen: The generation number where the error occurred

        Returns:
            Tuple of (generation_number, error_message) for tracking
        """
        error_msg = f"{type(error).__name__}: {str(error)}"
        tb = traceback.format_exc()

        # Write to terminal
        tqdm.write(f"\n  [ERROR] Generation {current_gen} failed: {error_msg}")
        tqdm.write(f"  [ERROR] Traceback:\n{tb}")

        # Log the error for debugging (not added to conversation)
        self.logger.log_root_turn(
            turn_number=self.turn_number,
            role="system",
            content=f"GENERATION ERROR: {error_msg}\n\nTraceback:\n{tb}",
        )
        self.turn_number += 1

        # Add error feedback to conversation so Root LLM knows what happened
        error_feedback = (
            f"Error in generation {current_gen}: {error_msg}\n\n"
            "Please try a different approach in the next iteration."
        )
        self.messages.append({"role": "user", "content": error_feedback})

        return (current_gen, error_msg)

    def _append_and_log(
        self,
        role: str,
        content: str,
        code_executed: str | None = None,
        execution_result: str | None = None,
    ) -> None:
        """
        Append message to conversation and log it.

        This is a helper to consolidate the repeated pattern of:
        1. Appending a message to self.messages
        2. Logging it via self.logger.log_root_turn()
        3. Incrementing self.turn_number

        Args:
            role: Message role ("user", "assistant", or "system")
            content: Message content
            code_executed: Optional code that was executed (for REPL results)
            execution_result: Optional result of code execution
        """
        self.messages.append({"role": role, "content": content})
        self.logger.log_root_turn(
            turn_number=self.turn_number,
            role=role,
            content=content,
            code_executed=code_executed,
            execution_result=execution_result,
        )
        self.turn_number += 1

    def _execute_single_generation(
        self,
        loop_iteration: int,
        gen_pbar: Any,  # tqdm progress bar
        children_pbar: Any,  # tqdm progress bar
        update_pbar_postfix: Any,  # callback function
    ) -> GenerationResult:
        """
        Execute one generation of the evolution loop.

        This handles:
        1. Resetting progress bars
        2. Checking context health and budget
        3. Calling the Root LLM
        4. Executing code blocks from response
        5. Handling trial selection or no-children case

        Args:
            loop_iteration: The current loop iteration number
            gen_pbar: The generation progress bar
            children_pbar: The children progress bar
            update_pbar_postfix: Callback to update progress bar stats

        Returns:
            GenerationResult indicating whether to break, why, and any errors
        """
        current_gen = self.evolution_api.current_generation

        # Reset children progress bar for this generation
        children_pbar.reset()
        children_pbar.set_description(f"  Gen {current_gen} children")
        update_pbar_postfix()

        # Check context health at the start of each generation
        self._check_context_health()

        # Check budget before LLM call
        self.cost_tracker.raise_if_over_budget()

        # Get system prompt with current generation info
        system_prompt = self._get_system_prompt()

        # Call Root LLM with cached messages
        cached_messages = self._prepare_messages_with_caching(self.messages)
        response = self.root_llm.generate(
            messages=cached_messages,
            system=system_prompt,
            max_tokens=4096,
            temperature=0.7,
            enable_caching=False,
        )

        assistant_message = response.content
        self._append_and_log("assistant", assistant_message)

        # Extract and execute code blocks
        code_blocks = self.extract_code_blocks(assistant_message)
        execution_results = []

        for code in code_blocks:
            result = self.execute_code_in_repl(code)
            execution_results.append(f"```\n{code}\n```\n\nResult:\n{result}")

            # Log the execution (special case - only logs, doesn't append to messages)
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
            return GenerationResult(
                should_break=True,
                termination_reason=(
                    self.evolution_api._termination_reason or "evolution_terminated"
                ),
                generations_completed=self.evolution_api.current_generation,
            )

        # Automatically advance generation if children were spawned
        if self.evolution_api.has_children_in_current_generation():
            selection_result = self._handle_trial_selection(
                system_prompt, gen_pbar, loop_iteration
            )
            if selection_result.should_break:
                return GenerationResult(
                    should_break=True,
                    termination_reason=selection_result.termination_reason,
                    generations_completed=selection_result.generations_completed,
                )
            return GenerationResult(
                should_break=False,
                generations_completed=selection_result.generations_completed,
            )
        else:
            # No children spawned - handle and collect any errors
            errors = self._handle_no_children(execution_results, current_gen)
            return GenerationResult(
                should_break=False,
                errors=errors if errors else None,
            )

    def _build_result(
        self,
        termination_reason: str,
        generations_completed: int,
        generation_errors: list[tuple[int, str]],
    ) -> OrchestratorResult:
        """
        Build final orchestrator result after evolution completes.

        Handles:
        1. Appending error summary to termination reason
        2. Saving experiment data
        3. Getting best trial info
        4. Computing statistics

        Args:
            termination_reason: The base termination reason
            generations_completed: Number of generations that completed
            generation_errors: List of (gen_num, error_msg) tuples

        Returns:
            OrchestratorResult with final statistics
        """
        # If there were generation errors, append them to termination reason
        if generation_errors and "error:" not in termination_reason:
            error_summary = "; ".join(f"gen{g}: {e}" for g, e in generation_errors)
            termination_reason = f"{termination_reason} (with errors: {error_summary})"

        # Save experiment
        self.logger.log_cost_tracking(self.cost_tracker.to_dict())
        self.logger.save_experiment(
            termination_reason=termination_reason,
            scratchpad=self.evolution_api.scratchpad,
        )

        # Get best trial info
        best_trials = self.evolution_api._get_best_trials(n=1)
        best_program = best_trials[0]["code"] if best_trials else None
        best_score = best_trials[0]["metrics"].get("score", 0) if best_trials else 0

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
        0. Calibration phase (optional): Root LLM tests child models
        1. Root LLM is called once per generation
        2. Children are spawned via REPL execution
        3. Generation advances automatically after spawning
        4. Results are fed back for next generation

        Returns:
            OrchestratorResult with final statistics
        """
        # Calibration phase (before generation 0)
        if self.config.calibration_notes_file:
            # Load pre-existing notes - skip calibration
            self._load_calibration_notes(self.config.calibration_notes_file)
        elif self._has_calibration_budget():
            # Run calibration phase
            self._run_calibration_phase()
            # Save notes for potential reuse
            self._save_calibration_notes()
        else:
            # No calibration budget configured, skip calibration
            self.evolution_api.end_calibration_phase()

        # Build initial messages for evolution
        self.build_initial_messages()

        termination_reason = "max_generations_reached"
        generations_completed = 0
        loop_iterations = 0
        max_iterations = self.config.root_llm.max_iterations
        if max_iterations is None:
            max_iterations = self.evolution_api.max_generations
        generation_errors: list[tuple[int, str]] = []  # Track errors per generation

        # Create outer progress bar for generations
        gen_pbar = tqdm(
            total=self.evolution_api.max_generations,
            desc="Generations",
            unit="gen",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        # Create inner progress bar for children in current generation
        children_pbar = tqdm(
            total=self.evolution_api.max_children_per_generation,
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
                    t.metrics.get("score", 0)
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
                best=f"{best_score:.16f}" if best_score else "N/A",
                cost=f"${cost_summary.total_cost:.2f}",
            )

        try:
            while self.evolution_api.current_generation < self.evolution_api.max_generations:
                if max_iterations is not None and loop_iterations >= max_iterations:
                    termination_reason = "max_iterations_reached"
                    break

                loop_iterations += 1

                try:
                    gen_result = self._execute_single_generation(
                        loop_iterations, gen_pbar, children_pbar, update_pbar_postfix
                    )
                    if gen_result.should_break:
                        termination_reason = gen_result.termination_reason or termination_reason
                        generations_completed = (
                            gen_result.generations_completed
                            if gen_result.generations_completed is not None
                            else generations_completed
                        )
                        break
                    if gen_result.generations_completed is not None:
                        generations_completed = gen_result.generations_completed
                    if gen_result.errors:
                        generation_errors.extend(gen_result.errors)

                except BudgetExceededError:
                    # Re-raise to break the loop
                    raise
                except Exception as e:
                    # Handle error and continue to next generation
                    current_gen = self.evolution_api.current_generation
                    error_info = self._handle_generation_error(e, current_gen)
                    generation_errors.append(error_info)
                    continue

        except BudgetExceededError as e:
            termination_reason = f"budget_exceeded: {str(e)}"
        finally:
            children_pbar.close()
            gen_pbar.close()

        if generations_completed == 0:
            generations_completed = self.evolution_api.current_generation

        return self._build_result(
            termination_reason, generations_completed, generation_errors
        )
