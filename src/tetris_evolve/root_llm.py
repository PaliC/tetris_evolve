"""
Root LLM Orchestrator for tetris_evolve.

Main orchestrator that runs the Root LLM evolution loop, executing REPL
code blocks and managing the conversation with the Root LLM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .config import Config, load_evaluator
from .cost_tracker import CostTracker
from .evolution_api import EvolutionAPI
from .exceptions import BudgetExceededError
from .llm.client import LLMClient, MockLLMClient
from .llm.prompts import get_root_system_prompt
from .logger import ExperimentLogger
from .repl import REPLEnvironment
from .utils.code_extraction import extract_repl_blocks


@dataclass
class OrchestratorResult:
    """Result of running the orchestrator."""

    terminated: bool
    reason: str
    num_iterations: int
    best_program: Optional[str] = None
    best_score: float = 0.0
    total_trials: int = 0
    successful_trials: int = 0
    cost_summary: Optional[Dict[str, Any]] = None


class RootLLMOrchestrator:
    """
    Main orchestrator for the Root LLM evolution loop.

    The orchestrator:
    1. Initializes all components (LLM clients, REPL, Evolution API, etc.)
    2. Builds the initial conversation with system prompt
    3. Runs the conversation loop:
       - Send messages to Root LLM
       - Extract and execute ```repl``` code blocks
       - Add results back to conversation
       - Check for termination conditions
    4. Returns final results
    """

    def __init__(
        self,
        config: Config,
        root_llm: Optional[Union[LLMClient, MockLLMClient]] = None,
        child_llm: Optional[Union[LLMClient, MockLLMClient]] = None,
        logger: Optional[ExperimentLogger] = None,
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
        self.max_iterations = config.root_llm.max_iterations or 30

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

        # Initialize Evolution API
        self.evolution_api = EvolutionAPI(
            evaluator=self.evaluator,
            child_llm=self.child_llm,
            cost_tracker=self.cost_tracker,
            logger=self.logger,
            max_generations=config.evolution.max_generations,
            max_children_per_generation=config.evolution.max_children_per_generation,
        )

        # Initialize REPL with Evolution API functions
        self.repl = REPLEnvironment(api_functions=self.evolution_api.get_api_functions())

        # Conversation state
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = get_root_system_prompt()
        self.turn_number = 0

    def build_initial_messages(self) -> List[Dict[str, str]]:
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
                    "Begin the evolution process. Start by exploring different "
                    "circle packing strategies to understand what works well."
                ),
            }
        ]
        return self.messages

    def extract_code_blocks(self, response: str) -> List[str]:
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

    def check_termination(self, response: str) -> bool:
        """
        Check if the evolution has been terminated.

        Args:
            response: The LLM response text

        Returns:
            True if evolution was terminated
        """
        return self.evolution_api.is_terminated

    def run(self) -> OrchestratorResult:
        """
        Run the Root LLM evolution loop.

        Returns:
            OrchestratorResult with final statistics
        """
        # Build initial messages
        self.build_initial_messages()

        termination_reason = "max_iterations_reached"
        iteration = 0

        try:
            for iteration in range(self.max_iterations):
                # Check budget before LLM call
                try:
                    self.cost_tracker.raise_if_over_budget()
                except BudgetExceededError as e:
                    termination_reason = f"budget_exceeded: {str(e)}"
                    break

                # Call Root LLM
                response = self.root_llm.generate(
                    messages=self.messages,
                    system=self.system_prompt,
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
                        content=f"REPL execution result",
                        code_executed=code,
                        execution_result=result,
                    )
                    self.turn_number += 1

                # Check if evolution was terminated
                if self.check_termination(assistant_message):
                    termination_reason = (
                        self.evolution_api._termination_reason or "evolution_terminated"
                    )
                    break

                # Add execution results as user message for next iteration
                if execution_results:
                    user_message = "Execution results:\n\n" + "\n\n---\n\n".join(
                        execution_results
                    )
                else:
                    user_message = (
                        "No REPL code was executed. Please continue with the "
                        "evolution process using ```repl``` code blocks."
                    )

                self.messages.append({"role": "user", "content": user_message})

                # Log the user turn
                self.logger.log_root_turn(
                    turn_number=self.turn_number,
                    role="user",
                    content=user_message,
                )
                self.turn_number += 1

        except BudgetExceededError as e:
            termination_reason = f"budget_exceeded: {str(e)}"

        # Save experiment
        self.logger.log_cost_tracking(self.cost_tracker.to_dict())
        self.logger.save_experiment(termination_reason=termination_reason)

        # Get best trial info
        best_trials = self.evolution_api._get_best_trials(n=1)
        best_program = best_trials[0]["code"] if best_trials else None
        best_score = (
            best_trials[0]["metrics"].get("sum_radii", 0) if best_trials else 0
        )

        # Compute statistics
        all_trials = self.evolution_api.all_trials
        total_trials = len(all_trials)
        successful_trials = sum(1 for t in all_trials.values() if t.success)

        return OrchestratorResult(
            terminated=True,
            reason=termination_reason,
            num_iterations=iteration + 1,
            best_program=best_program,
            best_score=best_score,
            total_trials=total_trials,
            successful_trials=successful_trials,
            cost_summary=self.cost_tracker.get_summary().__dict__,
        )
