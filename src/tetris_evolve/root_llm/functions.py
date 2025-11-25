"""
Root LLM REPL functions.

This module provides the functions that are injected into the shared REPL
for the Root LLM to use during evolution.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ..database import Program, ProgramDatabase, ProgramMetrics
from ..environment.base import EnvironmentConfig
from ..evaluation import evaluate_player_from_code, EvaluationResult
from ..rlm import SharedREPL


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""
    pass


@dataclass
class PerformanceAnalysis:
    """Analysis of a generation's performance."""
    generation: int
    num_programs: int
    avg_score: float
    std_score: float
    max_score: float
    min_score: float
    top_programs: List[Program]
    improvement_from_prev: Optional[float] = None
    diversity_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "num_programs": self.num_programs,
            "avg_score": self.avg_score,
            "std_score": self.std_score,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "top_program_ids": [p.program_id for p in self.top_programs],
            "improvement_from_prev": self.improvement_from_prev,
            "diversity_score": self.diversity_score,
        }


@dataclass
class HistoricalTrends:
    """Historical performance trends across generations."""
    generations: Dict[int, Dict[str, Any]]
    best_scores: List[float]
    avg_scores: List[float]
    improvement_rate: float
    convergence_indicator: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generations": self.generations,
            "best_scores": self.best_scores,
            "avg_scores": self.avg_scores,
            "improvement_rate": self.improvement_rate,
            "convergence_indicator": self.convergence_indicator,
        }


class RootLLMFunctions:
    """
    Functions available to the Root LLM in the shared REPL.

    These functions provide the Root LLM with capabilities to:
    - Spawn child RLLMs for code generation
    - Evaluate programs
    - Analyze performance
    - Manage generations
    - Control evolution lifecycle
    """

    def __init__(
        self,
        program_database: ProgramDatabase,
        env_config: EnvironmentConfig,
        max_generations: int = 100,
        max_child_rllms_per_generation: int = 50,
        episodes_per_evaluation: int = 10,
    ):
        """
        Initialize Root LLM functions.

        Args:
            program_database: Database for storing programs
            env_config: Environment configuration
            max_generations: Hard limit on generations
            max_child_rllms_per_generation: Hard limit on RLLMs per generation
            episodes_per_evaluation: Default episodes for evaluation
        """
        self.program_database = program_database
        self.env_config = env_config

        # Hard limits (cannot be modified)
        self._max_generations = max_generations
        self._max_child_rllms_per_generation = max_child_rllms_per_generation

        # Soft parameters (can be modified)
        self.episodes_per_evaluation = episodes_per_evaluation

        # State
        self.current_generation = 0
        self.rllms_spawned_this_generation = 0
        self.is_terminated = False
        self.selected_for_generation: Dict[int, List[str]] = {}

        # RLLM handler (set externally)
        self._rllm_handler: Optional[Callable[[str], str]] = None

    def set_rllm_handler(self, handler: Callable[[str], str]) -> None:
        """Set the handler for spawning RLLMs."""
        self._rllm_handler = handler

    def inject_into_repl(self, repl: SharedREPL) -> None:
        """
        Inject all functions and context into a REPL.

        Args:
            repl: SharedREPL to inject into
        """
        # Inject functions
        repl.inject_function("spawn_rllm", self.spawn_rllm)
        repl.inject_function("evaluate_program", self.evaluate_program)
        repl.inject_function("get_performance_analysis", self.get_performance_analysis)
        repl.inject_function("get_historical_trends", self.get_historical_trends)
        repl.inject_function("advance_generation", self.advance_generation)
        repl.inject_function("terminate_evolution", self.terminate_evolution)
        repl.inject_function("modify_parameters", self.modify_parameters)

        # Inject context variables
        repl.set_variable("program_database", self.program_database)
        repl.set_variable("current_generation", self.current_generation)
        repl.set_variable("env_config", self.env_config)

    def spawn_rllm(self, prompt: str) -> str:
        """
        Spawn a Recursive Child LLM with a custom prompt.

        Args:
            prompt: Complete prompt for the Child RLLM

        Returns:
            Result from Child RLLM (typically generated code)

        Raises:
            ResourceLimitError: If max RLLMs per generation exceeded
            RuntimeError: If evolution is terminated or no handler set
        """
        self._check_not_terminated()

        if self.rllms_spawned_this_generation >= self._max_child_rllms_per_generation:
            raise ResourceLimitError(
                f"Maximum RLLMs per generation ({self._max_child_rllms_per_generation}) exceeded"
            )

        if self._rllm_handler is None:
            raise RuntimeError("No RLLM handler set. Call set_rllm_handler() first.")

        self.rllms_spawned_this_generation += 1
        return self._rllm_handler(prompt)

    def evaluate_program(
        self,
        code: str,
        num_games: Optional[int] = None,
        store_result: bool = False,
        program_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run environment simulations and return metrics.

        Args:
            code: Player code to evaluate
            num_games: Number of games to play (defaults to episodes_per_evaluation)
            store_result: Whether to store result in database
            program_id: Program ID if storing result

        Returns:
            Metrics dictionary with performance data
        """
        if num_games is None:
            num_games = self.episodes_per_evaluation

        result = evaluate_player_from_code(
            code,
            self.env_config,
            num_episodes=num_games,
        )

        # Build result dict
        # FIXED: Proper fallback logic - use env_metrics["score"] if available,
        # otherwise fall back to generic_metrics["total_reward"]
        score_metric = result.env_metrics.get("score") or result.generic_metrics.get("total_reward")
        metrics_dict = {
            "success": result.code_errors == 0,
            "avg_score": score_metric.mean if score_metric else 0.0,
            "std_score": score_metric.std if score_metric else 0.0,
            "max_score": score_metric.max if score_metric else 0.0,
            "min_score": score_metric.min if score_metric else 0.0,
            "games_played": len(result.episode_results),
            "code_errors": result.code_errors,
        }

        if result.code_errors > 0:
            metrics_dict["error"] = result.error_messages[0] if result.error_messages else "Unknown error"

        # Store in database if requested
        if store_result and program_id:
            program = self.program_database.get_program(program_id)
            if program is None:
                program = Program(
                    program_id=program_id,
                    generation=self.current_generation,
                    code=code,
                )
                self.program_database.add_program(program)

            # FIXED: Proper fallback logic for lines_cleared
            lines_metric = result.env_metrics.get("lines_cleared") or result.generic_metrics.get("episode_length")
            program.metrics = ProgramMetrics(
                avg_score=metrics_dict["avg_score"],
                std_score=metrics_dict["std_score"],
                avg_lines_cleared=lines_metric.mean if lines_metric else 0.0,
                games_played=metrics_dict["games_played"],
                max_score=metrics_dict["max_score"],
                min_score=metrics_dict["min_score"],
            )

        return metrics_dict

    def get_performance_analysis(
        self,
        generation: Optional[int] = None
    ) -> PerformanceAnalysis:
        """
        Get detailed analysis of a generation's performance.

        Args:
            generation: Generation to analyze (defaults to current)

        Returns:
            PerformanceAnalysis with statistics and insights
        """
        if generation is None:
            generation = self.current_generation

        programs = self.program_database.get_programs_by_generation(generation)
        programs_with_metrics = [p for p in programs if p.metrics is not None]

        if not programs_with_metrics:
            return PerformanceAnalysis(
                generation=generation,
                num_programs=len(programs),
                avg_score=0.0,
                std_score=0.0,
                max_score=0.0,
                min_score=0.0,
                top_programs=[],
            )

        scores = [p.metrics.avg_score for p in programs_with_metrics]

        # Calculate improvement from previous generation
        improvement = None
        if generation > 0:
            prev_programs = self.program_database.get_programs_by_generation(generation - 1)
            prev_with_metrics = [p for p in prev_programs if p.metrics is not None]
            if prev_with_metrics:
                prev_avg = np.mean([p.metrics.avg_score for p in prev_with_metrics])
                curr_avg = np.mean(scores)
                if prev_avg > 0:
                    improvement = (curr_avg - prev_avg) / prev_avg

        # Get top programs
        top_programs = self.program_database.get_top_programs(generation=generation, n=5)

        return PerformanceAnalysis(
            generation=generation,
            num_programs=len(programs),
            avg_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            max_score=float(np.max(scores)),
            min_score=float(np.min(scores)),
            top_programs=top_programs,
            improvement_from_prev=improvement,
        )

    def get_historical_trends(self) -> HistoricalTrends:
        """
        Get performance trends across all generations.

        Returns:
            HistoricalTrends with improvement rates and convergence indicators
        """
        generations_data = {}
        best_scores = []
        avg_scores = []

        for gen in range(self.current_generation + 1):
            programs = self.program_database.get_programs_by_generation(gen)
            programs_with_metrics = [p for p in programs if p.metrics is not None]

            if programs_with_metrics:
                scores = [p.metrics.avg_score for p in programs_with_metrics]
                gen_data = {
                    "num_programs": len(programs),
                    "avg_score": float(np.mean(scores)),
                    "max_score": float(np.max(scores)),
                    "min_score": float(np.min(scores)),
                    "std_score": float(np.std(scores)),
                }
                generations_data[gen] = gen_data
                best_scores.append(gen_data["max_score"])
                avg_scores.append(gen_data["avg_score"])
            else:
                generations_data[gen] = {"num_programs": len(programs)}
                best_scores.append(0.0)
                avg_scores.append(0.0)

        # Calculate improvement rate (average improvement per generation)
        # FIXED: Handle case when starting from 0 - use absolute improvement if initial score is 0
        improvement_rate = 0.0
        if len(best_scores) >= 2:
            if best_scores[0] > 0:
                # Percentage improvement rate
                improvement_rate = (best_scores[-1] - best_scores[0]) / (best_scores[0] * len(best_scores))
            elif best_scores[-1] > 0:
                # Starting from 0 - report absolute improvement per generation
                improvement_rate = best_scores[-1] / len(best_scores)

        # Calculate convergence indicator (how much scores have stabilized)
        convergence_indicator = 0.0
        if len(best_scores) >= 3:
            recent_changes = [
                abs(best_scores[i] - best_scores[i-1])
                for i in range(max(1, len(best_scores)-3), len(best_scores))
            ]
            if best_scores[-1] > 0:
                convergence_indicator = 1.0 - (np.mean(recent_changes) / best_scores[-1])

        return HistoricalTrends(
            generations=generations_data,
            best_scores=best_scores,
            avg_scores=avg_scores,
            improvement_rate=improvement_rate,
            convergence_indicator=max(0.0, convergence_indicator),
        )

    def advance_generation(self, selected_program_ids: List[str]) -> int:
        """
        Move to next generation with selected programs.

        Args:
            selected_program_ids: IDs of programs to carry forward

        Returns:
            New generation number

        Raises:
            ResourceLimitError: If max generations exceeded
        """
        self._check_not_terminated()

        if self.current_generation >= self._max_generations:
            raise ResourceLimitError(
                f"Maximum generations ({self._max_generations}) exceeded"
            )

        # Record selection
        self.selected_for_generation[self.current_generation + 1] = selected_program_ids

        # Advance
        self.current_generation += 1
        self.rllms_spawned_this_generation = 0

        return self.current_generation

    def terminate_evolution(
        self,
        reason: str,
        best_program_id: str
    ) -> Dict[str, Any]:
        """
        Terminate the evolution process.

        Args:
            reason: Explanation for termination
            best_program_id: ID of the final best program

        Returns:
            Summary of evolution results
        """
        self.is_terminated = True

        best_program = self.program_database.get_program(best_program_id)
        final_score = best_program.metrics.avg_score if best_program and best_program.metrics else 0.0

        return {
            "reason": reason,
            "best_program_id": best_program_id,
            "final_score": final_score,
            "total_generations": self.current_generation + 1,
            "total_programs": len(self.program_database.get_all_programs()),
        }

    def modify_parameters(self, param_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically modify evolution parameters.

        Args:
            param_updates: Dictionary of parameter changes

        Returns:
            Current parameters after modification

        Raises:
            ValueError: If trying to modify hard limits
        """
        # Check for hard limits
        hard_limits = {"max_generations", "max_child_rllms_per_generation"}
        for key in param_updates:
            if key in hard_limits:
                raise ValueError(f"Cannot modify hard limit: {key}")

        # Apply updates
        if "episodes_per_evaluation" in param_updates:
            self.episodes_per_evaluation = param_updates["episodes_per_evaluation"]

        # Return current params
        return {
            "episodes_per_evaluation": self.episodes_per_evaluation,
            "max_generations": self._max_generations,
            "max_child_rllms_per_generation": self._max_child_rllms_per_generation,
        }

    def _check_not_terminated(self) -> None:
        """Raise error if evolution has been terminated."""
        if self.is_terminated:
            raise RuntimeError("Evolution has been terminated")
