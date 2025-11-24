"""
Evolution loop implementation.

This module implements the main evolution loop that coordinates all
components: Root LLM, Child RLLMs, evaluation, and persistence.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import time
import yaml

from ..database import Program, ProgramDatabase, ProgramMetrics
from ..environment.base import EnvironmentConfig
from ..environment.tetris import TetrisConfig
from ..evaluation import evaluate_player_from_code
from ..root_llm import RootLLMFunctions
from ..child_llm import ChildLLMExecutor, LLMClient
from ..rlm import SharedREPL


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    max_generations: int = 100
    max_child_rllms_per_generation: int = 50
    episodes_per_evaluation: int = 10
    initial_population_size: int = 10
    selection_size: int = 5
    max_time_minutes: int = 240

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvolutionConfig":
        """Create from dictionary."""
        return cls(
            max_generations=d.get("max_generations", 100),
            max_child_rllms_per_generation=d.get("max_child_rllms_per_generation", 50),
            episodes_per_evaluation=d.get("episodes_per_evaluation", 10),
            initial_population_size=d.get("initial_population_size", 10),
            selection_size=d.get("selection_size", 5),
            max_time_minutes=d.get("max_time_minutes", 240),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "EvolutionConfig":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        evolution_data = data.get("evolution", data)
        return cls.from_dict(evolution_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_generations": self.max_generations,
            "max_child_rllms_per_generation": self.max_child_rllms_per_generation,
            "episodes_per_evaluation": self.episodes_per_evaluation,
            "initial_population_size": self.initial_population_size,
            "selection_size": self.selection_size,
            "max_time_minutes": self.max_time_minutes,
        }


@dataclass
class GenerationSummary:
    """Summary of a generation's evolution."""
    generation: int
    num_programs: int
    best_score: float
    avg_score: float
    std_score: float = 0.0
    duration_seconds: float = 0.0
    rllms_spawned: int = 0
    selected_program_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "num_programs": self.num_programs,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
            "std_score": self.std_score,
            "duration_seconds": self.duration_seconds,
            "rllms_spawned": self.rllms_spawned,
            "selected_program_ids": self.selected_program_ids,
        }


class EvolutionLogger:
    """
    Handles logging for evolution runs.

    Creates structured logs following the directory structure
    defined in the design document.
    """

    def __init__(self, run_dir: Path):
        """
        Initialize logger.

        Args:
            run_dir: Directory for this evolution run
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.run_dir / "generations").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # Open log files
        self._evolution_log = self.run_dir / "evolution_log.jsonl"

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an evolution event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data,
        }
        with open(self._evolution_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_generation_summary(self, summary: GenerationSummary) -> None:
        """Log a generation summary."""
        gen_dir = self.run_dir / "generations" / f"gen_{summary.generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        with open(gen_dir / "generation_summary.json", "w") as f:
            json.dump(summary.to_dict(), f, indent=2)

        self.log_event("generation_complete", summary.to_dict())

    def save_program(self, program: Program) -> None:
        """Save a program with all its data."""
        gen_dir = self.run_dir / "generations" / f"gen_{program.generation:03d}"
        prog_dir = gen_dir / "programs" / program.program_id
        prog_dir.mkdir(parents=True, exist_ok=True)

        # Save code
        with open(prog_dir / "code.py", "w") as f:
            f.write(program.code)

        # Save metadata
        with open(prog_dir / "metadata.json", "w") as f:
            json.dump(program.to_dict(), f, indent=2)

    def create_checkpoint(self, generation: int, state: Dict[str, Any]) -> None:
        """Create a checkpoint."""
        checkpoint_path = self.run_dir / "checkpoints" / f"checkpoint_gen_{generation:03d}.json"

        with open(checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)

        self.log_event("checkpoint_created", {"generation": generation})


class EvolutionRunner:
    """
    Main evolution runner.

    Orchestrates the evolution process including:
    - Population initialization
    - Generation cycles
    - Selection and mutation
    - Logging and checkpointing
    """

    def __init__(
        self,
        config: EvolutionConfig,
        env_config: EnvironmentConfig,
        llm_client: LLMClient,
        run_dir: Path,
        database: Optional[ProgramDatabase] = None,
    ):
        """
        Initialize evolution runner.

        Args:
            config: Evolution configuration
            env_config: Environment configuration
            llm_client: LLM client for code generation
            run_dir: Directory for run output
            database: Optional existing database
        """
        self.config = config
        self.env_config = env_config
        self.llm_client = llm_client
        self.run_dir = Path(run_dir)

        # Initialize components
        self.database = database or ProgramDatabase()
        self.logger = EvolutionLogger(run_dir)

        # Create child LLM executor
        self.child_executor = ChildLLMExecutor(
            llm_client=llm_client,
            env_config=env_config,
        )

        # Create Root LLM functions
        self.root_functions = RootLLMFunctions(
            program_database=self.database,
            env_config=env_config,
            max_generations=config.max_generations,
            max_child_rllms_per_generation=config.max_child_rllms_per_generation,
            episodes_per_evaluation=config.episodes_per_evaluation,
        )

        # Set up RLLM handler
        self.root_functions.set_rllm_handler(self.child_executor.create_handler())

        # State
        self.current_generation = 0
        self.best_scores: List[float] = []
        self.start_time: Optional[float] = None

        # Log config
        self.logger.log_event("run_start", {"config": config.to_dict()})

    @property
    def is_time_exceeded(self) -> bool:
        """Check if time limit exceeded."""
        if self.start_time is None:
            return False
        elapsed_minutes = (time.time() - self.start_time) / 60
        return elapsed_minutes >= self.config.max_time_minutes

    def initialize_population(self) -> None:
        """Create initial population of programs."""
        self.logger.log_event("population_init_start", {
            "size": self.config.initial_population_size
        })

        # Generate initial diverse programs
        for i in range(self.config.initial_population_size):
            # Generate code
            prompt = self._get_initial_prompt(i)
            result = self.child_executor.execute(prompt)

            if result.success and result.code:
                program_id = self.database.generate_program_id()

                # Evaluate
                eval_result = evaluate_player_from_code(
                    result.code,
                    self.env_config,
                    num_episodes=self.config.episodes_per_evaluation,
                )

                # Create program
                score = 0.0
                if "score" in eval_result.env_metrics:
                    score = eval_result.env_metrics["score"].mean

                program = Program(
                    program_id=program_id,
                    generation=0,
                    code=result.code,
                    metrics=ProgramMetrics(
                        avg_score=score,
                        std_score=eval_result.env_metrics.get("score", eval_result.generic_metrics["total_reward"]).std if eval_result.env_metrics.get("score") else 0,
                        avg_lines_cleared=eval_result.env_metrics.get("lines_cleared", eval_result.generic_metrics["episode_length"]).mean if eval_result.env_metrics.get("lines_cleared") else 0,
                        games_played=len(eval_result.episode_results),
                    )
                )

                self.database.add_program(program)
                self.logger.save_program(program)

        self.logger.log_event("population_init_complete", {
            "programs_created": len(self.database.get_programs_by_generation(0))
        })

    def _get_initial_prompt(self, index: int) -> str:
        """Get prompt for initial population member."""
        strategies = [
            "Create a basic Tetris player that always hard drops",
            "Create a Tetris player that tries to keep the stack low",
            "Create a Tetris player that minimizes holes",
            "Create a Tetris player that clears lines efficiently",
            "Create a Tetris player with lookahead capability",
        ]
        strategy = strategies[index % len(strategies)]

        return f"""
{strategy}

The player should implement:
- select_action(observation) -> int (0-5)
- reset() -> None

Actions: 0=left, 1=right, 2=rotate_cw, 3=rotate_ccw, 4=soft_drop, 5=hard_drop

Return complete Python code for a TetrisPlayer class.
"""

    def run_generation(self) -> GenerationSummary:
        """Run a single generation of evolution."""
        gen_start = time.time()
        generation = self.current_generation

        self.logger.log_event("generation_start", {"generation": generation})

        # Get current programs
        programs = self.database.get_programs_by_generation(generation)
        programs_with_metrics = [p for p in programs if p.metrics]

        if not programs_with_metrics:
            # No programs to evolve from
            return GenerationSummary(
                generation=generation,
                num_programs=0,
                best_score=0.0,
                avg_score=0.0,
            )

        # Select top programs
        top_programs = sorted(
            programs_with_metrics,
            key=lambda p: p.metrics.avg_score,
            reverse=True
        )[:self.config.selection_size]

        # Generate new programs from top performers
        rllms_spawned = 0
        new_programs = []

        for parent in top_programs:
            # Generate mutation prompts
            prompts = self._get_mutation_prompts(parent)

            for prompt in prompts:
                if rllms_spawned >= self.config.max_child_rllms_per_generation:
                    break

                result = self.child_executor.execute(
                    prompt,
                    parent_code=parent.code,
                )
                rllms_spawned += 1

                if result.success and result.code:
                    # Evaluate new program
                    eval_result = evaluate_player_from_code(
                        result.code,
                        self.env_config,
                        num_episodes=self.config.episodes_per_evaluation,
                    )

                    score = 0.0
                    if "score" in eval_result.env_metrics:
                        score = eval_result.env_metrics["score"].mean

                    program = Program(
                        program_id=self.database.generate_program_id(),
                        generation=generation + 1,
                        code=result.code,
                        parent_ids=[parent.program_id],
                        metrics=ProgramMetrics(
                            avg_score=score,
                            std_score=eval_result.env_metrics.get("score", eval_result.generic_metrics["total_reward"]).std if eval_result.env_metrics.get("score") else 0,
                            avg_lines_cleared=eval_result.env_metrics.get("lines_cleared", eval_result.generic_metrics["episode_length"]).mean if eval_result.env_metrics.get("lines_cleared") else 0,
                            games_played=len(eval_result.episode_results),
                        )
                    )

                    self.database.add_program(program)
                    self.logger.save_program(program)
                    new_programs.append(program)

        # Calculate statistics
        all_scores = [p.metrics.avg_score for p in programs_with_metrics]
        best_score = max(all_scores) if all_scores else 0.0
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        self.best_scores.append(best_score)

        # Create summary
        summary = GenerationSummary(
            generation=generation,
            num_programs=len(programs),
            best_score=best_score,
            avg_score=avg_score,
            duration_seconds=time.time() - gen_start,
            rllms_spawned=rllms_spawned,
            selected_program_ids=[p.program_id for p in top_programs],
        )

        self.logger.log_generation_summary(summary)

        # Advance generation
        self.current_generation += 1
        self.root_functions.current_generation = self.current_generation

        return summary

    def _get_mutation_prompts(self, parent: Program) -> List[str]:
        """Get mutation prompts for a parent program."""
        return [
            f"Improve this Tetris player to score higher. Current score: {parent.metrics.avg_score if parent.metrics else 0}",
            f"Optimize hole management in this Tetris player",
        ]

    def run(self) -> Dict[str, Any]:
        """
        Run the full evolution loop.

        Returns:
            Dictionary with evolution results
        """
        self.start_time = time.time()

        # Initialize if needed
        if not self.database.get_programs_by_generation(0):
            self.initialize_population()

        # Run generations
        while (
            self.current_generation < self.config.max_generations
            and not self.is_time_exceeded
            and not self.root_functions.is_terminated
        ):
            self.run_generation()

            # Checkpoint every 5 generations
            if self.current_generation % 5 == 0:
                self.save_checkpoint()

        # Final results
        all_programs = self.database.get_all_programs()
        best_program = max(
            [p for p in all_programs if p.metrics],
            key=lambda p: p.metrics.avg_score,
            default=None
        )

        result = {
            "success": True,
            "generations_completed": self.current_generation,
            "total_programs": len(all_programs),
            "best_program_id": best_program.program_id if best_program else None,
            "best_score": best_program.metrics.avg_score if best_program and best_program.metrics else 0.0,
            "best_scores": self.best_scores,
            "duration_seconds": time.time() - self.start_time,
        }

        self.logger.log_event("run_complete", result)
        self.save_checkpoint()

        return result

    def save_checkpoint(self) -> None:
        """Save current state as checkpoint."""
        state = {
            "generation": self.current_generation,
            "config": self.config.to_dict(),
            "best_scores": self.best_scores,
        }
        self.logger.create_checkpoint(self.current_generation, state)
        self.database.save(self.run_dir / "database")

    @classmethod
    def resume(
        cls,
        run_dir: Path,
        llm_client: LLMClient,
        env_config: Optional[EnvironmentConfig] = None,
    ) -> "EvolutionRunner":
        """
        Resume evolution from a checkpoint.

        Args:
            run_dir: Directory of the run to resume
            llm_client: LLM client
            env_config: Environment config (loaded from checkpoint if not provided)

        Returns:
            EvolutionRunner ready to continue
        """
        run_dir = Path(run_dir)

        # Find latest checkpoint
        checkpoints = sorted(run_dir.glob("checkpoints/checkpoint_gen_*.json"))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {run_dir}")

        with open(checkpoints[-1]) as f:
            state = json.load(f)

        # Load config
        config = EvolutionConfig.from_dict(state["config"])

        # Load database
        database = ProgramDatabase.load(run_dir / "database")

        # Create runner
        if env_config is None:
            env_config = TetrisConfig()

        runner = cls(
            config=config,
            env_config=env_config,
            llm_client=llm_client,
            run_dir=run_dir,
            database=database,
        )

        # Restore state
        runner.current_generation = state["generation"]
        runner.best_scores = state.get("best_scores", [])
        runner.root_functions.current_generation = runner.current_generation

        return runner
