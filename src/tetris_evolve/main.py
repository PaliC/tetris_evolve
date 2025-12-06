"""
CLI entry point for tetris_evolve.

Runs evolutionary experiments for circle packing optimization.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file before any other imports that might use env vars
load_dotenv()

from .config import load_config
from .exceptions import BudgetExceededError, ConfigValidationError
from .resume import analyze_experiment
from .root_llm import RootLLMOrchestrator


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="tetris_evolve",
        description="LLM-driven evolutionary code generation for circle packing",
    )

    # Create mutually exclusive group for main actions
    action_group = parser.add_mutually_exclusive_group(required=True)

    action_group.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file (start new experiment)",
    )

    action_group.add_argument(
        "--resume",
        "-r",
        type=str,
        metavar="EXPERIMENT_DIR",
        help="Path to experiment directory to resume (restarts current generation)",
    )

    action_group.add_argument(
        "--analyze",
        "-a",
        type=str,
        metavar="EXPERIMENT_DIR",
        help="Analyze experiment directory and show resumption info",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args(args)


def run_analyze(experiment_dir: str, verbose: bool = False) -> int:
    """
    Analyze an experiment directory and display resumption info.

    Args:
        experiment_dir: Path to experiment directory
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    experiment_path = Path(experiment_dir)

    if not experiment_path.exists():
        print(f"Error: Experiment directory not found: {experiment_path}", file=sys.stderr)
        return 1

    try:
        info = analyze_experiment(experiment_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid experiment state: {e}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("EXPERIMENT ANALYSIS")
    print("=" * 60)
    print(info)
    print("=" * 60)

    # Provide actionable suggestions
    print("\nResume Options:")
    if info.can_resume:
        print(f"  --resume {experiment_path}")
        print(f"      Restart generation {info.current_generation} from scratch")
    else:
        if info.current_generation >= info.total_generations_configured:
            print("  Experiment is complete (all generations finished)")
        else:
            print("  No resume options available")

    return 0


def run_resume(experiment_dir: str, verbose: bool = False) -> int:
    """
    Resume an interrupted experiment by restarting the current generation.

    Args:
        experiment_dir: Path to experiment directory
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    experiment_path = Path(experiment_dir)

    if not experiment_path.exists():
        print(f"Error: Experiment directory not found: {experiment_path}", file=sys.stderr)
        return 1

    try:
        # Analyze first to show info
        info = analyze_experiment(experiment_path)

        if verbose:
            print("=" * 50)
            print("RESUMING EXPERIMENT")
            print("=" * 50)
            print(info)
            print("-" * 50)

        # Create orchestrator from resume
        orchestrator = RootLLMOrchestrator.from_resume(
            experiment_dir=experiment_path,
        )

        if verbose:
            print(f"Experiment directory: {orchestrator.logger.base_dir}")
            print("Starting evolution...")
            print("-" * 50)

        result = orchestrator.run()

        # Print results
        print("\n" + "=" * 50)
        print("EVOLUTION COMPLETE")
        print("=" * 50)
        print(f"Termination reason: {result.reason}")
        print(f"Generations: {result.num_generations}")
        print(f"Total trials: {result.total_trials}")
        print(f"Successful trials: {result.successful_trials}")
        print(f"Best score: {result.best_score:.4f}")

        if result.cost_summary:
            print(f"Total cost: ${result.cost_summary.get('total_cost', 0):.4f}")

        print(f"\nResults saved to: {orchestrator.logger.base_dir}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except BudgetExceededError as e:
        print(f"Error: Budget exceeded: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


def run_new_experiment(config_path: str, verbose: bool = False) -> int:
    """
    Run a new experiment from configuration.

    Args:
        config_path: Path to YAML configuration file
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}", file=sys.stderr)
        return 1

    try:
        config = load_config(config_file)
    except ConfigValidationError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 1

    if verbose:
        print(f"Loaded configuration from: {config_file}")
        print(f"Experiment: {config.experiment.name}")
        print(f"Output directory: {config.experiment.output_dir}")
        print(f"Budget: ${config.budget.max_total_cost:.2f}")
        print(f"Max generations: {config.evolution.max_generations}")

    # Run the orchestrator
    try:
        orchestrator = RootLLMOrchestrator(config)

        if verbose:
            print(f"\nExperiment directory: {orchestrator.logger.base_dir}")
            print("Starting evolution...")
            print("-" * 50)

        result = orchestrator.run()

        # Print results
        print("\n" + "=" * 50)
        print("EVOLUTION COMPLETE")
        print("=" * 50)
        print(f"Termination reason: {result.reason}")
        print(f"Generations: {result.num_generations}")
        print(f"Total trials: {result.total_trials}")
        print(f"Successful trials: {result.successful_trials}")
        print(f"Best score: {result.best_score:.4f}")

        if result.cost_summary:
            print(f"Total cost: ${result.cost_summary.get('total_cost', 0):.4f}")

        print(f"\nResults saved to: {orchestrator.logger.base_dir}")

        return 0

    except BudgetExceededError as e:
        print(f"Error: Budget exceeded: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


def main(args=None):
    """
    Main entry point for the CLI.

    Args:
        args: Optional list of command line arguments (for testing)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parsed_args = parse_args(args)

    if parsed_args.analyze:
        return run_analyze(parsed_args.analyze, parsed_args.verbose)
    elif parsed_args.resume:
        return run_resume(parsed_args.resume, parsed_args.verbose)
    else:
        return run_new_experiment(parsed_args.config, parsed_args.verbose)


if __name__ == "__main__":
    sys.exit(main())
