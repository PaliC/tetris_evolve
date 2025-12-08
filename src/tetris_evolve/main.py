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

from .config import load_config  # noqa: E402
from .exceptions import BudgetExceededError, ConfigValidationError  # noqa: E402
from .root_llm import RootLLMOrchestrator  # noqa: E402


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="tetris_evolve",
        description="LLM-driven evolutionary code generation for circle packing",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args(args)


def main(args=None):
    """
    Main entry point for the CLI.

    Args:
        args: Optional list of command line arguments (for testing)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parsed_args = parse_args(args)

    # Load configuration
    config_path = Path(parsed_args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = load_config(config_path)
    except ConfigValidationError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 1

    if parsed_args.verbose:
        print(f"Loaded configuration from: {config_path}")
        print(f"Experiment: {config.experiment.name}")
        print(f"Output directory: {config.experiment.output_dir}")
        print(f"Budget: ${config.budget.max_total_cost:.2f}")
        print(f"Max iterations: {config.root_llm.max_iterations}")

    # Run the orchestrator
    try:
        orchestrator = RootLLMOrchestrator(config)

        if parsed_args.verbose:
            print(f"\nExperiment directory: {orchestrator.logger.base_dir}")
            print("Starting evolution...")
            print("-" * 50)

        result = orchestrator.run()

        # Print results
        print("\n" + "=" * 50)
        print("EVOLUTION COMPLETE")
        print("=" * 50)
        print(f"Termination reason: {result.reason}")
        print(f"Iterations: {result.num_generations}")
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


if __name__ == "__main__":
    sys.exit(main())
