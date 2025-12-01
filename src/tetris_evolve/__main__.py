"""CLI entry point for tetris_evolve."""

import argparse
import sys
from pathlib import Path

from .evolution.controller import EvolutionConfig, EvolutionController


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tetris_evolve",
        description="LLM-driven evolutionary code generation for Tetris players.",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--resume",
        "-r",
        type=Path,
        help="Path to experiment directory to resume",
    )

    parser.add_argument(
        "--max-generations",
        type=int,
        help="Maximum number of generations (overrides config)",
    )

    parser.add_argument(
        "--max-cost",
        type=float,
        help="Maximum cost in USD (overrides config)",
    )

    parser.add_argument(
        "--max-time",
        type=int,
        help="Maximum time in minutes (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory for experiments (overrides config)",
    )

    parser.add_argument(
        "--root-model",
        type=str,
        help="Model for root LLM (overrides config)",
    )

    parser.add_argument(
        "--child-model",
        type=str,
        help="Model for child LLM (overrides config)",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version and exit",
    )

    return parser


def load_config(args: argparse.Namespace) -> EvolutionConfig:
    """Load configuration from file and CLI overrides.

    Args:
        args: Parsed command line arguments.

    Returns:
        EvolutionConfig with all settings applied.
    """
    if args.config:
        config = EvolutionConfig.from_yaml(args.config)
    else:
        config = EvolutionConfig()

    # Apply CLI overrides
    if args.max_generations is not None:
        config.max_generations = args.max_generations
    if args.max_cost is not None:
        config.max_cost_usd = args.max_cost
    if args.max_time is not None:
        config.max_time_minutes = args.max_time
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.root_model is not None:
        config.root_model = args.root_model
    if args.child_model is not None:
        config.child_model = args.child_model

    return config


def run_evolution(config: EvolutionConfig, resume_dir: Path | None = None) -> int:
    """Run the evolution loop.

    Args:
        config: Evolution configuration.
        resume_dir: Optional experiment directory to resume from.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    controller = EvolutionController(config)

    print(f"Starting evolution with config:")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Max cost: ${config.max_cost_usd:.2f}")
    print(f"  Max time: {config.max_time_minutes} minutes")
    print(f"  Root model: {config.root_model}")
    print(f"  Child model: {config.child_model}")
    print(f"  Output dir: {config.output_dir}")
    print()

    try:
        if resume_dir:
            print(f"Resuming from: {resume_dir}")
            result = controller.resume(resume_dir)
        else:
            result = controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during evolution: {e}", file=sys.stderr)
        return 1

    # Print results
    print()
    print("=" * 50)
    print("Evolution Complete")
    print("=" * 50)
    print(f"Experiment ID: {result.experiment_id}")
    print(f"Generations completed: {result.generations_completed}")
    print(f"Total cost: ${result.total_cost_usd:.2f}")
    print(f"Total time: {result.total_time_minutes:.1f} minutes")
    print(f"Termination reason: {result.termination_reason}")

    if result.best_trial_id:
        print()
        print(f"Best trial: {result.best_trial_id}")
        print(f"Best score: {result.best_score:.1f}")
        print()
        print("Best code preview:")
        print("-" * 50)
        if result.best_code:
            # Show first 500 chars of code
            preview = result.best_code[:500]
            if len(result.best_code) > 500:
                preview += "\n... (truncated)"
            print(preview)
        print("-" * 50)
    else:
        print("\nNo successful trials found.")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(f"tetris_evolve {__version__}")
        return 0

    if args.resume and args.config:
        print("Warning: Both --config and --resume specified. Config file settings will apply to resumed run.")

    config = load_config(args)
    return run_evolution(config, args.resume)


if __name__ == "__main__":
    sys.exit(main())
