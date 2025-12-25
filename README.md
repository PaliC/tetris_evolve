# MangoEvolve

LLM-driven evolutionary code generation for optimization problems.

## Overview

This project combines ideas from:
- **AlphaEvolve** (DeepMind): Evolutionary program generation using LLMs
- **Recursive LLMs (RLM)**: Hierarchical LLM spawning with REPL environments

A "Root LLM" orchestrates an evolutionary process, spawning "Child LLMs" to generate candidate programs, evaluating them, and iteratively improving the best candidates.

### Current Target: Circle Packing

Pack 26 circles into a unit square to maximize the sum of their radii.

- **Benchmark**: AlphaEvolve achieved 2.635
- **Best PoC result**: 2.08 (hexagonal packing)
- **Evaluation**: Deterministic, fast (~60ms per trial)

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
# Clone the repository
git clone <repository-url>
cd mango_evolve

# Install dependencies
uv sync
```

### Set up API Key

To run with real LLMs, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Quick Start

### Run with Default Configuration

```bash
# Run the evolution with the example config
uv run python -m mango_evolve --config configs/example_config_sonnet.yaml

# Run with verbose output
uv run python -m mango_evolve --config configs/example_config_haiku.yaml --verbose
```

### Run PoC Examples (No API Key Required)

```bash
# Circle packing full integration (recommended)
uv run python experiments/poc_circle_packing_integration.py

# Core component tests
uv run python experiments/poc_repl.py
uv run python experiments/poc_cost_tracker.py
uv run python experiments/poc_evaluator.py
```

### Run Tests

```bash
# Run all tests (uses mock LLMs)
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_integration.py -v

# Run E2E tests with real LLM (requires API key)
ANTHROPIC_API_KEY=your_key uv run pytest tests/test_e2e.py -v
```

## Architecture

```
Root LLM (REPL) --> spawn_child_llm() --> Child LLM --> Code
        |                                               |
        |                                               v
        |                                        evaluate_program()
        |                                               |
        <----------- metrics, reasoning <---------------+
        |
        v
  advance_generation() / terminate_evolution()
```

### Key Components

- **Root LLM Orchestrator**: Manages the evolution loop, executes REPL code
- **Evolution API**: Functions available to Root LLM (spawn, evaluate, advance, terminate)
- **REPL Environment**: Python execution environment with Evolution API injected
- **Cost Tracker**: Token usage tracking with budget enforcement
- **Evaluator**: Pluggable evaluation system (currently Circle Packing)
- **Logger**: Structured logging for experiments, generations, and trials

## Configuration Reference

Create a YAML configuration file (see `configs/example_config.yaml`):

```yaml
# Experiment settings
experiment:
  name: "circle_packing_v1"    # Experiment name (used in output directory)
  output_dir: "./experiments"   # Where to save results

# Root LLM configuration (orchestrator)
root_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0
  max_iterations: 30            # Maximum conversation turns

# Child LLM configuration (program generators)
child_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0

# Evaluation configuration
evaluation:
  evaluator_fn: "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator"
  evaluator_kwargs:
    n_circles: 26               # Number of circles to pack
    timeout_seconds: 30         # Max evaluation time per trial

# Evolution settings
evolution:
  max_generations: 10           # Maximum generations
  max_children_per_generation: 10

# Budget settings
budget:
  max_total_cost: 20.0          # Maximum spend in USD
```

### Configuration Options

| Section | Field | Type | Description |
|---------|-------|------|-------------|
| experiment | name | string | Experiment identifier |
| experiment | output_dir | string | Output directory for logs |
| root_llm | model | string | Anthropic model ID |
| root_llm | cost_per_million_input_tokens | float | Cost per million input tokens (USD) |
| root_llm | cost_per_million_output_tokens | float | Cost per million output tokens (USD) |
| root_llm | max_iterations | int | Max conversation turns |
| child_llm | model | string | Anthropic model ID |
| child_llm | cost_per_million_input_tokens | float | Cost per million input tokens (USD) |
| child_llm | cost_per_million_output_tokens | float | Cost per million output tokens (USD) |
| evaluation | evaluator_fn | string | Module path to evaluator class |
| evaluation | evaluator_kwargs | dict | Arguments for evaluator constructor |
| evolution | max_generations | int | Maximum evolution generations |
| evolution | max_children_per_generation | int | Max trials per generation |
| budget | max_total_cost | float | Maximum total cost (USD) |

## API Reference

### Evolution API Functions

These functions are available to the Root LLM in ```repl``` code blocks:

#### `spawn_child_llm(prompt: str, parent_id: str = None) -> dict`

Spawn a child LLM to generate a program.

```python
result = spawn_child_llm("Write a circle packing algorithm")
# Returns: {
#   'trial_id': 'trial_0_0',
#   'code': '...',
#   'metrics': {'valid': True, 'score': 2.08, ...},
#   'success': True,
#   'error': None
# }
```

#### `evaluate_program(code: str) -> dict`

Evaluate code directly without spawning a child LLM.

```python
metrics = evaluate_program(code_string)
# Returns: {'valid': True, 'score': 2.08, ...}
```

#### `advance_generation(selected_trial_ids: list, reasoning: str) -> int`

Move to the next generation with selected trials as parents.

```python
new_gen = advance_generation(['trial_0_1'], 'Best score in generation')
# Returns: 1 (new generation number)
```

#### `terminate_evolution(reason: str, best_program: str = None) -> dict`

End the evolution and return final results.

```python
result = terminate_evolution('Found satisfactory solution', best_code)
# Returns: {'terminated': True, 'reason': '...', 'num_generations': 3, ...}
```

## Output Structure

When running an experiment, results are saved to:

```
{output_dir}/{experiment_name}_{timestamp}/
├── config.json              # Configuration snapshot
├── experiment.json          # Experiment summary
├── root_llm_log.jsonl       # Root LLM conversation
├── cost_tracking.json       # Token usage and costs
└── generations/
    ├── gen_0/
    │   ├── summary.json     # Generation summary
    │   ├── trial_0_0.json   # Individual trial
    │   └── trial_0_1.json
    └── gen_1/
        └── ...
```

## Development

### Project Structure

```
src/mango_evolve/
├── __init__.py
├── config.py              # Configuration loading
├── cost_tracker.py        # Budget enforcement
├── logger.py              # Experiment logging
├── repl.py                # REPL environment
├── evolution_api.py       # Evolution API
├── root_llm.py            # Root orchestrator
├── main.py                # CLI entry point
├── exceptions.py          # Custom exceptions
├── evaluation/
│   ├── __init__.py
│   └── circle_packing.py  # Circle packing evaluator
├── llm/
│   ├── __init__.py
│   ├── client.py          # Anthropic wrapper
│   └── prompts.py         # System prompts
└── utils/
    ├── __init__.py
    └── code_extraction.py # Code parsing utilities

tests/
├── conftest.py            # Test fixtures
├── test_config.py
├── test_cost_tracker.py
├── test_evaluator.py
├── test_evolution_api.py
├── test_root_llm.py
├── test_integration.py    # Integration tests
└── test_e2e.py            # E2E tests (require API key)
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=mango_evolve

# Run specific test categories
uv run pytest tests/test_integration.py -v
uv run pytest -k "not e2e"  # Skip E2E tests
```

### Adding a New Evaluator

1. Create a new evaluator class in `src/mango_evolve/evaluation/`:

```python
class MyEvaluator:
    def __init__(self, **kwargs):
        self.config = kwargs

    def evaluate(self, code: str) -> dict:
        # Run the code and return metrics
        return {
            'valid': True,
            'score': 42.0,
            'error': None,
        }
```

2. Update your config to use the new evaluator:

```yaml
evaluation:
  evaluator_fn: "mango_evolve.evaluation.my_module:MyEvaluator"
  evaluator_kwargs:
    param1: value1
```

## Documentation

- [Design Document](docs/DESIGN.md) - Detailed architecture
- [Circle Packing Design](docs/DESIGN_CIRCLE_PACKING.md) - Current target problem
- [Implementation TODO](docs/IMPLEMENTATION_TODO.md) - Task list with dependencies

## Example Results (from PoC)

```
Testing Circle Packing Evolution Pipeline...
============================================================

Generation 0: Exploring diverse strategies...
  gen0_trial0: VALID, score=1.7687  (grid)
  gen0_trial1: VALID, score=2.0800  (hexagonal)
  gen0_trial2: INVALID, score=0.0000 (greedy - failed)
  gen0_trial3: VALID, score=1.6512  (corner-first)
  gen0_trial4: VALID, score=1.0795  (concentric)

Best strategy: hexagonal
  Sum of radii: 2.0800
```

## License

MIT
