# LLM Tetris Optimizer

An evolutionary code generation system that uses LLMs to discover optimal Tetris-playing strategies. Combines ideas from [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (evolutionary code optimization) and [Recursive LLMs](https://alexzhang13.github.io/blog/2025/rlm/) (LLM-driven decision making).

## Overview

This system uses a **Root LLM** to orchestrate an evolutionary process where **Child LLMs** generate Tetris-playing code. The Root LLM:
1. Spawns Child LLMs to generate code
2. Evaluates code against Tetris games
3. Decides which strategies to evolve further
4. Terminates when satisfied

Unlike traditional evolutionary algorithms, the selection of which trials advance is decided by an LLM rather than algorithmic rules.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Root LLM                          │
│  (orchestrates evolution via REPL environment)       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  REPL Functions:                                     │
│  ├── spawn_child_llm(prompt) → code + reasoning      │
│  ├── evaluate_program(code) → metrics                │
│  ├── advance_generation(selected_ids) → gen_num      │
│  └── terminate_evolution(reason) → summary           │
│                                                      │
└──────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌────────────────────┐
│   Child LLMs    │        │  Tetris Evaluator  │
│ (code generation│        │    (PufferLib/     │
│   per trial)    │        │  Tetris-Gymnasium) │
└─────────────────┘        └────────────────────┘
```

## Documentation

- **[Design Document](docs/DESIGN.md)** - Comprehensive technical design with architecture, APIs, and detailed TODO list
- **[AlphaEvolve Summary](docs/research/alphaevolve_summary.md)** - Technical summary of AlphaEvolve concepts
- **[RLM Summary](docs/research/rlm_summary.md)** - Technical summary of Recursive LLM concepts

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run an experiment
tetris-evolve run config.yaml

# Resume an interrupted experiment
tetris-evolve resume ./results/experiment_123

# Generate a report
tetris-evolve report ./results/experiment_123
```

## Configuration

```yaml
# config.yaml
experiment_name: "tetris_evolution_v1"
output_dir: "./results"

root_lm:
  model: "claude-sonnet-4-20250514"
  cost:
    input: 0.003
    output: 0.015

child_lm:
  model: "claude-sonnet-4-20250514"
  cost:
    input: 0.003
    output: 0.015

evolution:
  max_generations: 20
  max_children_per_generation: 10

max_cost: 50.00

initial_prompt: |
  You are an evolutionary algorithm controller...
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run validation experiments
python experiments/validate_design.py
```

## Key Concepts

### From AlphaEvolve
- **Evolutionary loop**: Generate → Evaluate → Select → Repeat
- **Program database**: Store and track all trials across generations
- **LLM ensemble**: Different models for different purposes

### From Recursive LLMs
- **REPL environment**: LLM controls execution via Python code blocks
- **LLM-driven decisions**: The LLM decides how to decompose the problem
- **Context management**: Information flows through REPL variables

### Our Innovations
- **LLM-driven selection**: Root LLM decides which trials advance (vs. algorithmic selection)
- **Pluggable REPL functions**: System is extensible by adding new functions
- **Comprehensive observability**: Full logging of prompts, responses, and costs

## Project Status

**Phase**: Design Complete

See the [detailed TODO list](docs/DESIGN.md#detailed-todo-list) for implementation roadmap.

## License

MIT
