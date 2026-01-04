# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MangoEvolve is an evolutionary code generation system that uses LLMs to iteratively develop optimization algorithms. A "Root LLM" orchestrates the evolution process via a REPL environment, spawning "Child LLMs" to generate candidate programs, evaluating them, and selecting the best to inform future generations.

**Current target**: Circle Packing - pack 26 circles into a unit square to maximize the sum of their radii (AlphaEvolve benchmark: 2.635).

## Commands

```bash
# Install dependencies
uv sync

# Run evolution with a config
uv run python -m mango_evolve --config configs/example_config_sonnet.yaml

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_evaluator.py -v

# Run a specific test
uv run pytest tests/test_evaluator.py::test_function_name -v

# Lint
uv run ruff check src tests

# Type check
uv run ty check
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

- **Root LLM Orchestrator** (`root_llm.py`): Manages the evolution loop, executes REPL code blocks
- **Evolution API** (`evolution_api.py`): Functions available to Root LLM (spawn_child_llm, evaluate_program, advance_generation, terminate_evolution)
- **REPL Environment** (`repl.py`): Python execution environment with Evolution API injected
- **LLM Providers** (`llm/providers/`): Pluggable LLM backends - Anthropic and OpenRouter supported
- **Evaluator** (`evaluation/`): Pluggable evaluation system loaded dynamically from config

### LLM Provider System

The system supports multiple LLM providers configured via YAML:

```yaml
root_llm:
  provider: "anthropic"  # or "openrouter"
  model: "claude-sonnet-4-20250514"
  reasoning:
    enabled: true  # Enable reasoning/thinking for supported models

child_llms:
  - alias: "fast"
    model: "google/gemini-3-flash-preview"
    provider: "openrouter"
    calibration_calls: 2
  - alias: "strong"
    model: "claude-sonnet-4-20250514"
    provider: "anthropic"

default_child_llm_alias: "fast"
```

OpenRouter configs support multiple child LLM definitions with aliases, allowing the Root LLM to select different models for different tasks.

### Adding a New Evaluator

Create a class with an `evaluate(code: str) -> dict` method in `src/mango_evolve/evaluation/`, then reference it in config:

```yaml
evaluation:
  evaluator_fn: "mango_evolve.evaluation.my_module:MyEvaluator"
  evaluator_kwargs:
    param1: value1
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Required for Anthropic provider
- `OPENROUTER_API_KEY`: Required for OpenRouter provider

## Test Patterns

- Tests use mock LLMs by default (no API key required)
- E2E tests in `test_e2e.py` require `ANTHROPIC_API_KEY`
- Test fixtures in `conftest.py` include `sample_config`, `sample_valid_packing_code`, etc.
