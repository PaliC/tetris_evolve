# MangoEvolve

LLM-driven evolutionary code generation for optimization problems.

## Results

MangoEvolve exceeded the DeepMind AlphaEvolve benchmark on the circle packing problem:

| Run | Best Score | vs AlphaEvolve (2.635) | Generations |
|-----|------------|------------------------|-------------|
| 1 | **2.6359850561** | +0.00098 (new record) | 16 |
| 2 | 2.6359831209 | +0.00098 | 16 |
| 3 | 2.6359830899 | +0.00098 | 20 |

**Configuration**: Claude Opus 4.5 root with extended thinking (xhigh), mixed child LLMs (Opus 4.5, Sonnet 4, Gemini 3 Flash). Budget: $50 max per run.

The winning algorithms used sequential basinhopping with SLSQP optimization, discovering that different random seeds escape different local optima.

## How It Works

```
Root LLM (REPL) --> spawn_children() --> Child LLM --> Code
      │                                                  │
      │                                                  ▼
      │                                        evaluate_program()
      │                                                  │
      ◀──────────── metrics, reasoning ◀─────────────────┘
      │
      ▼
advance_generation() / terminate_evolution()
```

A Root LLM orchestrates evolution via a Python REPL with an injected Evolution API. It spawns Child LLMs to generate candidate programs, evaluates them, selects the best, and iterates. The Root LLM writes Python code to control the process—it can try different models, resurrect old trials, cross-pollinate approaches, and adapt strategy based on results.

### Comparison to Related Work

| | AlphaEvolve | ShinkaEvolve | MangoEvolve |
|-|-------------|--------------|-------------|
| **Core Idea** | LLM-based evolutionary program search | ? | REPL-based Root LLM orchestrating evolution |
| **Root Control** | Prompt-based | ? | Python REPL with Evolution API |
| **Model Selection** | Single model | ? | Multi-provider, Root chooses per-task |
| **Trial History** | Current generation | ? | Full history access + resurrection |

MangoEvolve's key differentiator is giving the Root LLM a Python REPL with full control over the evolution process. Instead of following a fixed evolutionary algorithm, the Root LLM can write code to:

- Spawn specific child models for different tasks (`spawn_children([{"prompt": prompt, "model": "opus"}])`)
- Access any historical trial (`trials["trial_2_5"].code`)
- Resurrect promising old approaches
- Combine techniques across lineages
- Terminate early when satisfied

## Quick Start

```bash
# Install
git clone <repository-url> && cd mango_evolve
uv sync

# Set API keys (as needed)
export ANTHROPIC_API_KEY=your_key
export OPENROUTER_API_KEY=your_key  # for multi-provider configs

# Run
uv run python -m mango_evolve --config configs/example_config_sonnet.yaml
```

## Configuration

Configs are YAML files. Here's a minimal example:

```yaml
experiment:
  name: "my_experiment"

root_llm:
  model: "claude-sonnet-4-5-20250929"
  provider: "anthropic"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0
  max_iterations: 30

child_llms:
  - alias: "sonnet"
    model: "claude-sonnet-4-5-20250929"
    provider: "anthropic"
    cost_per_million_input_tokens: 3.0
    cost_per_million_output_tokens: 15.0

default_child_llm_alias: "sonnet"

evaluation:
  evaluator_fn: "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator"
  evaluator_kwargs:
    n_circles: 26
    timeout_seconds: 300

evolution:
  max_generations: 10
  max_children_per_generation: 15

budget:
  max_total_cost: 20.0
```

### Multi-Provider with Extended Thinking

For best results, use OpenRouter with multiple models and extended thinking:

```yaml
root_llm:
  provider: "openrouter"
  model: "anthropic/claude-opus-4.5"
  cost_per_million_input_tokens: 5.0
  cost_per_million_output_tokens: 25.0
  max_iterations: 500
  reasoning:
    enabled: true
    effort: "xhigh"

child_llms:
  - alias: "opus"
    model: "anthropic/claude-opus-4.5"
    provider: "openrouter"
    cost_per_million_input_tokens: 5.0
    cost_per_million_output_tokens: 25.0
    reasoning:
      enabled: true
      effort: "high"

  - alias: "gemini-flash"
    model: "google/gemini-3-flash-preview"
    provider: "openrouter"
    cost_per_million_input_tokens: 0.50
    cost_per_million_output_tokens: 3.00
    reasoning:
      enabled: true
      effort: "high"

default_child_llm_alias: "gemini-flash"

evolution:
  max_generations: 20
  max_children_per_generation: 16

budget:
  max_total_cost: 50.0
```

The `reasoning.effort` levels are: `minimal`, `low`, `medium`, `high`, `xhigh`.

### Configuration Reference

| Section | Field | Description |
|---------|-------|-------------|
| `experiment.name` | string | Experiment identifier |
| `root_llm.model` | string | Model ID (e.g., `claude-sonnet-4-5-20250929`) |
| `root_llm.provider` | string | `anthropic` or `openrouter` |
| `root_llm.reasoning.enabled` | bool | Enable extended thinking (OpenRouter) |
| `root_llm.reasoning.effort` | string | Thinking level: minimal/low/medium/high/xhigh |
| `child_llms` | list | Array of child LLM configs |
| `child_llms[].alias` | string | Name for Root LLM to reference this model |
| `default_child_llm_alias` | string | Which child to use when not specified |
| `evaluation.evaluator_fn` | string | `module.path:ClassName` format |
| `evolution.max_generations` | int | Max generations (default: 10) |
| `evolution.max_children_per_generation` | int | Max trials per gen (default: 10) |
| `budget.max_total_cost` | float | Max USD spend (default: 20.0) |

See `configs/` for more examples.

## Adding Custom Evaluators

Create a class with an `evaluate(code: str) -> dict` method:

```python
# src/mango_evolve/evaluation/my_evaluator.py
class MyEvaluator:
    def __init__(self, **kwargs):
        self.config = kwargs

    def evaluate(self, code: str) -> dict:
        # Execute code, return metrics
        return {'valid': True, 'score': 42.0, 'error': None}
```

Reference in config:

```yaml
evaluation:
  evaluator_fn: "mango_evolve.evaluation.my_evaluator:MyEvaluator"
  evaluator_kwargs:
    param1: value1
```

## Development

```bash
uv run pytest                    # Run tests (mock LLMs, no API key needed)
uv run pytest tests/test_e2e.py  # E2E tests (requires ANTHROPIC_API_KEY)
uv run ruff check src tests      # Lint
uv run ty check                  # Type check
```

## License

MIT
