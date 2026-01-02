# MangoEvolve

LLM-driven evolutionary code generation for optimization problems. An open-source implementation inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) with a focus on **LLM-as-orchestrator** architecture.

## POC Results

MangoEvolve has been validated on the **Circle Packing** benchmark (pack 26 circles into a unit square to maximize the sum of their radii):

| Run | Best Score | Target | Result | Generations | Duration |
|-----|------------|--------|--------|-------------|----------|
| 1 | **2.6359831209** | 2.6359830990 | **Exceeded target** | 16 | ~1h 13m |
| 2 | **2.6359850561** | 2.6359830990 | **Exceeded target** | 16 | ~2h 20m |
| 3 | 2.6359830899 | 2.6359850561 | 99.999925% | 20 | ~2h 15m |

The system autonomously evolved from naive implementations (score ~2.08) to match and exceed the known optimal configuration, discovering techniques like:
- Hierarchical initialization patterns (4-4-4-14 layer structure)
- Multi-pass basin-hopping with different random seeds
- Analytical gradients with bisection refinement
- Sequential optimization with warm restarts

## Core Ideas

### Comparison with AlphaEvolve

**AlphaEvolve** (DeepMind) uses:
- A handcrafted evolutionary controller
- Fixed mutation/crossover operators
- LLMs as program generators only
- Specialized infrastructure (Gemini ecosystem)

**MangoEvolve** takes a different approach:
- **LLM-as-Orchestrator**: A "Root LLM" controls the entire evolution loop via a REPL environment
- **No hardcoded operators**: The Root LLM decides which trials to mutate, how to craft prompts, and when to try radically different approaches
- **Multi-model support**: Mix models from different providers (Anthropic, OpenAI, Google, etc.) via OpenRouter
- **Transparent reasoning**: All decisions are logged with the LLM's reasoning, making evolution interpretable
- **Simple, hackable codebase**: ~1200 lines of core code, easy to extend

### The REPL Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Root LLM (Orchestrator)                   │
│  - Receives evolution state and feedback                        │
│  - Decides strategy (exploration vs exploitation)               │
│  - Writes Python code blocks to control evolution               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Writes ```python``` blocks
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       REPL Environment                           │
│  Executes code with injected API:                               │
│  - spawn_children_parallel(children)  # Generate programs       │
│  - evaluate_program(code)             # Direct evaluation       │
│  - update_scratchpad(notes)           # Persistent memory       │
│  - get_trial_code(trial_ids)          # Retrieve past code      │
│  - terminate_evolution(reason)        # End evolution           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │  Child LLM │      │  Child LLM │      │  Child LLM │
    │  (Sonnet)  │      │  (Gemini)  │      │  (GPT-4.1) │
    └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Evaluator│        │ Evaluator│        │ Evaluator│
    │ (Sandbox)│        │ (Sandbox)│        │ (Sandbox)│
    └──────────┘        └──────────┘        └──────────┘
```

The Root LLM sees:
1. **Evolution memory**: Lineage map showing parent-child relationships and scores
2. **Scratchpad**: Persistent notes it can update across generations
3. **Generation feedback**: Results from previous trials with reasoning

Then it writes Python code to spawn the next generation, crafting prompts that include parent code (via `{{CODE_TRIAL_X_Y}}` tokens) and strategic instructions.

### Why This Matters

Traditional evolutionary algorithms require hand-tuning operators, selection pressure, and diversity mechanisms. MangoEvolve lets the LLM figure this out:

- **Stuck at a plateau?** The Root LLM might try larger perturbations or completely different initializations
- **Found a promising approach?** It can focus multiple children on refining that lineage
- **One model keeps failing?** It can switch to a different child LLM

This makes the system more adaptive without requiring algorithm expertise.

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mango-evolve.git
cd mango-evolve

# Install dependencies
uv sync
```

### API Keys

Set up your API keys based on which provider you want to use:

```bash
# For Anthropic models (Claude)
export ANTHROPIC_API_KEY=your_key_here

# For OpenRouter (access to multiple providers)
export OPENROUTER_API_KEY=your_key_here
```

## Quick Start

```bash
# Run evolution with the example config
uv run python -m mango_evolve --config configs/example_config_sonnet.yaml

# Run with a more powerful multi-model config
uv run python -m mango_evolve --config configs/openrouter_opus_thinking_mixed.yaml
```

## Configuration

MangoEvolve is configured via YAML files. Here's a complete reference:

### Basic Config (Single Model)

```yaml
experiment:
  name: "circle_packing_v1"
  output_dir: "./experiments"

# Root LLM orchestrates the evolution
root_llm:
  provider: "anthropic"
  model: "claude-sonnet-4-5-20250929"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0
  max_iterations: 30

# Child LLMs generate programs
child_llms:
  - alias: "sonnet"
    model: "claude-sonnet-4-5-20250929"
    provider: "anthropic"
    cost_per_million_input_tokens: 3.0
    cost_per_million_output_tokens: 15.0
    calibration_calls: 2  # Test calls before evolution

default_child_llm_alias: "sonnet"

# Evaluation - dynamically loads your evaluator
evaluation:
  evaluator_fn: "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator"
  evaluator_kwargs:
    n_circles: 26
    timeout_seconds: 300

evolution:
  max_generations: 10
  max_children_per_generation: 15

budget:
  max_total_cost: 20.0  # USD limit
```

### Advanced Config (Multi-Model with Reasoning)

```yaml
experiment:
  name: "shinka_evolve_circle_packing"
  output_dir: "./experiments"

# Root LLM with extended thinking for strategic decisions
root_llm:
  provider: "openrouter"
  model: "anthropic/claude-opus-4.5"
  cost_per_million_input_tokens: 5.0
  cost_per_million_output_tokens: 25.0
  max_iterations: 500
  reasoning:
    enabled: true
    effort: "xhigh"  # low, medium, high, xhigh

# Multiple child LLMs - Root can choose based on task
child_llms:
  # Strong model for complex mutations
  - alias: "opus"
    model: "anthropic/claude-opus-4.5"
    provider: "openrouter"
    cost_per_million_input_tokens: 5.0
    cost_per_million_output_tokens: 25.0
    calibration_calls: 5
    reasoning:
      enabled: true
      effort: "high"

  # Balanced quality/cost
  - alias: "sonnet"
    model: "anthropic/claude-sonnet-4.5"
    provider: "openrouter"
    cost_per_million_input_tokens: 3.0
    cost_per_million_output_tokens: 15.0
    calibration_calls: 5
    reasoning:
      enabled: true
      effort: "high"

  # Fast and cheap for exploration
  - alias: "gemini-flash"
    model: "google/gemini-3-flash-preview"
    provider: "openrouter"
    cost_per_million_input_tokens: 0.5
    cost_per_million_output_tokens: 3.0
    calibration_calls: 5
    reasoning:
      enabled: true

  # Very fast for high-volume exploration
  - alias: "grok-fast"
    model: "x-ai/grok-4.1-fast"
    provider: "openrouter"
    cost_per_million_input_tokens: 0.2
    cost_per_million_output_tokens: 0.5
    calibration_calls: 5

default_child_llm_alias: "gemini-flash"

evaluation:
  evaluator_fn: "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator"
  evaluator_kwargs:
    n_circles: 26
    timeout_seconds: 300

evolution:
  max_generations: 20
  max_children_per_generation: 16

budget:
  max_total_cost: 50.0
```

### Configuration Reference

| Section | Field | Type | Description |
|---------|-------|------|-------------|
| **experiment** | name | string | Experiment identifier (used in output directory) |
| | output_dir | string | Base directory for experiment outputs |
| **root_llm** | provider | string | `"anthropic"` or `"openrouter"` |
| | model | string | Model ID (e.g., `"claude-sonnet-4-5-20250929"` or `"anthropic/claude-opus-4.5"`) |
| | max_iterations | int | Maximum Root LLM turns (conversation limit) |
| | reasoning.enabled | bool | Enable extended thinking/reasoning |
| | reasoning.effort | string | Reasoning depth: `"low"`, `"medium"`, `"high"`, `"xhigh"` |
| **child_llms** | alias | string | Name to reference this model (e.g., `"fast"`, `"strong"`) |
| | model | string | Model ID for this child |
| | provider | string | Provider for this child |
| | calibration_calls | int | Test calls allowed during calibration phase |
| | reasoning | object | Same as root_llm.reasoning |
| **evaluation** | evaluator_fn | string | Module path to evaluator class |
| | evaluator_kwargs | dict | Arguments passed to evaluator constructor |
| **evolution** | max_generations | int | Maximum evolution generations |
| | max_children_per_generation | int | Max trials spawned per generation |
| **budget** | max_total_cost | float | Maximum total spend in USD |

## Adding a Custom Evaluator

Create a class with an `evaluate(code: str) -> dict` method:

```python
# src/mango_evolve/evaluation/my_problem.py
import subprocess
import tempfile

class MyEvaluator:
    def __init__(self, timeout_seconds: int = 60, **kwargs):
        self.timeout = timeout_seconds

    def evaluate(self, code: str) -> dict:
        """
        Evaluate generated code.

        Args:
            code: Python code string to evaluate

        Returns:
            dict with at least:
            - valid: bool - whether the code ran successfully
            - score: float - optimization objective (higher is better)
            - error: str | None - error message if invalid
        """
        try:
            # Run code in isolated subprocess
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()

                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

            if result.returncode != 0:
                return {
                    "valid": False,
                    "score": 0.0,
                    "error": result.stderr[:500]
                }

            # Parse score from stdout (your format)
            score = float(result.stdout.strip())

            return {
                "valid": True,
                "score": score,
                "error": None
            }

        except subprocess.TimeoutExpired:
            return {"valid": False, "score": 0.0, "error": "Timeout"}
        except Exception as e:
            return {"valid": False, "score": 0.0, "error": str(e)}
```

Then reference it in your config:

```yaml
evaluation:
  evaluator_fn: "mango_evolve.evaluation.my_problem:MyEvaluator"
  evaluator_kwargs:
    timeout_seconds: 120
```

## Output Structure

Each experiment creates a timestamped directory:

```
experiments/circle_packing_opus_thinking_mixed_20251231_135159/
├── config.json              # Configuration snapshot
├── experiment.json          # Final summary with best trial, termination reason
├── root_llm_log.jsonl       # Full Root LLM conversation (turns, code, results)
├── cost_tracking.json       # Token usage and costs per model
├── calibration_notes.yaml   # Notes from calibration phase (if any)
└── generations/
    ├── gen_0/
    │   ├── summary.json     # Generation stats, selections, reasoning
    │   ├── scratchpad.md    # Root LLM's notes at this generation
    │   ├── trial_0_0.json   # Individual trial (code, metrics, prompt)
    │   ├── trial_0_1.json
    │   └── ...
    ├── gen_1/
    │   └── ...
    └── ...
```

## Development

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_evaluator.py::test_function_name -v

# Run E2E tests (requires API key)
ANTHROPIC_API_KEY=your_key uv run pytest tests/test_e2e.py -v

# Lint
uv run ruff check src tests

# Type check
uv run ty check
```

## How It Works (Technical Details)

### Evolution Loop

1. **Calibration Phase** (optional): Root LLM tests child models with trial prompts to understand their capabilities
2. **Generation 0**: Root LLM spawns diverse initial children exploring different strategies
3. **Selection**: After each generation, Root LLM selects 2-5 trials to carry forward based on:
   - **Performance**: Best scores
   - **Diversity**: Different approaches worth exploring
   - **Potential**: Promising ideas that might improve with refinement
4. **Mutation**: Root LLM crafts prompts for child LLMs, often including parent code via `{{CODE_TRIAL_X_Y}}` tokens
5. **Repeat**: Until termination (max generations, budget, or explicit termination)

### Key Mechanisms

- **Scratchpad**: Root LLM maintains persistent notes across generations (`update_scratchpad()`)
- **Lineage Tracking**: Parent-child relationships are tracked for mutation inheritance
- **Parallel Spawning**: Children run in parallel via multiprocessing for efficiency
- **Prompt Caching**: OpenRouter/Anthropic prompt caching reduces costs for repeated system prompts
- **Context Pruning**: Old generation messages are summarized to prevent context overflow

### API Functions Available to Root LLM

```python
# Spawn multiple children in parallel (preferred)
spawn_children_parallel([
    {"prompt": "...", "parent_id": "trial_0_1", "model": "sonnet"},
    {"prompt": "...", "parent_id": "trial_0_3", "model": "gemini-flash"},
])

# Spawn single child (for calibration or sequential)
spawn_child_llm(prompt, parent_id=None, model=None, temperature=0.7)

# Evaluate code directly (without child LLM)
evaluate_program(code_string)

# Retrieve code from specific trials
get_trial_code(["trial_0_1", "trial_1_3"])

# Get top trials across all generations
get_top_trials(n=5)

# Update persistent notes
update_scratchpad("## Key Insights\n- Basin hopping works better...")

# End calibration phase
end_calibration_phase()

# Terminate evolution early
terminate_evolution("Found optimal solution", best_code)
```

## Example Configs

See the `configs/` directory for ready-to-use configurations:

- `example_config_sonnet.yaml` - Simple single-model setup (Anthropic)
- `example_config_haiku.yaml` - Fast, cheap exploration
- `openrouter_opus_thinking_mixed.yaml` - Multi-model with reasoning (best results)
- `shinka_evolve_circle_packing.yaml` - Production config with diverse models

## License

MIT

## Acknowledgments

- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind) for the original evolutionary program generation concept
- [Recursive LLM](https://arxiv.org/abs/2502.07503) for hierarchical LLM spawning ideas
