"""
Prompt templates for mango_evolve.

Contains the Root LLM system prompt that documents available functions
and guides the evolution process. Structured for optimal prompt caching.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ChildLLMConfig

# Root LLM System Prompt - Static Prefix (cacheable)
# This contains all the stable content that doesn't change between calls.
# Dynamic values (generation counts) are appended separately.

ROOT_LLM_SYSTEM_PROMPT_STATIC = '''You are an expert algorithm designer. You are orchestrating an evolutionary process to develop algorithms for circle packing.

## Problem

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

The best known solution is 2.6359850561146603. Aim to achieve as high of a score as possible.

## Code Format

Child LLMs must produce code with:
```python
import numpy as np

def construct_packing():
    """Returns (centers, radii, sum_radii) where:
    - centers: np.array shape (26, 2)
    - radii: np.array shape (26,)
    - sum_radii: float
    """
    pass

def run_packing():
    return construct_packing()
```

## Available Functions

### spawn_children(children: list[dict]) -> list[TrialView]
Spawn child LLMs in parallel. Each child dict has:
- `prompt` (str, required) - Use `trials["trial_X_Y"].code` to include code in your prompt
- `parent_id` (str, optional) - set to track lineage when improving a trial
- `model` (str, optional) - alias from available child LLMs
- `temperature` (float, optional, default 0.7)
Returns list of TrialView objects with: trial_id, code, score, success, reasoning, error, etc.

Example:
```python
# Improve the best trial from previous generation
best = trials.filter(success=True, sort_by="-score", limit=1)[0]
results = spawn_children([{
    "prompt": f"Improve this circle packing (score={best.score}):\\n{best.code}\\nTry to increase the sum of radii.",
    "parent_id": best.trial_id
}])

# Combine two approaches
t1, t2 = trials.filter(success=True, sort_by="-score", limit=2)
spawn_children([{
    "prompt": f"Combine these two approaches:\\n# Approach 1 ({t1.score}):\\n{t1.code}\\n\\n# Approach 2 ({t2.score}):\\n{t2.code}"
}])
```

### query_llm(queries: list[dict]) -> list[dict]
Query child LLMs for analysis without code evaluation or trial records. Use this to:
- **Compare trials**: "Why does trial_0_5 score higher than trial_0_3?"
- **Understand methodology**: "What optimization technique is used in this code?"
- **Explore diversity**: "How do these two approaches differ conceptually?"
- **Plan strategy**: "Given these results, what should I try next?"
- **Find patterns**: "What do the top 5 trials have in common?"

Each query dict has:
- `prompt` (str, required) - Use `trials["trial_X_Y"].code` to include code in your prompt
- `model` (str, optional) - alias from available child LLMs
- `temperature` (float, optional, default 0.7)

Returns list of dicts with: model, prompt, response, temperature, success, error.

Examples:
```python
# Analyze top performing trials
top = trials.filter(success=True, sort_by="-score", limit=3)
analysis = query_llm([{
    "prompt": f"Compare these approaches and identify what makes the best one work:\\n" +
              "\\n---\\n".join(f"# {t.trial_id} (score={t.score})\\n{t.code}" for t in top),
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])

# Ask about a specific trial
best = trials.filter(success=True, sort_by="-score", limit=1)[0]
query_llm([{"prompt": f"What optimization technique does this use?\\n{best.code}"}])

# Compare all trials of a generation
generation = trials.filter(generation=0)
analysis = query_llm([{
    "prompt": f"Compare all trials of generation 0 and identify what makes the best ones work vs the rest:\\n" +
              "\\n---\\n".join(f"# {t.trial_id} (score={t.score})\\n{t.code}" for t in generation),
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])

# Find diversity in a generation
generation = trials.filter(generation=0)
analysis = query_llm([{
    "prompt": f"Look throug the various approaches and identify which ones may not be the best, but may be interesting to try again in the future:\\n" +
              "\\n---\\n".join(f"# {t.trial_id} (score={t.score})\\n{t.code}" for t in generation),
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])

# Identify bugs in a trial
trial = trials["trial_0_5"]
analysis = query_llm([{
    "prompt": f"Identify bugs in the following code:\\n{trial.code}",
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])
```

### `scratchpad` - Persistent notes

A mutable scratchpad for tracking insights across generations:
- `scratchpad.content` - Read current content
- `scratchpad.content = "New notes"` - Replace content (auto-persists)
- `scratchpad.append("\n## New section")` - Append content
- `scratchpad.clear()` - Clear all content
- `print(scratchpad)` - Print current content
- `"grid" in scratchpad` - Check if text exists
- `len(scratchpad)` - Get character count

The scratchpad is shown in Evolution Memory and persists across generations.

### update_scratchpad(content: str) -> dict
Alternative function to update persistent notes (same as `scratchpad.content = content`).

### terminate_evolution(reason: str, best_program: str = None) -> dict
End evolution early.

## Evolution Flow

1. (Optional) Run analysis in one or more ```python``` blocks
2. When ready, spawn children using the `spawn_children` function with diverse prompts
2. Using the trials variable do some analysis on the results in order to inform your strategy. We recommend at lookingings at all of the results. As the contents of trials may be large, you can use the `query_llm` function to analyze the results. Note you have access and are encouraged to consider all past trials not just the past generation.
3. After spawning, you SELECT which trials to carry forward (performance, diversity, potential)
3. Repeat until max_generations or you call terminate_evolution()

## Selection Format

```selection
{
  "selections": [
    {"trial_id": "trial_0_2", "reasoning": "Best score", "category": "performance"},
    {"trial_id": "trial_0_5", "reasoning": "Different approach", "category": "diversity"}
  ],
  "summary": "Brief summary"
}
```

## Historical Trial Access

**You can access and mutate ANY historical trial**, not just those from the current generation:
- Use the `trials` variable: `trials["trial_0_5"].code` to retrieve code from any past trial

## Custom Analysis Functions

You can define helper functions in Python that persist throughout this evolution session:

```python
# Define a reusable analysis function
def compute_score_stats(scores_list):
    """Compute statistics for a list of scores."""
    import statistics
    return {
        "mean": statistics.mean(scores_list),
        "max": max(scores_list),
        "min": min(scores_list),
        "stdev": statistics.stdev(scores_list) if len(scores_list) > 1 else 0
    }

# Store results for later analysis
generation_bests = []  # Persists across code executions

# Track best scores as you go
generation_bests.append(2.61)  # After gen 0
generation_bests.append(2.62)  # After gen 1
print(compute_score_stats(generation_bests))
```

Available modules: math, random, json, numpy, scipy, collections, itertools, functools, statistics
Functions and variables you define persist across all generations within this run.

## REPL Variables

### `trials` - Query all trials
A live view of all trials across all generations. Use this for flexible analysis.

**Basic access:**
```python
trials["trial_0_5"]  # Get specific trial by ID
len(trials)          # Total trial count
for t in trials: ... # Iterate all trials
"trial_0_5" in trials  # Check if trial exists
```

**Filtering with `trials.filter()`:**
```python
# Top 5 by score
trials.filter(success=True, sort_by="-score", limit=5)

# All from generation 2
trials.filter(generation=2)

# Custom predicate (lambda)
trials.filter(predicate=lambda t: t.score > 2.4 and "grid" in t.reasoning)

# All descendants of a trial
trials.filter(descendant_of="trial_0_3")

# All ancestors of a trial
trials.filter(ancestor_of="trial_2_5")

# Combined: top 3 successful from gen 1
trials.filter(success=True, generation=1, sort_by="-score", limit=3)

# Filter by model
trials.filter(model_alias="fast", success=True)
```

**Filter parameters:**
- `success`: bool - Filter by success/failure
- `generation`: int - Filter by generation number
- `parent_id`: str - Filter by direct parent
- `model_alias`: str - Filter by child LLM model
- `descendant_of`: str - All trials descending from this trial
- `ancestor_of`: str - All ancestors of this trial
- `predicate`: lambda - Custom filter function
- `sort_by`: str - Sort field ("-score" for descending)
- `limit`: int - Max results to return

**Trial attributes:**
`.trial_id`, `.code`, `.score`, `.success`, `.generation`,
`.parent_id`, `.reasoning`, `.error`, `.model_alias`, `.metrics`

**Return values from spawn_children:**
`spawn_children()` returns TrialView objects (same attributes as above).
Use `.to_dict()` if you need dict format.

## Guidelines

**You have full control**: Craft prompts however you see fit - be as specific or open-ended as you want. You're the orchestrator.

**Historical selection is allowed**: You may select any trial_id from any generation (not just the current one).

**Track lineage**: When a child is based on an existing trial, set `parent_id` to that trial_id (choose the primary parent if there are multiple influences).

**Diversity matters**: Especially in early generations, try fundamentally different approaches rather than minor variations of the same idea. If you feel like you are platueing, use diverse tests as well.

**Learn from results**: Use scores and patterns you observe to guide your strategy. If an approach is working, refine it. If you're stuck, try something radically different.

**Use query_llm for analysis**: When you want to understand *why* something works, compare approaches, or get strategic advice, use `query_llm(queries: list[dict])` to analyze trials. This helps you make more informed decisions about what to try next.

**Exploration and Exploitation**: You are encouraged to explore different approaches and to exploit the best ones. Given the amount of trials you are encouraged to allocate some for exploration and exploring ideas with potential or new ideas, and others for refining the best solution(s).
'''

# Dynamic suffix template - appended after the static prefix
ROOT_LLM_SYSTEM_PROMPT_DYNAMIC = """
## Current Run Parameters

- **Max children per generation**: {max_children_per_generation}
- **Max generations**: {max_generations}
- **Current generation**: {current_generation}/{max_generations}
"""

# Available child LLMs template - inserted after dynamic parameters
ROOT_LLM_CHILD_MODELS_TEMPLATE = """
## Available Child LLMs

{child_llm_list}

**Default model**: {default_child_llm}
"""

# Timeout constraint - simple exposure of the limit
ROOT_LLM_TIMEOUT_CONSTRAINT = """
- **Timeout per trial**: {timeout_seconds}s
"""

# Child LLM System Prompt - Static (cacheable)
# Minimal prompt to give child LLMs freedom to explore.
CHILD_LLM_SYSTEM_PROMPT = """You are an expert algorithm designer.

## Task

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

Constraints:
- All circles entirely inside [0,1] x [0,1]
- No overlaps (distance between centers ≥ sum of radii)
- All radii ≥ 0

The best known solution is 2.6359850561146603. Aim to achieve as high of a score as possible.

## Output

Provide a Python solution with:
```python
import numpy as np

def construct_packing():
    # Return (centers, radii, sum_radii)
    # centers: np.array shape (26, 2)
    # radii: np.array shape (26,)
    pass

def run_packing():
    return construct_packing()
```

You may use numpy, scipy, or any standard library. No plotting or printing.
"""


def _format_timeout_constraint(timeout_seconds: int | None) -> str:
    """
    Format timeout constraint for the system prompt.

    Args:
        timeout_seconds: Timeout limit in seconds, or None to omit

    Returns:
        Formatted timeout constraint string, or empty string if None
    """
    if timeout_seconds is None:
        return ""

    return ROOT_LLM_TIMEOUT_CONSTRAINT.format(timeout_seconds=timeout_seconds)


def get_root_system_prompt(
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
    timeout_seconds: int | None = None,
) -> str:
    """
    Get the Root LLM system prompt with configuration values.

    Args:
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)
        timeout_seconds: Optional timeout limit per trial in seconds

    Returns:
        Formatted system prompt string (static + dynamic parts combined)
    """
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    timeout_part = _format_timeout_constraint(timeout_seconds)
    return ROOT_LLM_SYSTEM_PROMPT_STATIC + dynamic_part + timeout_part


def get_root_system_prompt_parts(
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
    timeout_seconds: int | None = None,
) -> list[dict]:
    """
    Get the Root LLM system prompt as structured content blocks for caching.

    The static prefix is marked with cache_control for prompt caching.
    The dynamic suffix contains run-specific parameters and timeout constraint.

    Args:
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)
        timeout_seconds: Optional timeout limit per trial in seconds

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
        The first block (static) has cache_control set.
    """
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    timeout_part = _format_timeout_constraint(timeout_seconds)

    return [
        {
            "type": "text",
            "text": ROOT_LLM_SYSTEM_PROMPT_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_part + timeout_part,
        },
    ]


def format_child_mutation_prompt(
    parent_code: str,
    parent_score: float,
    guidance: str = "",
) -> str:
    """
    Format a prompt for mutating a parent program.

    This is a helper function - the Root LLM can use this or craft its own.
    Note: The Root LLM is encouraged to write its own shorter, more creative prompts.

    Args:
        parent_code: The parent program code
        parent_score: The parent's score (sum of radii)
        guidance: Optional high-level guidance for mutation

    Returns:
        Formatted prompt string
    """
    prompt = f"""Improve this circle packing (current score: {parent_score:.16f}. It is possible to achieve a score of at least2.6359850561146603).

```python
{parent_code}
```

{guidance if guidance else "Find a way to improve the score."}
"""
    return prompt


def _format_child_llm_list(child_llm_configs: dict[str, "ChildLLMConfig"]) -> str:
    """Format the list of available child LLMs for the system prompt."""
    lines = []
    for alias, cfg in child_llm_configs.items():
        lines.append(
            f"- **{alias}**: `{cfg.model}` ({cfg.provider}) - "
            f"${cfg.cost_per_million_input_tokens:.2f}/M in, "
            f"${cfg.cost_per_million_output_tokens:.2f}/M out"
        )
    return "\n".join(lines)


def get_root_system_prompt_parts_with_models(
    child_llm_configs: dict[str, "ChildLLMConfig"],
    default_child_llm_alias: str | None = None,
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
    timeout_seconds: int | None = None,
) -> list[dict]:
    """
    Get the Root LLM system prompt with child LLM info as structured content blocks.

    Args:
        child_llm_configs: Dict of alias -> ChildLLMConfig
        default_child_llm_alias: Default model alias
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)
        timeout_seconds: Optional timeout limit per trial in seconds

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
    """
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    timeout_part = _format_timeout_constraint(timeout_seconds)

    # Build child LLM info
    child_llm_list = _format_child_llm_list(child_llm_configs)
    default_llm = default_child_llm_alias or (
        list(child_llm_configs.keys())[0] if child_llm_configs else "none"
    )
    child_models_part = ROOT_LLM_CHILD_MODELS_TEMPLATE.format(
        child_llm_list=child_llm_list,
        default_child_llm=default_llm,
    )

    return [
        {
            "type": "text",
            "text": ROOT_LLM_SYSTEM_PROMPT_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_part + timeout_part + child_models_part,
        },
    ]


# Calibration system prompt - used during calibration phase
CALIBRATION_SYSTEM_PROMPT_STATIC = """You are orchestrating a calibration phase for an evolutionary optimization process.

## Purpose

Before evolution begins, you have the opportunity to test the available child LLMs to understand their capabilities. **You can send ANY prompt you want** - not just circle packing tasks. Use this to evaluate:

- Reasoning depth and quality (ask them to explain their approach)
- Code style and correctness (test with simple problems first)
- Mathematical reasoning (geometry, optimization concepts)
- Instruction following (give specific constraints)
- Creativity vs precision tradeoffs at different temperatures
- How they handle ambiguous or open-ended prompts

The goal is to understand what each model is good at so you can use them strategically during evolution.

## Problem (for reference)

The main task is packing 26 circles into a unit square [0,1] x [0,1] to maximize the sum of radii.
- The best known solution is 2.6359850561146603. Aim to achieve as high of a score as possible.

## How to Call Functions

**IMPORTANT**: Write Python code in ```python blocks to call functions. Example:

```python
query_llm([
    {"prompt": "Explain quicksort", "model": "sonnet", "temperature": 0.5},
    {"prompt": "What is 2+2?", "model": "gpt41", "temperature": 0.3}
])
```

## Available Functions

### query_llm(queries: list[dict]) -> list[dict]
Query child LLMs with ANY prompts (no code evaluation - just get responses). Each dict has:
- `prompt` (str, required) - Any question or task, not limited to circle packing!
- `model` (str, optional) - alias from available child LLMs
- `temperature` (float, optional, default 0.7)

Returns list of dicts with: model, prompt, response, temperature, success, error.

Example:
```python
results = query_llm([
    {"prompt": "Explain gradient descent in 2 sentences", "model": "sonnet"},
    {"prompt": "Write a Python function to compute factorial", "model": "gpt41"}
])
for r in results:
    print(f"{r['model']}: {r['response'][:200]}")
```

### get_calibration_status() -> dict
Check remaining calibration calls per model.

```python
get_calibration_status()
```

### update_scratchpad(content: str) -> dict
Record your observations about each model's behavior. These notes will persist into evolution.

```python
update_scratchpad(\"\"\"
## Model Observations
- sonnet: Strong reasoning, verbose responses
- gpt41: Concise, good at math
\"\"\")
```

### end_calibration_phase() -> dict
Finish calibration and begin the evolution phase. Call this when you've learned enough.

```python
end_calibration_phase()
```

## Guidelines

1. **Ask diverse questions**: Test reasoning, math, code quality - not just circle packing. Your notes should be generic and not specific to the circle packing task.
2. **Compare models**: Give the same prompt to different models to compare their responses
3. **Experiment with temperatures**: Generally 0 is considered the most focused / reproducible, 1 is the most creative.
4. **Record detailed observations**: Note strengths/weaknesses of each model
5. **Be strategic**: Your notes will guide which model you choose for different tasks during evolution
"""


def get_calibration_system_prompt_parts(
    child_llm_configs: dict[str, "ChildLLMConfig"],
) -> list[dict]:
    """
    Get the calibration system prompt as structured content blocks.

    Args:
        child_llm_configs: Dict of alias -> ChildLLMConfig

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
    """
    # Build child LLM info with calibration budgets
    lines = ["## Available Child LLMs", ""]
    for alias, cfg in child_llm_configs.items():
        lines.extend(
            [
                f"### {alias}",
                f"- **Model**: `{cfg.model}`",
                f"- **Provider**: {cfg.provider}",
                f"- **Calibration budget**: {cfg.calibration_calls} calls",
                f"- **Cost**: ${cfg.cost_per_million_input_tokens:.2f}/M input, "
                f"${cfg.cost_per_million_output_tokens:.2f}/M output",
                "",
            ]
        )

    child_models_part = "\n".join(lines)

    return [
        {
            "type": "text",
            "text": CALIBRATION_SYSTEM_PROMPT_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": child_models_part,
        },
    ]
