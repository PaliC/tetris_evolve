"""
Prompt templates for mango_evolve.

Contains the Root LLM system prompt that documents available functions
and guides the evolution process. Structured for optimal prompt caching.
"""

# Root LLM System Prompt - Static Prefix (cacheable)
# This contains all the stable content that doesn't change between calls.
# Dynamic values (generation counts) are appended separately.

ROOT_LLM_SYSTEM_PROMPT_STATIC = '''You are orchestrating an evolutionary process to develop algorithms for circle packing.

## Problem

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

**Target**: 2.635983099011548 (best known)

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

### spawn_children_parallel(children: list[dict]) -> list[dict]
Spawn child LLMs in parallel. Each child dict has `prompt` (str) and optional `parent_id` (str).
Returns: `[{trial_id, code, metrics, reasoning, success, error}, ...]`

### get_trial_code(trial_ids: list[str]) -> dict[str, str | None]
Get code from previous trials by ID.

### update_scratchpad(content: str) -> dict
Update your persistent notes. These are shown at the start of each generation along with the auto-generated lineage map (showing trial ancestry and scores).

### terminate_evolution(reason: str, best_program: str = None) -> dict
End evolution early.

## Code References

Use `{{{{CODE_TRIAL_X_Y}}}}` in prompts to inject code from trial X_Y.
Example: `{{{{CODE_TRIAL_0_3}}}}` becomes the code from trial_0_3.

## Evolution Flow

1. You spawn children with diverse prompts each generation
2. After spawning, you SELECT which trials to carry forward (performance, diversity, potential)
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

## Guidelines

**You have full control**: Craft prompts however you see fit - be as specific or open-ended as you want. You're the orchestrator.

**Diversity matters**: Especially in early generations, try fundamentally different approaches rather than minor variations of the same idea.

**Learn from results**: Use scores and patterns you observe to guide your strategy. If an approach is working, refine it. If you're stuck, try something radically different.
'''

# Dynamic suffix template - appended after the static prefix
ROOT_LLM_SYSTEM_PROMPT_DYNAMIC = '''
## Current Run Parameters

- **Max children per generation**: {max_children_per_generation}
- **Max generations**: {max_generations}
- **Current generation**: {current_generation}/{max_generations}
'''

# Child LLM System Prompt - Static (cacheable)
# Minimal prompt to give child LLMs freedom to explore.
CHILD_LLM_SYSTEM_PROMPT = '''You are an expert algorithm designer.

## Task

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

Constraints:
- All circles entirely inside [0,1] x [0,1]
- No overlaps (distance between centers ≥ sum of radii)
- All radii ≥ 0

Target: sum of radii ≈ 2.635983099011548

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
'''


def get_root_system_prompt(
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
) -> str:
    """
    Get the Root LLM system prompt with configuration values.

    Args:
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)

    Returns:
        Formatted system prompt string (static + dynamic parts combined)
    """
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    return ROOT_LLM_SYSTEM_PROMPT_STATIC + dynamic_part


def get_root_system_prompt_parts(
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
) -> list[dict]:
    """
    Get the Root LLM system prompt as structured content blocks for caching.

    The static prefix is marked with cache_control for prompt caching.
    The dynamic suffix contains run-specific parameters.

    Args:
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
        The first block (static) has cache_control set.
    """
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )

    return [
        {
            "type": "text",
            "text": ROOT_LLM_SYSTEM_PROMPT_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_part,
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
    prompt = f"""Improve this circle packing (current score: {parent_score:.16f}, target: 2.635983099011548).

```python
{parent_code}
```

{guidance if guidance else "Find a way to improve the score."}
"""
    return prompt
