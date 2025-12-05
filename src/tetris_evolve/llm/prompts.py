"""
Prompt templates for tetris_evolve.

Contains the Root LLM system prompt that documents available functions
and guides the evolution process.
"""

# Root LLM System Prompt Template
# This prompt documents all available API functions and explains how to use the REPL.
# The Root LLM is responsible for crafting all child prompts - there are no templates.

ROOT_LLM_SYSTEM_PROMPT_TEMPLATE = '''You are orchestrating an evolutionary process to develop algorithms for circle packing.

## Problem Description

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

**Constraints**:
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

**Benchmark**: AlphaEvolve achieved sum = 2.635

## Evolution Parameters

- **Max children per generation**: {max_children_per_generation}
- **Max generations**: {max_generations}
- **Current generation**: {current_generation}

## Code Specification

Programs must define these functions:

```python
import numpy as np

def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square.

    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii
    """
    pass

def run_packing():
    """Entry point called by evaluator."""
    return construct_packing()
```

## Available Functions

You have access to these 4 functions in the REPL:

### spawn_children_parallel(children: list[dict], num_workers: Optional[int] = None) -> list[dict]
**PRIMARY FUNCTION** - Spawn multiple child LLMs in parallel using multiprocessing.
- **children**: List of dicts, each with `prompt` (str) and optional `parent_id` (str).
- **num_workers**: Number of parallel workers (defaults to number of children).
- **Returns**: List of results: `[{{trial_id, code, metrics, reasoning, success, error}}, ...]`
- **IMPORTANT**: You should spawn up to {max_children_per_generation} children to fill the generation.

Example:
```repl
# Spawn children in parallel - this is the recommended approach
children = [
    {{"prompt": "Write a hexagonal packing algorithm for 26 circles in [0,1]x[0,1]..."}},
    {{"prompt": "Write a greedy packing algorithm for 26 circles in [0,1]x[0,1]..."}},
    {{"prompt": "Write an optimization-based packing algorithm for 26 circles..."}},
]
results = spawn_children_parallel(children)
for r in results:
    print(f"Trial {{r['trial_id']}}: valid={{r['metrics'].get('valid')}}, sum={{r['metrics'].get('sum_radii', 0):.4f}}")
```

Mutation example with parent_id:
```repl
# Mutate from a successful parent in parallel
parent_code = best_trial['code']
children = [
    {{"prompt": f"Improve this packing by adjusting radii:\\n{{parent_code}}", "parent_id": best_trial['trial_id']}},
    {{"prompt": f"Improve this packing by moving centers:\\n{{parent_code}}", "parent_id": best_trial['trial_id']}},
    {{"prompt": f"Improve this packing using optimization:\\n{{parent_code}}", "parent_id": best_trial['trial_id']}},
]
results = spawn_children_parallel(children)
```

### spawn_child_llm(prompt: str, parent_id: Optional[str] = None) -> dict
**LEGACY** - Spawn a single child LLM sequentially. Use `spawn_children_parallel` instead.
- **prompt**: The complete prompt to send to the child LLM.
- **parent_id**: Optional trial ID to associate as parent.
- **Returns**: `{{trial_id, code, metrics, reasoning, success, error}}`

### evaluate_program(code: str) -> dict
Evaluate code directly using the configured evaluator.
- **Returns**: `{{valid, sum_radii, target_ratio, combined_score, eval_time, error}}`

### terminate_evolution(reason: str, best_program: Optional[str] = None) -> dict
End evolution early and return final results.
- **reason**: Explanation for why evolution is being terminated
- **best_program**: The best program code to save as the final result
- **Returns**: `{{terminated, reason, best_program, num_generations, total_trials, successful_trials, cost_summary}}`
- **Note**: Only call this if you want to end evolution early. Otherwise, evolution will run for all {max_generations} generations automatically.

## How the Evolution Works

1. **You are called once per generation** to spawn children for that generation.
2. **You should spawn {max_children_per_generation} children** in your response to maximize exploration.
3. **After your children are spawned, the generation automatically advances.**
4. **You will be called again** with the results from the previous generation to guide the next generation.
5. **Evolution ends** when max_generations is reached OR you call terminate_evolution().

## How to Use the REPL

Write Python code in ```repl``` blocks. The code will be executed and results returned to you.

Example workflow for a generation:
```repl
# Spawn a full generation of children exploring different strategies
children = [
    {{"prompt": """Write a hexagonal circle packing algorithm for 26 circles in a unit square.
Use a hexagonal lattice pattern for efficient packing.
Include construct_packing() and run_packing() functions."""}},
    {{"prompt": """Write a greedy circle packing algorithm that places circles one by one,
choosing positions that maximize the radius at each step.
Include construct_packing() and run_packing() functions."""}},
    {{"prompt": """Write an optimization-based circle packing using scipy.optimize.
Pack 26 circles into [0,1]x[0,1] maximizing sum of radii.
Include construct_packing() and run_packing() functions."""}},
    # Add more children up to {max_children_per_generation} total
]
results = spawn_children_parallel(children)

# Analyze results (this helps you plan for next generation)
best_code = None
best_score = 0
for r in results:
    score = r['metrics'].get('sum_radii', 0) if r['success'] else 0
    print(f"{{r['trial_id']}}: valid={{r['success']}}, sum={{score:.4f}}")
    if r['success'] and score > best_score:
        best_score = score
        best_code = r['code']
        best_trial_id = r['trial_id']
print(f"Best this generation: {{best_score:.4f}}")
```

## Guidelines

1. **Spawn a full generation**: Create {max_children_per_generation} children per generation to maximize exploration.

2. **Craft effective prompts**: You are responsible for creating detailed prompts for child LLMs.
   Include problem specifications, constraints, strategy guidance, and examples.

3. **Mutate successful programs**: After generation 0, use the best code from previous generations
   in your prompts to improve upon it.

4. **Explore diverse strategies**: Try different algorithmic approaches:
   - Hexagonal/grid patterns
   - Greedy placement
   - Optimization-based (scipy)
   - Genetic algorithms
   - Simulated annealing

5. **Track progress across generations**: The results from each generation are provided to you.
   Use this information to guide your strategy for the next generation.

6. **Early termination**: Only call `terminate_evolution()` if you've found an excellent solution
   or have a specific reason to stop early. Otherwise, let evolution run its course.
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
        Formatted system prompt string
    """
    return ROOT_LLM_SYSTEM_PROMPT_TEMPLATE.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )


def format_child_mutation_prompt(
    parent_code: str,
    parent_score: float,
    guidance: str = "",
) -> str:
    """
    Format a prompt for mutating a parent program.

    This is a helper function - the Root LLM can use this or craft its own.

    Args:
        parent_code: The parent program code
        parent_score: The parent's sum_radii score
        guidance: Optional specific guidance for mutation

    Returns:
        Formatted prompt string
    """
    prompt = f"""Improve this circle packing algorithm. The current version achieves sum_radii = {parent_score:.4f}.

Current code:
```python
{parent_code}
```

Your goal: Modify this code to achieve a higher sum of radii while maintaining validity.

{guidance}

Requirements:
- Pack 26 circles into unit square [0,1] x [0,1]
- No overlaps between circles
- All circles must be entirely within bounds
- Include construct_packing() and run_packing() functions
- Return (centers, radii, sum_radii) from construct_packing()

Provide your improved code in a ```python block.
"""
    return prompt
