"""
Prompt templates for tetris_evolve.

Contains the Root LLM system prompt that documents available functions
and guides the evolution process. Structured for optimal prompt caching.
"""

# Root LLM System Prompt - Static Prefix (cacheable)
# This contains all the stable content that doesn't change between calls.
# Dynamic values (generation counts) are appended separately.

ROOT_LLM_SYSTEM_PROMPT_STATIC = '''You are orchestrating an evolutionary process to develop algorithms for circle packing.

## Problem Description

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

**Constraints**:
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

**Benchmark**: So far the upper bound is 2.63597770931127

## Referencing Code from Previous Trials

You can reference code from any previous trial in your prompts using the token format:
```
{{{{CODE_TRIAL_X_Y}}}}
```
Where X is the generation number and Y is the trial number within that generation.

**Examples:**
- `{{{{CODE_TRIAL_0_3}}}}` - Code from generation 0, trial 3 (trial_0_3)
- `{{{{CODE_TRIAL_1_2}}}}` - Code from generation 1, trial 2 (trial_1_2)

When you include these tokens in a child LLM prompt, they will be automatically replaced with the actual code from that trial. This is useful for:
- Having child LLMs improve upon specific previous solutions
- Combining approaches from different trials
- Providing working examples for child LLMs to learn from

**Example usage in a prompt:**
```repl
children = [
    {{"prompt": """Improve this circle packing algorithm. Here is the code to improve:

{{{{CODE_TRIAL_0_3}}}}

Make the following improvements:
1. Optimize the radius calculation
2. Add local search refinement
""", "parent_id": "trial_0_3"}},
]
results = spawn_children_parallel(children)
```

If a referenced trial does not exist, the token will be replaced with `[CODE NOT FOUND: trial_X_Y]`.

## Code Specification

Programs must define these functions:

```python
import numpy as np

def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square. This function should not plot or print anything.

    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii
    """
    pass

def run_packing():
    """Entry point called by evaluator. This function should not plot or print anything."""
    return construct_packing()
```

## Available Functions

You have access to these 5 functions in the REPL:

### spawn_children_parallel(children: list[dict], num_workers: Optional[int] = None) -> list[dict]
**PRIMARY FUNCTION** - Spawn multiple child LLMs in parallel using multiprocessing.
- **children**: List of dicts, each with `prompt` (str) and optional `parent_id` (str).
- **num_workers**: Number of parallel workers (defaults to number of children).
- **Returns**: List of results: `[{{trial_id, code, metrics, reasoning, success, error}}, ...]`
- **IMPORTANT**: You should spawn children up to the configured limit to fill the generation.

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
    print(f"Trial {{r['trial_id']}}: valid={{r['metrics'].get('valid')}}, score={{r['metrics'].get('score', 0):.4f}}")
```

Mutation example with parent_id and {{{{CODE_TRIAL_X_Y}}}} token:
```repl
# Mutate from trial_0_3 in parallel using the code reference token
# The {{{{CODE_TRIAL_0_3}}}} token is automatically replaced with the trial's code
children = [
    {{"prompt": """Improve this packing by adjusting radii:

{{{{CODE_TRIAL_0_3}}}}

Focus on: increasing individual radii while maintaining validity.""", "parent_id": "trial_0_3"}},
    {{"prompt": """Improve this packing by moving centers:

{{{{CODE_TRIAL_0_3}}}}

Focus on: repositioning circles for better space utilization.""", "parent_id": "trial_0_3"}},
    {{"prompt": """Improve this packing using optimization:

{{{{CODE_TRIAL_0_3}}}}

Focus on: adding scipy.optimize refinement step.""", "parent_id": "trial_0_3"}},
]
results = spawn_children_parallel(children)
```

### evaluate_program(code: str) -> dict
Evaluate code directly using the configured evaluator.
- **Returns**: `{{valid, score, eval_time, error}}`
  - `score`: Sum of circle radii (higher is better)

### terminate_evolution(reason: str, best_program: Optional[str] = None) -> dict
End evolution early and return final results.
- **reason**: Explanation for why evolution is being terminated
- **best_program**: The best program code to save as the final result
- **Returns**: `{{terminated, reason, best_program, num_generations, total_trials, successful_trials, cost_summary}}`
- **Note**: Only call this if you want to end evolution early. Otherwise, evolution will run for all configured generations automatically.

### get_trial_code(trial_ids: list[str]) -> dict[str, str | None]
Retrieve the code for specific trials by their IDs.
- **trial_ids**: List of trial IDs to retrieve code for. Example: `["trial_0_3", "trial_1_2"]`
- **Returns**: Dictionary mapping trial_id to code string (or None if not found)

Use this function when you need to inspect or analyze specific trial code. For including
code in child LLM prompts, prefer using the `{{{{CODE_TRIAL_X_Y}}}}` token syntax.

Example:
```repl
# Get code from specific trials to analyze
codes = get_trial_code(["trial_0_3", "trial_0_5", "trial_1_2"])
for trial_id, code in codes.items():
    if code:
        print(f"--- {{trial_id}} ---")
        print(code[:500])  # Print first 500 chars
```

## How the Evolution Works

1. **You are called once per generation** to spawn children for that generation.
2. **You should spawn children up to the configured limit** in your response to maximize exploration.
3. **After children spawn, you will be asked to SELECT which trials to carry forward.**
4. **Selection criteria**: Choose as many trials as you want based on your judgement. It can contain only one trial up to all of them if you see them all adding value:
   - **Performance**: High-scoring trials with proven results
   - **Diversity**: Trials using different approaches worth exploring
   - **Potential**: Trials that might improve with refinement, even if current scores are lower
5. **You will be called again** with the results from the previous generation to guide the next generation.
6. **Evolution ends** when max_generations is reached OR you call terminate_evolution().

## Trial Selection Format

When asked to select trials, respond with a ```selection``` block:
```selection
{{
  "selections": [
    {{"trial_id": "trial_0_2", "reasoning": "Highest score of 2.45", "category": "performance"}},
    {{"trial_id": "trial_0_5", "reasoning": "Novel hexagonal approach", "category": "diversity"}},
    {{"trial_id": "trial_0_1", "reasoning": "Optimization approach has room for improvement", "category": "potential"}}
  ],
  "summary": "Selected top performer plus two diverse approaches for exploration"
}}
```

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
    # Add more children to fill the generation
]
results = spawn_children_parallel(children)

# Analyze results (this helps you plan for next generation)
best_code = None
best_score = 0
for r in results:
    score = r['metrics'].get('score', 0) if r['success'] else 0
    print(f"{{r['trial_id']}}: valid={{r['success']}}, score={{score:.4f}}")
    if r['success'] and score > best_score:
        best_score = score
        best_code = r['code']
        best_trial_id = r['trial_id']
print(f"Best this generation: {{best_score:.4f}}")
```

## Guidelines

1. **Spawn a full generation**: Create children up to the configured limit per generation to maximize exploration.

2. **Craft effective prompts**: You are responsible for creating detailed prompts for child LLMs.
   Include problem specifications, constraints, strategy guidance, and examples.

3. **Select promising trials**: When asked for selection, balance performance with diversity.
   Don't just pick the top scorers - also select diverse approaches that might lead to breakthroughs.

4. **Mutate successful programs**: After generation 0, use the `{{{{CODE_TRIAL_X_Y}}}}` token to
   reference code from selected trials. This automatically injects the trial's code into your prompt.
   Example: `{{{{CODE_TRIAL_0_3}}}}` will be replaced with trial_0_3's code.

5. **Explore diverse strategies**: Try different algorithmic approaches:
   - Hexagonal/grid patterns
   - Greedy placement
   - Optimization-based (scipy)
   - Genetic algorithms
   - Simulated annealing

6. **Track progress across generations**: The results from each generation are provided to you.
   Use this information to guide your strategy for the next generation.

7. **Early termination**: Only call `terminate_evolution()` if you've found an excellent solution
   or have a specific reason to stop early. Otherwise, let evolution run its course.
'''

# Dynamic suffix template - appended after the static prefix
ROOT_LLM_SYSTEM_PROMPT_DYNAMIC = '''
## Current Run Parameters

- **Max children per generation**: {max_children_per_generation}
- **Max generations**: {max_generations}
- **Current generation**: {current_generation}/{max_generations}
'''

# Child LLM System Prompt - Static (cacheable)
# This provides consistent problem context and output format for all child LLMs.
CHILD_LLM_SYSTEM_PROMPT = '''You are an expert algorithm designer specializing in circle packing optimization.

## Problem

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

**Constraints**:
- All circles must be entirely inside the unit square (center ± radius within [0,1])
- No two circles may overlap (distance between centers > sum of radii)
- All radii must be non-negative

**Benchmark**: The best known solution achieves sum of radii ≈ 2.635

## Required Output Format

You MUST provide your solution as a Python code block with these exact functions:

```python
import numpy as np

def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square.

    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates of circle centers
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii
    """
    # Your implementation here
    pass

def run_packing():
    """Entry point called by evaluator."""
    return construct_packing()
```

## Guidelines

1. **Always return valid numpy arrays** with correct shapes
2. **Ensure all constraints are satisfied** - the evaluator will reject invalid solutions
3. **Focus on maximizing sum of radii** - this is the optimization objective
4. **Do not include plotting or printing** - the evaluator runs headless
5. **You may use scipy, numpy, and standard library** for optimization

Provide your complete solution in a single ```python code block.
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

    Args:
        parent_code: The parent program code
        parent_score: The parent's score (sum of radii)
        guidance: Optional specific guidance for mutation

    Returns:
        Formatted prompt string
    """
    prompt = f"""Improve this circle packing algorithm. The current version achieves score = {parent_score:.4f}.

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
