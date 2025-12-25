# LLM-Evolve Design Document: Circle Packing

## Overview

This document describes the architecture for an evolutionary code generation system that uses LLMs to iteratively develop circle packing algorithms. The system combines ideas from:

1. **AlphaEvolve** (DeepMind): Evolutionary algorithm for program generation using LLMs
2. **Recursive LLMs (RLM)**: Hierarchical LLM spawning for complex decision-making

**Target Problem**: Pack n circles (default: 26) into a unit square to maximize the sum of their radii.

- **Benchmark**: AlphaEvolve achieved sum = 2.635 for n=26
- **Evaluation**: Geometric validity (no overlaps, within bounds) + sum of radii

---

## Why Circle Packing?

Circle packing is an ideal test problem because:

1. **Deterministic evaluation**: No randomness, same code always produces same result
2. **Clear objective**: Maximize sum of radii
3. **Easy to validate**: Geometric constraints are simple to check
4. **Known benchmark**: AlphaEvolve's 2.635 gives a target to aim for
5. **Fast evaluation**: Typically < 1 second per trial
6. **Rich strategy space**: Grid, hexagonal, greedy, optimization-based approaches

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    ROOT LLM (REPL)                          │ │
│  │                                                              │ │
│  │  Available Functions:                                        │ │
│  │  - spawn_child_llm(prompt, parent_id) -> trial_result       │ │
│  │  - evaluate_program(code) -> metrics                        │ │
│  │  - advance_generation(trial_ids, reasoning) -> gen_num      │ │
│  │  - terminate_evolution(reason) -> final_result              │ │
│  │  - get_best_trials(n) -> list[trial]                        │ │
│  │  - get_cost_remaining() -> float                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    CHILD LLM                                │ │
│  │  Input: Prompt describing strategy to try                   │ │
│  │  Output: Python code with construct_packing() function      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              CIRCLE PACKING EVALUATOR                       │ │
│  │  - Executes code in isolated subprocess                     │ │
│  │  - Validates: no overlaps, within unit square               │ │
│  │  - Returns: score, validity                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    LOGGING                                  │ │
│  │  experiment.json, generations/, cost_tracking.json          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Specification

Child LLMs must generate code that defines:

```python
def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square.

    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii

    Constraints:
        - All circles must be inside [0, 1] x [0, 1]
        - No two circles may overlap
        - Radii must be non-negative
    """
    pass

def run_packing():
    """Entry point called by evaluator."""
    return construct_packing()
```

---

## Evaluation Metrics

The evaluator returns:

```python
{
    "valid": bool,           # True if packing satisfies all constraints
    "score": float,          # Sum of all radii (0 if invalid)
    "eval_time": float,      # Seconds to evaluate
    "error": Optional[str],  # Error message if any
}
```

**Validity checks**:
1. centers.shape == (26, 2)
2. radii.shape == (26,)
3. No NaN values
4. All radii >= 0
5. All circles within unit square: `0 <= x-r` and `x+r <= 1` (same for y)
6. No overlaps: `distance(i, j) >= radii[i] + radii[j]` for all pairs

---

## Evolution API

### spawn_child_llm(prompt: str, parent_id: Optional[str] = None) -> dict

Spawn a child LLM to generate circle packing code.

**Parameters**:
- `prompt`: Instructions for the child LLM (strategy to try)
- `parent_id`: Optional trial ID to base mutations on

**Returns**:
```python
{
    "trial_id": str,
    "code": str,           # Generated Python code
    "metrics": dict,       # Evaluation results
    "reasoning": str,      # Child's explanation
    "success": bool,       # True if valid packing
    "error": Optional[str]
}
```

### evaluate_program(code: str) -> dict

Evaluate a circle packing program directly.

**Returns**: Same metrics dict as evaluator.

### advance_generation(selected_trial_ids: list[str], reasoning: str) -> int

Move to next generation with selected trials as parents.

**Returns**: New generation number.

### terminate_evolution(reason: str) -> dict

End evolution and return final results.

**Returns**:
```python
{
    "reason": str,
    "total_generations": int,
    "total_trials": int,
    "successful_trials": int,
    "best_trial": dict,    # Best trial's full data
    "total_cost": float,
}
```

### get_best_trials(n: int = 5) -> list[dict]

Get top n trials by score across all generations.

### get_cost_remaining() -> float

Get remaining budget in USD.

---

## Configuration

```yaml
experiment:
  name: "circle_packing_001"
  output_dir: "./experiments"

root_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0
  max_iterations: 30

child_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0

evolution:
  max_generations: 10
  max_children_per_generation: 10

budget:
  max_total_cost: 20.0

evaluation:
  n_circles: 26
  target_sum: 2.635
  timeout_seconds: 30
```

---

## Example Root LLM Session

```python
# Generation 0: Explore diverse strategies
strategies = [
    "grid-based packing with uniform spacing",
    "hexagonal close-packing pattern",
    "greedy placement maximizing each radius",
    "concentric rings from center outward",
    "corner-first with varying sizes",
]

results = []
for strategy in strategies:
    trial = spawn_child_llm(f"Implement {strategy} for 26 circles in unit square")
    results.append(trial)
    print(f"{trial['trial_id']}: score={trial['metrics']['score']:.4f}")

# Analyze and select
best = get_best_trials(3)
print(f"Best strategies: {[t['trial_id'] for t in best]}")

# Advance to generation 1
advance_generation([t['trial_id'] for t in best], "Selected top 3 performers")

# Generation 1: Refine best approaches
for parent in best:
    # Mutate successful strategies
    spawn_child_llm(
        f"Improve this approach: {parent['reasoning'][:200]}. "
        "Try numerical optimization to fine-tune positions.",
        parent_id=parent['trial_id']
    )

# Check progress
current_best = get_best_trials(1)[0]
print(f"Current best: {current_best['metrics']['score']:.4f}")

# Terminate when satisfied or budget low
if get_cost_remaining() < 2.0:
    terminate_evolution("Budget low")
```

---

## Child LLM Prompt Template

```
You are tasked with writing a circle packing algorithm.

## Problem
Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

## Constraints
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

## Function Specification

```python
import numpy as np

def construct_packing():
    """
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26,) with radius of each circle
        sum_radii: float - sum of all radii
    """
    pass

def run_packing():
    return construct_packing()
```

## Strategy Guidance
{user_prompt}

{parent_context}

## Requirements
1. Code must be self-contained (imports at top)
2. Must define run_packing() or construct_packing()
3. Must return valid numpy arrays of correct shape
4. Use numpy (available as np)

Provide your implementation in a Python code block.
```

---

## Proven Strategies (from PoC)

Our proof-of-concept testing showed these results:

| Strategy | Valid | Sum of Radii | Target Ratio |
|----------|-------|--------------|--------------|
| Hexagonal | Yes | 2.08 | 0.79 |
| Grid | Yes | 1.77 | 0.67 |
| Corner-first | Yes | 1.65 | 0.63 |
| Concentric | Yes | 1.08 | 0.41 |
| Greedy | No | - | - |

**Key insights**:
- Hexagonal packing performs best among simple strategies
- Greedy approaches can fail validation due to numerical issues
- Optimization-based approaches (scipy) have potential but need careful implementation

---

## Implementation TODO

### Phase 1: Core (Completed in PoC)
- [x] Circle packing evaluator
- [x] REPL environment
- [x] Cost tracking
- [x] Mock LLM integration

### Phase 2: Production
- [ ] Real LLM client (Anthropic API)
- [ ] Experiment logger (file-based)
- [ ] Configuration loader
- [ ] Main entry point

### Phase 3: Polish
- [ ] Integration tests
- [ ] E2E tests with real LLM
- [ ] Example configurations
- [ ] Documentation

---

## Files

```
src/tetris_evolve/
├── evaluation/
│   └── circle_packing.py      # Evaluator (DONE)
├── config.py                   # Configuration loading
├── cost_tracker.py             # Budget enforcement
├── logger.py                   # Experiment logging
├── repl.py                     # REPL environment
├── evolution_api.py            # Evolution API
├── llm/
│   ├── client.py               # Anthropic wrapper
│   └── prompts.py              # Prompt templates
├── root_llm.py                 # Root orchestrator
└── main.py                     # CLI entry point

experiments/
├── poc_circle_packing_integration.py  # Integration PoC (DONE)
└── ...
```

---

## Appendix: Validation Code

```python
def validate_packing(centers, radii, n_circles=26):
    """
    Validate that circles don't overlap and are inside unit square.
    """
    tolerance = 1e-6

    # Check shapes
    if centers.shape != (n_circles, 2):
        return False, "Invalid centers shape"
    if radii.shape != (n_circles,):
        return False, "Invalid radii shape"

    # Check for NaN
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values detected"

    # Check radii non-negative
    if (radii < 0).any():
        return False, "Negative radii"

    # Check within bounds
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tolerance or x + r > 1 + tolerance:
            return False, f"Circle {i} outside x-bounds"
        if y - r < -tolerance or y + r > 1 + tolerance:
            return False, f"Circle {i} outside y-bounds"

    # Check no overlaps
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - tolerance:
                return False, f"Circles {i} and {j} overlap"

    return True, None
```
